#include <lr/render/Features.h>

#include "PassHelpers.h"
#include "ShadowCascades.h"
#include "kernels/PathTrace.h"
#include "kernels/Post.h"
#include "kernels/Shading.h"
#include "kernels/Visibility.h"

#include <lr/render/MaterialModel.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr::render {

namespace {

// Room for this many transparent fragments per pixel on average; more are dropped.
constexpr auto kTransparentFragmentsPerPixel = std::size_t{2};

std::size_t roundUpToPowerOfTwo(std::size_t const value) {
    auto result = std::size_t{1};
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

bool sameCamera(core::CameraDesc const &a, core::CameraDesc const &b) noexcept {
    return a.position == b.position and a.target == b.target and a.up == b.up and a.verticalFov == b.verticalFov
       and a.nearPlane == b.nearPlane and a.farPlane == b.farPlane;
}

// Part of a buffer of words, or all of it (count 0).
struct WordRange {
    BufferRef<tc::uint> buffer;
    std::size_t first = 0;
    std::size_t count = 0;
};

template <tc::uint B>
void bindRange(PassContext &context, tc::BufferBinding<tc::uint, B> &binding, WordRange const &range) {
    auto &buffer = context.buffer(range.buffer);
    if (range.count == 0) {
        binding.attach(buffer);
    } else {
        binding.attach(buffer, range.first, range.count);
    }
    context.device().bindBuffer(binding);
}

// The instances one raster pass draws (see CompactVisible) and its workgroup counts.
struct VisibleSet {
    BufferRef<tc::uvec4> visible;
    BufferRef<tc::DispatchIndirectCommand> arguments;
};

struct CullRequest {
    std::string cullPass;
    std::string compactPass;
    core::mat4 viewProjection{1.0f};
    bool blended = false; // the instances whose material blends, or the others
};

struct CompactRequest {
    BufferRef<tc::uint> triangleCounts;
    VisibleSet drawn;
    tc::uint instanceCount = 0;
};

// K is CompactVisible or CompactVisibleSubgroups, which have the same bindings.
template <class K>
void recordCompaction(PassContext &context, CompactRequest const &request) {
    auto &device = context.device();
    auto const &scene = context.scene();
    auto kernel = K{};
    device.useKernel(kernel);
    bind(device, kernel.instances, *scene.instances);
    bind(device, kernel.meshes, *scene.meshes);
    bind(device, kernel.triangleCounts, context.buffer(request.triangleCounts));
    bind(device, kernel.visible, context.buffer(request.drawn.visible));
    bind(device, kernel.rasterArguments, context.buffer(request.drawn.arguments));
    set(device, kernel.instanceCount, request.instanceCount);
    context.dispatch(kernel, tc::uvec3{1, 1, 1});
}

VisibleSet cullAndCompact(FrameGraphBuilder &graph, ViewContext const &view, CullRequest const &request) {
    auto const instanceCapacity = roundUpToPowerOfTwo(std::max<std::size_t>(view.instanceCount, 1));
    auto const drawn = VisibleSet{graph.createBuffer<tc::uvec4>(request.compactPass + ": visible", instanceCapacity + 1),
                                graph.createBuffer<tc::DispatchIndirectCommand>(request.compactPass + ": arguments", 1)};
    auto const triangleCounts = graph.createBuffer<tc::uint>(request.cullPass + ": triangles", instanceCapacity);
    auto const instanceCount = static_cast<tc::uint>(view.instanceCount);

    auto &cull = graph.addPass(request.cullPass);
    cull.write(triangleCounts);
    cull.record([triangleCounts, instanceCount, viewProjection = request.viewProjection,
                 blended = request.blended ? 1u : 0u](PassContext &context) {
        if (instanceCount == 0) {
            return;
        }
        auto &device = context.device();
        auto const &scene = context.scene();
        auto kernel = kernels::CullInstances{};
        device.useKernel(kernel);
        bind(device, kernel.instances, *scene.instances);
        bind(device, kernel.meshes, *scene.meshes);
        bind(device, kernel.materials, *scene.materials);
        bind(device, kernel.triangleCounts, context.buffer(triangleCounts));
        set(device, kernel.viewProjection, viewProjection);
        set(device, kernel.instanceCount, instanceCount);
        set(device, kernel.blended, blended);
        context.dispatch(kernel, tc::uvec3{groupsFor(instanceCount, 64), 1, 1});
    });

    auto &compact = graph.addPass(request.compactPass);
    compact.read(triangleCounts);
    compact.write(drawn.visible);
    compact.write(drawn.arguments);
    compact.record([compaction = CompactRequest{triangleCounts, drawn, instanceCount},
                    subgroups = view.capabilities.subgroups](PassContext &context) {
        if (subgroups) {
            recordCompaction<kernels::CompactVisibleSubgroups>(context, compaction);
        } else {
            recordCompaction<kernels::CompactVisible>(context, compaction);
        }
    });
    return drawn;
}

// The bindings and uniforms of LR_TRIANGLE_RASTERIZER.
template <class K>
void bindRasterizer(PassContext &context, K &kernel, VisibleSet const &drawn, core::mat4 const &viewProjection,
                    core::ivec2 const size) {
    auto &device = context.device();
    auto const &scene = context.scene();
    bindMaterialTextures(device, kernel, scene.textures);
    bind(device, kernel.instances, *scene.instances);
    bind(device, kernel.vertices, *scene.vertices);
    bind(device, kernel.indices, *scene.indices);
    bind(device, kernel.materials, *scene.materials);
    bind(device, kernel.visible, context.buffer(drawn.visible));
    set(device, kernel.viewProjection, viewProjection);
    set(device, kernel.size, size);
}

struct RasterRequest {
    VisibleSet drawn;
    core::mat4 viewProjection{1.0f};
    core::ivec2 size{0};
    tc::uint mode = 0;
    WordRange depth;
    WordRange triangles;
    tc::uint fragmentCapacity = 0;
};

void recordRaster(PassContext &context, RasterRequest const &request) {
    auto &device = context.device();
    auto kernel = kernels::RasterVisibility{};
    device.useKernel(kernel);
    bindRasterizer(context, kernel, request.drawn, request.viewProjection, request.size);
    bindRange(context, kernel.depth, request.depth);
    bindRange(context, kernel.triangles, request.triangles);
    set(device, kernel.mode, request.mode);
    set(device, kernel.fragmentCapacity, request.fragmentCapacity);
    context.dispatchIndirect(kernel, request.drawn.arguments);
}

// What shading kernels of a view take besides the scene.
struct ShadingView {
    core::mat4 inverseViewProjection{1.0f};
    core::vec4 environment{0.0f};
    core::ivec2 extent{0};
    tc::uint lightCount = 0;
    ShadowOutputs shadows;
};

ShadingView shadingViewOf(ViewContext const &view, ShadowOutputs const &shadows) {
    return ShadingView{view.inverseViewProjection, tc::vec4{view.environment.radiance, 0.0f}, view.extent,
                       view.lightCount, shadows};
}

SurfaceShadingResources surfaceResources(PassContext &context, ShadingView const &view,
                                         BufferRef<tc::uvec4> const visible) {
    auto resources = SurfaceShadingResources{};
    resources.scene = &context.scene();
    resources.visible = &context.buffer(visible);
    resources.shadow = &context.buffer(view.shadows.map);
    resources.inverseViewProjection = view.inverseViewProjection;
    resources.environment = view.environment;
    resources.size = view.extent;
    resources.lightCount = view.lightCount;
    resources.shadowLight = view.shadows.light;
    return resources;
}

// A shadow map without cascades, which shadows nothing.
ShadowOutputs noShadows(FrameGraphBuilder &graph) {
    auto const map = graph.createBuffer<tc::uint>("shadow map", shadowHeaderWords());
    auto &pass = graph.addPass("clear shadow map");
    pass.write(map);
    pass.record([map](PassContext &context) { clearWords(context, map, 0, shadowHeaderWords(), 0u); });
    return ShadowOutputs{map, kernels::noShadowLight()};
}

// The view's shadow map, or one without cascades when no shadow feature ran.
ShadowOutputs shadowsOf(FrameGraphBuilder &graph) {
    auto &board = graph.blackboard();
    if (not board.contains<ShadowOutputs>()) {
        board.put(noShadows(graph));
    }
    return board.get<ShadowOutputs>();
}

struct VisibilityFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Visibility; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const pixels = pixelsOf(view.extent);
        auto const triangles = graph.createBuffer<tc::uint>("visibility", pixels);
        auto const depth = graph.createBuffer<tc::uint>("depth", pixels);
        auto const totals = graph.createReadbackBuffer<tc::uvec2>("visible totals", 1);
        if (not view.capabilities.singlePassVisibility) {
            clearSeparate(graph, depth, triangles, pixels);
        }
        auto const drawn = cullAndCompact(
            graph, view, CullRequest{"cull instances", "compact visible instances", view.viewProjection, false});
        auto const outputs = VisibilityOutputs{drawn.visible, triangles, depth};
        if (view.capabilities.singlePassVisibility) {
            rasterizeOnce(graph, view, drawn, outputs);
        } else {
            rasterizeTwice(graph, view, drawn, outputs);
        }

        auto &stats = graph.addPass("visibility statistics");
        stats.read(drawn.visible);
        stats.write(totals);
        stats.record([visible = drawn.visible, totals](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::CopyVisibleTotals{};
            device.useKernel(kernel);
            bind(device, kernel.visible, context.buffer(visible));
            bind(device, kernel.totals, context.buffer(totals));
            context.dispatch(kernel, tc::uvec3{1, 1, 1});
        });
        graph.readback<tc::uvec2>(totals, [](tc::uvec2 const *data, std::size_t, ViewResult &result) {
            result.stats.visibleInstances = data[0].x;
            result.stats.visibleTriangles = data[0].y;
        });
        graph.blackboard().put(outputs);
    }

private:
    static void clearSeparate(FrameGraphBuilder &graph, BufferRef<tc::uint> const depth,
                              BufferRef<tc::uint> const triangles, std::size_t const pixels) {
        auto &clearPass = graph.addPass("clear visibility");
        clearPass.write(depth);
        clearPass.write(triangles);
        clearPass.record([depth, triangles, pixels](PassContext &context) {
            clearWords(context, depth, 0, pixels, 0xFFFFFFFFu);
            clearWords(context, triangles, 0, pixels, noTriangle());
        });
    }

    // The depth first, then the triangles at the nearest depth.
    static void rasterizeTwice(FrameGraphBuilder &graph, ViewContext const &view, VisibleSet const &drawn,
                               VisibilityOutputs const &outputs) {
        for (auto const mode : {kernels::rasterDepth(), kernels::rasterTriangles()}) {
            auto &raster = graph.addPass(mode == kernels::rasterDepth() ? "rasterize depth" : "rasterize triangles");
            raster.read(drawn.visible);
            raster.read(drawn.arguments);
            raster.read(outputs.depth);
            if (mode == kernels::rasterDepth()) {
                raster.write(outputs.depth);
            } else {
                raster.read(outputs.triangles);
                raster.write(outputs.triangles);
            }
            auto request = RasterRequest{};
            request.drawn = drawn;
            request.viewProjection = view.viewProjection;
            request.size = view.extent;
            request.mode = mode;
            request.depth = WordRange{outputs.depth};
            request.triangles = WordRange{outputs.triangles};
            raster.record([request](PassContext &context) { recordRaster(context, request); });
        }
    }

    // Both at once with 64-bit atomics, then unpacked into the buffers rasterizeTwice()
    // fills, so the passes after see the same.
    static void rasterizeOnce(FrameGraphBuilder &graph, ViewContext const &view, VisibleSet const &drawn,
                              VisibilityOutputs const &outputs) {
        auto const pixels = static_cast<tc::uint>(pixelsOf(view.extent));
        auto const packed = graph.createBuffer<tc::u64>("packed visibility", pixels);

        auto &clearPass = graph.addPass("clear packed visibility");
        clearPass.write(packed);
        clearPass.record([packed, pixels](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::ClearPackedVisibility{};
            device.useKernel(kernel);
            bind(device, kernel.packed, context.buffer(packed));
            set(device, kernel.count, pixels);
            context.dispatch(kernel, tc::uvec3{groupsFor(pixels, 64), 1, 1});
        });

        auto &raster = graph.addPass("rasterize visibility");
        raster.read(drawn.visible);
        raster.read(drawn.arguments);
        raster.read(packed);
        raster.write(packed);
        raster.record([drawn, packed, viewProjection = view.viewProjection, size = view.extent](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::RasterPackedVisibility{};
            device.useKernel(kernel);
            bindRasterizer(context, kernel, drawn, viewProjection, size);
            bind(device, kernel.packed, context.buffer(packed));
            context.dispatchIndirect(kernel, drawn.arguments);
        });

        auto &unpack = graph.addPass("unpack visibility");
        unpack.read(packed);
        unpack.write(outputs.depth);
        unpack.write(outputs.triangles);
        unpack.record([packed, depth = outputs.depth, triangles = outputs.triangles, pixels](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::UnpackVisibility{};
            device.useKernel(kernel);
            bind(device, kernel.packed, context.buffer(packed));
            bind(device, kernel.depth, context.buffer(depth));
            bind(device, kernel.triangles, context.buffer(triangles));
            set(device, kernel.count, pixels);
            context.dispatch(kernel, tc::uvec3{groupsFor(pixels, 64), 1, 1});
        });
    }
};

// The header of a shadow map: see shadowHeaderWords().
std::vector<tc::uint> shadowHeader(std::vector<ShadowCascade> const &cascades, std::uint32_t const resolution) {
    auto header = std::vector<tc::uint>(shadowHeaderWords(), 0u);
    header[0] = static_cast<tc::uint>(cascades.size());
    header[1] = resolution;
    for (auto index = std::size_t{0}; index < cascades.size(); ++index) {
        auto const first = shadowCascadeWord(static_cast<tc::uint>(index));
        auto const &cascade = cascades[index];
        for (auto column = 0; column < 4; ++column) {
            for (auto row = 0; row < 4; ++row) {
                header[first + static_cast<tc::uint>(column * 4 + row)] =
                    tc::floatBitsToUint(cascade.worldToShadow[column][row]);
            }
        }
        header[first + 16] = tc::floatBitsToUint(cascade.texelSize);
        header[first + 17] = tc::floatBitsToUint(cascade.depthScale);
    }
    return header;
}

// Cascade sides are multiples of 16 texels, so every cascade's depths start on a
// multiple of 256 bytes, where storage buffer ranges may start on any device.
std::uint32_t shadowResolution(ShadowSettings const &settings) {
    return std::clamp((settings.resolution + 15u) / 16u * 16u, 16u, 8192u);
}

struct ShadowFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Shadow; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const hasCasters = view.sceneBounds.min.x <= view.sceneBounds.max.x;
        if (not view.shadowLight or not hasCasters) {
            graph.blackboard().put(noShadows(graph));
            return;
        }
        auto request = CascadeRequest{};
        request.camera = view.camera;
        request.aspect = static_cast<float>(view.extent.x) / static_cast<float>(view.extent.y);
        request.lightDirection = view.shadowLight->direction;
        request.sceneBounds = view.sceneBounds;
        request.settings = view.settings.shadows;
        auto const cascades = fitShadowCascades(request);
        auto const resolution = shadowResolution(view.settings.shadows);
        auto const texels = static_cast<std::size_t>(resolution) * resolution;
        auto const depths = cascades.size() * texels;
        auto const map = graph.createBuffer<tc::uint>("shadow map", shadowHeaderWords() + depths);
        auto const header = graph.createUploadBuffer<tc::uint>("shadow map header", shadowHeaderWords());

        auto &clearPass = graph.addPass("clear shadow map");
        clearPass.write(header);
        clearPass.write(map);
        clearPass.record([map, header, words = shadowHeader(cascades, resolution), depths](PassContext &context) {
            auto &device = context.device();
            auto &staged = context.buffer(header);
            std::copy(words.begin(), words.end(), staged.data());
            device.uploadBuffer(staged);
            device.copyBuffer(staged, 0, context.buffer(map), 0, words.size() * sizeof(tc::uint));
            clearWords(context, map, shadowHeaderWords(), depths, 0xFFFFFFFFu);
        });

        for (auto index = std::size_t{0}; index < cascades.size(); ++index) {
            auto const name = std::to_string(index);
            auto const drawn = cullAndCompact(graph, view, CullRequest{"cull shadow casters " + name,
                                                                      "compact shadow casters " + name,
                                                                      cascades[index].viewProjection, false});
            auto &raster = graph.addPass("rasterize shadow cascade " + name);
            raster.read(drawn.visible);
            raster.read(drawn.arguments);
            raster.read(map);
            raster.write(map);
            auto rasterRequest = RasterRequest{};
            rasterRequest.drawn = drawn;
            rasterRequest.viewProjection = cascades[index].viewProjection;
            rasterRequest.size = core::ivec2{static_cast<int>(resolution), static_cast<int>(resolution)};
            rasterRequest.mode = kernels::rasterDepth();
            rasterRequest.depth = WordRange{map, shadowHeaderWords() + index * texels, texels};
            rasterRequest.triangles = rasterRequest.depth;
            raster.record([rasterRequest](PassContext &context) { recordRaster(context, rasterRequest); });
        }
        graph.blackboard().put(ShadowOutputs{map, view.shadowLight->index});
    }
};

// The GPU thread's side of one model's pass.
struct ModelShadingImpl final : ModelShading {
    ModelShadingImpl(PassContext &context_, ModelShadingResources const &frame_) noexcept
        : context(context_), frame(frame_) {}

protected:
    PassContext &pass() noexcept override { return context; }
    ModelShadingResources const &resources() const noexcept override { return frame; }

private:
    PassContext &context;
    ModelShadingResources frame;
};

// The pixels of each material model, in lists the models' kernels run over (see
// modelListEntry()).
struct ModelLists {
    BufferRef<tc::uvec2> pixels;
    BufferRef<tc::DispatchIndirectCommand> arguments;
};

struct ClassifyRequest {
    VisibilityOutputs visibility;
    BufferRef<tc::uint> counts;
    BufferRef<tc::uint> places;
    ImageRef<HdrFormat> color;
    core::ivec2 extent{0};
    core::vec4 environment{0.0f};
    tc::uint modelCount = 0;
};

// K is ClassifyPixels or ClassifyPixelsSubgroups, which have the same bindings.
template <class K>
void recordClassification(PassContext &context, ClassifyRequest const &request) {
    auto &device = context.device();
    auto const &scene = context.scene();
    auto kernel = K{};
    device.useKernel(kernel);
    bind(device, kernel.color, context.image(request.color));
    bind(device, kernel.instances, *scene.instances);
    bind(device, kernel.materials, *scene.materials);
    bind(device, kernel.visible, context.buffer(request.visibility.visible));
    bind(device, kernel.triangles, context.buffer(request.visibility.triangles));
    bind(device, kernel.counts, context.buffer(request.counts));
    bind(device, kernel.places, context.buffer(request.places));
    set(device, kernel.size, request.extent);
    set(device, kernel.environment, request.environment);
    set(device, kernel.modelCount, request.modelCount);
    context.dispatch(kernel, groupsCovering(request.extent));
}

struct ShadingFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Shading; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const visibility = graph.blackboard().get<VisibilityOutputs>();
        auto const shading = shadingViewOf(view, shadowsOf(graph));
        auto const lists = classify(graph, view, visibility);
        for (auto index = std::size_t{0}; index < view.materialModels.size(); ++index) {
            auto const &model = view.materialModels[index];
            auto &pass = graph.addPass("shade " + std::string{model->program()});
            pass.read(lists.pixels);
            pass.read(lists.arguments);
            pass.read(visibility.visible);
            pass.read(shading.shadows.map);
            pass.write(view.sceneColor);
            pass.record([model, lists, shading, visible = visibility.visible, color = view.sceneColor,
                         id = core::MaterialModelId{static_cast<std::uint32_t>(index)}](PassContext &context) {
                auto frame = ModelShadingResources{};
                frame.surface = surfaceResources(context, shading, visible);
                frame.color = &context.image(color);
                frame.pixels = &context.buffer(lists.pixels);
                frame.arguments = lists.arguments;
                frame.model = id;
                auto modelShading = ModelShadingImpl{context, frame};
                model->shade(modelShading);
            });
        }
    }

private:
    // Sky pixels get the environment; the others go into their material model's list.
    static ModelLists classify(FrameGraphBuilder &graph, ViewContext const &view, VisibilityOutputs const &visibility) {
        auto const pixels = pixelsOf(view.extent);
        auto const counts = graph.createBuffer<tc::uint>("material model counts", maxMaterialModels());
        auto const places = graph.createBuffer<tc::uint>("material model places", pixels);
        auto const lists = ModelLists{graph.createBuffer<tc::uvec2>("material model pixels", maxMaterialModels() + pixels),
                                      graph.createBuffer<tc::DispatchIndirectCommand>("material model arguments",
                                                                                      maxMaterialModels())};
        auto const modelCount = static_cast<tc::uint>(view.materialModels.size());

        auto &clearPass = graph.addPass("clear material model counts");
        clearPass.write(counts);
        clearPass.record([counts](PassContext &context) { clearWords(context, counts, 0, maxMaterialModels(), 0u); });

        auto &classifyPass = graph.addPass("classify pixels");
        classifyPass.read(visibility.visible);
        classifyPass.read(visibility.triangles);
        classifyPass.read(counts);
        classifyPass.write(counts);
        classifyPass.write(places);
        classifyPass.write(view.sceneColor);
        auto request = ClassifyRequest{visibility, counts, places, view.sceneColor, view.extent,
                                       tc::vec4{view.environment.radiance, 0.0f}, modelCount};
        classifyPass.record([request, subgroups = view.capabilities.subgroups](PassContext &context) {
            if (subgroups) {
                recordClassification<kernels::ClassifyPixelsSubgroups>(context, request);
            } else {
                recordClassification<kernels::ClassifyPixels>(context, request);
            }
        });

        auto &allocate = graph.addPass("allocate material model lists");
        allocate.read(counts);
        allocate.write(lists.pixels);
        allocate.write(lists.arguments);
        allocate.record([counts, lists, modelCount](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::AllocateModelLists{};
            device.useKernel(kernel);
            bind(device, kernel.counts, context.buffer(counts));
            bind(device, kernel.pixels, context.buffer(lists.pixels));
            bind(device, kernel.arguments, context.buffer(lists.arguments));
            set(device, kernel.modelCount, modelCount);
            context.dispatch(kernel, tc::uvec3{1, 1, 1});
        });

        auto &scatter = graph.addPass("scatter material model pixels");
        scatter.read(visibility.triangles);
        scatter.read(places);
        scatter.read(lists.pixels);
        scatter.write(lists.pixels);
        scatter.record([triangles = visibility.triangles, places, lists,
                        count = static_cast<tc::uint>(pixels)](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::ScatterModelPixels{};
            device.useKernel(kernel);
            bind(device, kernel.triangles, context.buffer(triangles));
            bind(device, kernel.places, context.buffer(places));
            bind(device, kernel.pixels, context.buffer(lists.pixels));
            set(device, kernel.count, count);
            context.dispatch(kernel, tc::uvec3{groupsFor(count, 64), 1, 1});
        });
        return lists;
    }
};

struct TransparencyFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Transparent; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        if (view.transparentInstanceCount == 0) {
            return;
        }
        auto const visibility = graph.blackboard().get<VisibilityOutputs>();
        auto const pixels = pixelsOf(view.extent);
        auto const capacity = pixels * kTransparentFragmentsPerPixel;
        auto const lists = graph.createBuffer<tc::uint>(
            "fragment lists", fragmentWords(static_cast<tc::uint>(pixels)) + 3 * capacity);
        auto const opaque = graph.createImage<HdrFormat>("opaque colour", view.extent);

        auto &clearPass = graph.addPass("clear fragment lists");
        clearPass.write(lists);
        clearPass.record([lists, pixels](PassContext &context) {
            clearWords(context, lists, 0, pixels, noFragment());
            clearWords(context, lists, fragmentCount(static_cast<tc::uint>(pixels)), 1, 0u);
        });

        auto const drawn = cullAndCompact(graph, view,
                                          CullRequest{"cull transparent instances", "compact transparent instances",
                                                      view.viewProjection, true});
        auto &raster = graph.addPass("rasterize transparent");
        raster.read(drawn.visible);
        raster.read(drawn.arguments);
        raster.read(visibility.depth);
        raster.read(lists);
        raster.write(lists);
        auto rasterRequest = RasterRequest{};
        rasterRequest.drawn = drawn;
        rasterRequest.viewProjection = view.viewProjection;
        rasterRequest.size = view.extent;
        rasterRequest.mode = kernels::rasterTransparent();
        rasterRequest.depth = WordRange{visibility.depth};
        rasterRequest.triangles = WordRange{lists};
        rasterRequest.fragmentCapacity = static_cast<tc::uint>(capacity);
        raster.record([rasterRequest](PassContext &context) { recordRaster(context, rasterRequest); });

        auto &copy = graph.addPass("copy opaque colour");
        copy.read(view.sceneColor);
        copy.write(opaque);
        copy.record([color = view.sceneColor, opaque, extent = view.extent](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::CopyHdr{};
            device.useKernel(kernel);
            bind(device, kernel.source, context.image(color));
            bind(device, kernel.target, context.image(opaque));
            set(device, kernel.size, extent);
            context.dispatch(kernel, groupsCovering(extent));
        });

        auto const shading = shadingViewOf(view, shadowsOf(graph));
        auto &blend = graph.addPass("blend transparent");
        blend.read(drawn.visible);
        blend.read(lists);
        blend.read(opaque);
        blend.read(shading.shadows.map);
        blend.write(view.sceneColor);
        blend.record([shading, visible = drawn.visible, lists, opaque, color = view.sceneColor](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::BlendTransparent{};
            device.useKernel(kernel);
            bind(device, kernel.color, context.image(color));
            bind(device, kernel.behind, context.image(opaque));
            bindSurfaceShading(device, kernel, surfaceResources(context, shading, visible));
            bind(device, kernel.fragments, context.buffer(lists));
            context.dispatch(kernel, groupsCovering(shading.extent));
        });
    }
};

struct PathTraceFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Shading; }

    FeatureRequirements requirements() const noexcept override {
        auto requirements = FeatureRequirements{};
        requirements.rayTracing = true;
        return requirements;
    }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const accumulation = graph.history<AccumulationFormat>("path trace accumulation", view.extent);
        auto const continues = continuesAccumulation(view, accumulation.valid);
        auto const total = std::make_shared<std::uint32_t>(0);
        auto &pass = graph.addPass("path trace");
        pass.read(accumulation.previous);
        pass.write(accumulation.current);
        pass.write(view.sceneColor);
        pass.record([accumulation, continues, total, counts = sampleCounts, viewId = view.view, color = view.sceneColor,
                     extent = view.extent, inverseViewProjection = view.inverseViewProjection,
                     environment = tc::vec4{view.environment.radiance, 0.0f},
                     lightCount = static_cast<tc::uint>(view.lightCount)](PassContext &context) {
            auto &device = context.device();
            auto const &scene = context.scene();
            if (scene.tlasNodes == nullptr) {
                throw std::logic_error{"lr::render: the path tracer ran before the scene had hierarchies"};
            }
            auto &count = (*counts)[viewId];
            auto const samples = continues and count.residency == scene.textures.residency ? count.samples : 0u;
            count = SampleCount{samples + 1, scene.textures.residency};
            *total = samples + 1;

            auto kernel = kernels::PathTrace{};
            device.useKernel(kernel);
            bind(device, kernel.previous, context.image(accumulation.previous));
            bind(device, kernel.current, context.image(accumulation.current));
            bind(device, kernel.color, context.image(color));
            bindMaterialTextures(device, kernel, scene.textures);
            bind(device, kernel.instances, *scene.instances);
            bind(device, kernel.vertices, *scene.vertices);
            bind(device, kernel.materials, *scene.materials);
            bind(device, kernel.lights, *scene.lights);
            bind(device, kernel.tlasNodes, *scene.tlasNodes);
            bind(device, kernel.blasNodes, *scene.blasNodes);
            bind(device, kernel.blasTriangles, *scene.blasTriangles);
            bind(device, kernel.blasRoots, *scene.blasRoots);
            set(device, kernel.inverseViewProjection, inverseViewProjection);
            set(device, kernel.environment, environment);
            set(device, kernel.size, extent);
            set(device, kernel.lightCount, lightCount);
            set(device, kernel.tlasNodeCount, scene.tlasNodeCount);
            set(device, kernel.sampleIndex, samples);
            context.dispatch(kernel, groupsCovering(extent));
        });
        graph.onFinish([total](ViewResult &result) { result.stats.accumulatedSamples = *total; });
    }

private:
    // What the samples averaged so far show, on the render thread.
    struct Accumulation {
        core::CameraDesc camera;
        core::ivec2 extent;
        std::uint64_t sceneVersion;
    };
    // The samples averaged so far, and the texture residency they saw, on the GPU thread:
    // only the GPU thread knows when textures finish streaming in.
    struct SampleCount {
        std::uint32_t samples = 0;
        std::uint64_t residency = 0;
    };

    // Whether the view still shows what its samples so far show.
    bool continuesAccumulation(ViewContext const &view, bool const historyValid) {
        auto const found = accumulations.find(view.view);
        auto const continues = historyValid and found != accumulations.end()
                           and sameCamera(found->second.camera, view.camera) and found->second.extent == view.extent
                           and found->second.sceneVersion == view.sceneVersion;
        accumulations[view.view] = Accumulation{view.camera, view.extent, view.sceneVersion};
        return continues;
    }

    std::unordered_map<ViewId, Accumulation> accumulations;
    std::shared_ptr<std::unordered_map<ViewId, SampleCount>> sampleCounts =
        std::make_shared<std::unordered_map<ViewId, SampleCount>>();
};

struct TonemapFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Output; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto &pass = graph.addPass("tonemap");
        pass.read(view.sceneColor);
        pass.write(view.output);
        pass.record([color = view.sceneColor, output = view.output, extent = view.extent,
                     exposure = view.settings.exposure,
                     filmic = view.settings.toneMapping == ToneMapping::Aces ? 1u : 0u](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::Tonemap{};
            device.useKernel(kernel);
            bind(device, kernel.color, context.image(color));
            bind(device, kernel.output, context.buffer(output));
            set(device, kernel.size, extent);
            set(device, kernel.exposure, exposure);
            set(device, kernel.filmic, filmic);
            context.dispatch(kernel, groupsCovering(extent));
        });
    }
};

} // namespace

std::shared_ptr<RenderFeature> makeVisibilityFeature() { return std::make_shared<VisibilityFeature>(); }
std::shared_ptr<RenderFeature> makeShadowFeature() { return std::make_shared<ShadowFeature>(); }
std::shared_ptr<RenderFeature> makeShadingFeature() { return std::make_shared<ShadingFeature>(); }
std::shared_ptr<RenderFeature> makeTransparencyFeature() { return std::make_shared<TransparencyFeature>(); }
std::shared_ptr<RenderFeature> makeTonemapFeature() { return std::make_shared<TonemapFeature>(); }
std::shared_ptr<RenderFeature> makePathTraceFeature() { return std::make_shared<PathTraceFeature>(); }

} // namespace lr::render
