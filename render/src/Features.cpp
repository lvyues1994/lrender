#include <lr/render/Features.h>

#include "MotionPass.h"
#include "PassHelpers.h"
#include "ShadowCascades.h"
#include "kernels/Denoise.h"
#include "kernels/Lights.h"
#include "kernels/PathTrace.h"
#include "kernels/Post.h"
#include "kernels/Shading.h"
#include "kernels/Visibility.h"

#include <lr/render/MaterialModel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace lr::render {

namespace {

// Room for this many transparent fragments per pixel on average; more are dropped.
constexpr auto kTransparentFragmentsPerPixel = std::size_t{2};

// The denoiser shows its colour while the path tracer's mean holds fewer samples than the
// first, and fades into the mean up to the second.
constexpr auto kDenoiseFadeFrom = 16.0f;
constexpr auto kDenoiseFadeTo = 64.0f;
// A-trous iterations, the first one step apart, each twice the step of the one before.
constexpr auto kDenoiseIterations = 5;

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

// What one raster pass draws: its visible clusters (see CullClusters) and its workgroup
// counts, and the visible instances the clusters came from, through which the triangles
// the pass writes resolve (see compactBlockInstances()).
struct VisibleSet {
    BufferRef<tc::uvec4> visible;
    BufferRef<tc::DispatchIndirectCommand> arguments;
    BufferRef<tc::uvec4> instances;
};

struct CullRequest {
    std::string cullPass;
    std::string compactPass;
    std::string clusterPass;
    std::string compactClustersPass;
    core::mat4 viewProjection{1.0f};
    // CullClusters' viewer: a position (w 1) or the direction the view looks along (w 0).
    core::vec4 viewer{0.0f};
    bool blended = false; // the instances whose material blends, or the others
};

// One compaction: of instances, or of the candidate clusters, which come `items` at most
// and as many as candidates[0].z.
struct CompactRequest {
    BufferRef<tc::uvec4> blocks; // the sums of each block of items
    BufferRef<tc::uvec4> visible;
    BufferRef<tc::DispatchIndirectCommand> arguments;
    tc::uint items = 0;
    tc::uint capacity = 0; // entries `visible` holds
};

// Entries and their tails, after the totals.
std::size_t listWords(std::size_t const capacity) { return 1 + 2 * capacity; }

void recordScan(PassContext &context, CompactRequest const &request, tc::uint const blocks, tc::uint const counted) {
    auto &device = context.device();
    auto scan = kernels::ScanVisibleBlocks{};
    device.useKernel(scan);
    bind(device, scan.blocks, context.buffer(request.blocks));
    bind(device, scan.visible, context.buffer(request.visible));
    bind(device, scan.rasterArguments, context.buffer(request.arguments));
    set(device, scan.blockCount, blocks);
    set(device, scan.counted, counted);
    set(device, scan.capacity, request.capacity);
    context.dispatch(scan, tc::uvec3{1, 1, 1});
}

// S and W are SumVisibleBlocks and WriteVisible, or their subgroup variants, which have
// the same bindings.
template <class S, class W>
void recordCompaction(PassContext &context, CompactRequest const &request, BufferRef<tc::uvec4> const countsRef) {
    auto &device = context.device();
    auto const &scene = context.scene();
    auto const blocks = kernels::compactBlocks(request.items);
    auto &counts = context.buffer(countsRef);
    auto &sums = context.buffer(request.blocks);
    if (blocks > 0) {
        auto sum = S{};
        device.useKernel(sum);
        bind(device, sum.counts, counts);
        bind(device, sum.blocks, sums);
        set(device, sum.instanceCount, request.items);
        context.dispatch(sum, tc::uvec3{blocks, 1, 1});
    }
    recordScan(context, request, blocks, 2);
    if (blocks > 0) {
        auto write = W{};
        device.useKernel(write);
        bind(device, write.instances, *scene.instances);
        bind(device, write.meshes, *scene.meshes);
        bind(device, write.counts, counts);
        bind(device, write.blocks, sums);
        bind(device, write.visible, context.buffer(request.visible));
        set(device, write.instanceCount, request.items);
        context.dispatch(write, tc::uvec3{blocks, 1, 1});
    }
}

// The candidate clusters' counts and the list of instances they came from.
struct ClusterSource {
    BufferRef<tc::uint> counts;
    BufferRef<tc::uvec4> candidates;
};

// S and W are SumClusterBlocks and WriteVisibleClusters, or their subgroup variants.
template <class S, class W>
void recordClusterCompaction(PassContext &context, CompactRequest const &request, ClusterSource const &source) {
    auto &device = context.device();
    auto const &scene = context.scene();
    auto const blocks = kernels::compactBlocks(request.items);
    auto &counts = context.buffer(source.counts);
    auto &sums = context.buffer(request.blocks);
    auto &candidates = context.buffer(source.candidates);
    auto sum = S{};
    device.useKernel(sum);
    bind(device, sum.counts, counts);
    bind(device, sum.blocks, sums);
    bind(device, sum.candidates, candidates);
    context.dispatch(sum, tc::uvec3{blocks, 1, 1});
    recordScan(context, request, blocks, 1);
    auto write = W{};
    device.useKernel(write);
    bind(device, write.counts, counts);
    bind(device, write.blocks, sums);
    bind(device, write.visible, context.buffer(request.visible));
    bind(device, write.candidates, candidates);
    bind(device, write.instances, *scene.instances);
    bind(device, write.meshes, *scene.meshes);
    bind(device, write.clusters, *scene.clusters);
    context.dispatch(write, tc::uvec3{blocks, 1, 1});
}

// The candidates' counts compacted, in order, into `drawn`.
void compactClusters(FrameGraphBuilder &graph, std::string const &pass, ClusterSource const &source,
                     VisibleSet const &drawn, std::size_t const candidateCapacity, bool const subgroups) {
    auto const blocks = graph.createBuffer<tc::uvec4>(
        pass + ": blocks", std::max<std::size_t>(kernels::compactBlocks(static_cast<tc::uint>(candidateCapacity)), 1));
    auto &compact = graph.addPass(pass);
    compact.read(source.counts);
    compact.read(source.candidates);
    compact.write(blocks);
    compact.write(drawn.visible);
    compact.write(drawn.arguments);
    compact.record([request = CompactRequest{blocks, drawn.visible, drawn.arguments,
                                             static_cast<tc::uint>(candidateCapacity),
                                             static_cast<tc::uint>(candidateCapacity)},
                    source, subgroups](PassContext &context) {
        if (subgroups) {
            recordClusterCompaction<kernels::SumClusterBlocksSubgroups, kernels::WriteVisibleClustersSubgroups>(
                context, request, source);
        } else {
            recordClusterCompaction<kernels::SumClusterBlocks, kernels::WriteVisibleClusters>(context, request, source);
        }
    });
}

VisibleSet clusterListOf(FrameGraphBuilder &graph, std::string const &pass, VisibleSet const &instances,
                         std::size_t const candidateCapacity) {
    return VisibleSet{graph.createBuffer<tc::uvec4>(pass + ": visible", listWords(candidateCapacity)),
                      graph.createBuffer<tc::DispatchIndirectCommand>(pass + ": arguments", 1), instances.instances};
}

// A raster pass's culling: what it draws, and what occlusion culling's second phase needs
// of the candidates.
struct Culled {
    VisibleSet drawn;
    BufferRef<tc::uint> counts;
    BufferRef<tc::uint> states;
    BufferRef<tc::DispatchIndirectCommand> candidateArguments;
    std::size_t candidateCapacity = 0;
};

// The instances in view, then their clusters that may show (with `history`, those of them
// visible the frame before), each step listed in order.
Culled cullAndCompact(FrameGraphBuilder &graph, ViewContext const &view, CullRequest const &request,
                      HistoryBuffer<tc::uint> const *const history = nullptr) {
    auto const instanceCapacity = roundUpToPowerOfTwo(std::max<std::size_t>(view.instanceCount, 1));
    auto const instanceCount = static_cast<tc::uint>(view.instanceCount);
    auto const candidateCapacity = roundUpToPowerOfTwo(std::max<std::size_t>(view.clusterInstanceCount, 1));
    auto const clusterCounts = graph.createBuffer<tc::uvec4>(request.cullPass + ": clusters", instanceCapacity);
    auto const instanceBlocks = graph.createBuffer<tc::uvec4>(
        request.compactPass + ": blocks", std::max<std::size_t>(kernels::compactBlocks(instanceCount), 1));
    auto const instances = graph.createBuffer<tc::uvec4>(request.compactPass + ": visible", listWords(instanceCapacity));
    auto const clusterArguments = graph.createBuffer<tc::DispatchIndirectCommand>(request.compactPass + ": arguments", 1);
    auto const triangleCounts = graph.createBuffer<tc::uint>(request.clusterPass + ": triangles", candidateCapacity);
    auto const states =
        graph.createBuffer<tc::uint>(request.clusterPass + ": states", history != nullptr ? candidateCapacity : 1);
    // CullClusters reads no history without it.
    auto const previous = history != nullptr ? history->previous : states;
    auto const culled = Culled{clusterListOf(graph, request.compactClustersPass,
                                             VisibleSet{{}, {}, instances}, candidateCapacity),
                               triangleCounts, states, clusterArguments, candidateCapacity};

    auto &cull = graph.addPass(request.cullPass);
    cull.write(clusterCounts);
    cull.record([clusterCounts, instanceCount, viewProjection = request.viewProjection,
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
        bind(device, kernel.counts, context.buffer(clusterCounts));
        set(device, kernel.viewProjection, viewProjection);
        set(device, kernel.instanceCount, instanceCount);
        set(device, kernel.blended, blended);
        context.dispatch(kernel, tc::uvec3{groupsFor(instanceCount, 64), 1, 1});
    });

    auto const subgroups = view.capabilities.subgroups;
    auto &compact = graph.addPass(request.compactPass);
    compact.read(clusterCounts);
    compact.write(instanceBlocks);
    compact.write(instances);
    compact.write(clusterArguments);
    compact.record([compaction = CompactRequest{instanceBlocks, instances, clusterArguments, instanceCount,
                                                static_cast<tc::uint>(instanceCapacity)},
                    clusterCounts, subgroups](PassContext &context) {
        if (subgroups) {
            recordCompaction<kernels::SumVisibleBlocksSubgroups, kernels::WriteVisibleSubgroups>(context, compaction,
                                                                                                  clusterCounts);
        } else {
            recordCompaction<kernels::SumVisibleBlocks, kernels::WriteVisible>(context, compaction, clusterCounts);
        }
    });

    auto const occlusion = history == nullptr ? kernels::occlusionOff()
                         : history->valid     ? kernels::occlusionWithHistory()
                                              : kernels::occlusionWithoutHistory();
    auto &cullClusters = graph.addPass(request.clusterPass);
    cullClusters.read(instances);
    cullClusters.read(clusterArguments);
    if (history != nullptr) {
        cullClusters.read(previous);
    }
    cullClusters.write(triangleCounts);
    cullClusters.write(states);
    cullClusters.record([instances, clusterArguments, triangleCounts, previous, states,
                         viewProjection = request.viewProjection, viewer = request.viewer,
                         tests = view.cullClusters ? kernels::cullClusterBoxes() | kernels::cullClusterCones() : 0u,
                         occlusion](PassContext &context) {
        auto &device = context.device();
        auto const &scene = context.scene();
        auto kernel = kernels::CullClusters{};
        device.useKernel(kernel);
        bind(device, kernel.instances, *scene.instances);
        bind(device, kernel.meshes, *scene.meshes);
        bind(device, kernel.clusters, *scene.clusters);
        bind(device, kernel.materials, *scene.materials);
        bind(device, kernel.visible, context.buffer(instances));
        bind(device, kernel.counts, context.buffer(triangleCounts));
        bind(device, kernel.previous, context.buffer(previous));
        bind(device, kernel.states, context.buffer(states));
        set(device, kernel.viewProjection, viewProjection);
        set(device, kernel.viewer, viewer);
        set(device, kernel.tests, tests);
        set(device, kernel.occlusion, occlusion);
        context.dispatchIndirect(kernel, clusterArguments);
    });

    compactClusters(graph, request.compactClustersPass, ClusterSource{triangleCounts, instances}, culled.drawn,
                    candidateCapacity, subgroups);
    return culled;
}

// The levels of a view's depth pyramid (see pyramidLevelSize()): their sizes, and the words
// all of them take.
struct Pyramid {
    std::vector<core::ivec2> sizes;
    std::size_t words = 0;
};

Pyramid pyramidOf(core::ivec2 const extent) {
    auto pyramid = Pyramid{};
    auto size = kernels::pyramidLevelSize(extent);
    for (;;) {
        pyramid.sizes.push_back(size);
        pyramid.words += static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y);
        if (size.x == 1 and size.y == 1) {
            return pyramid;
        }
        size = kernels::pyramidLevelSize(size);
    }
}

// Builds the pyramid from the depth of `source`, plain depths or packed visibility.
template <class First, class T>
void buildPyramid(FrameGraphBuilder &graph, ViewContext const &view, BufferRef<T> const source,
                  BufferRef<tc::uint> const pyramid) {
    auto &pass = graph.addPass("build depth pyramid");
    pass.read(source);
    pass.write(pyramid);
    pass.record([source, pyramid, levels = pyramidOf(view.extent), extent = view.extent](PassContext &context) {
        auto &device = context.device();
        auto first = First{};
        device.useKernel(first);
        bind(device, first.hiz, context.buffer(pyramid));
        if constexpr (std::is_same_v<T, tc::u64>) {
            bind(device, first.packed, context.buffer(source));
        } else {
            bind(device, first.depth, context.buffer(source));
        }
        set(device, first.sourceSize, extent);
        set(device, first.targetSize, levels.sizes[0]);
        set(device, first.sourceOffset, 0u);
        set(device, first.targetOffset, 0u);
        context.dispatch(first, groupsCovering(levels.sizes[0]));
        auto offset = tc::uint{0};
        for (auto level = std::size_t{1}; level < levels.sizes.size(); ++level) {
            auto const below = levels.sizes[level - 1];
            auto const next = offset + static_cast<tc::uint>(below.x * below.y);
            auto kernel = kernels::DownsamplePyramid{};
            device.useKernel(kernel);
            bind(device, kernel.hiz, context.buffer(pyramid));
            set(device, kernel.sourceSize, below);
            set(device, kernel.targetSize, levels.sizes[level]);
            set(device, kernel.sourceOffset, offset);
            set(device, kernel.targetOffset, next);
            context.dispatch(kernel, groupsCovering(levels.sizes[level]));
            offset = next;
        }
    });
}

// Occlusion culling's second phase: the candidates that stayed tested against the pyramid,
// the history of this frame written, and those not drawn yet listed.
VisibleSet cullOccluded(FrameGraphBuilder &graph, ViewContext const &view, Culled const &early,
                        BufferRef<tc::uint> const pyramid, HistoryBuffer<tc::uint> const &history,
                        std::size_t const historyCapacity) {
    auto const late = clusterListOf(graph, "compact late clusters", early.drawn, early.candidateCapacity);
    auto &pass = graph.addPass("cull occluded clusters");
    pass.read(early.drawn.instances);
    pass.read(early.candidateArguments);
    pass.read(early.states);
    pass.read(pyramid);
    pass.write(early.counts);
    pass.write(history.current);
    pass.record([instances = early.drawn.instances, arguments = early.candidateArguments, counts = early.counts,
                 states = early.states, current = history.current, pyramid, historyCapacity,
                 viewProjection = view.viewProjection, extent = view.extent,
                 levels = static_cast<tc::uint>(pyramidOf(view.extent).sizes.size())](PassContext &context) {
        clearWords(context, current, 0, historyCapacity, 0u);
        auto &device = context.device();
        auto const &scene = context.scene();
        auto kernel = kernels::CullOccludedClusters{};
        device.useKernel(kernel);
        bind(device, kernel.instances, *scene.instances);
        bind(device, kernel.meshes, *scene.meshes);
        bind(device, kernel.clusters, *scene.clusters);
        bind(device, kernel.visible, context.buffer(instances));
        bind(device, kernel.counts, context.buffer(counts));
        bind(device, kernel.states, context.buffer(states));
        bind(device, kernel.current, context.buffer(current));
        bind(device, kernel.hiz, context.buffer(pyramid));
        set(device, kernel.viewProjection, viewProjection);
        set(device, kernel.size, extent);
        set(device, kernel.levels, levels);
        context.dispatchIndirect(kernel, arguments);
    });
    compactClusters(graph, "compact late clusters", ClusterSource{early.counts, early.drawn.instances}, late,
                    early.candidateCapacity, view.capabilities.subgroups);
    return late;
}

// The large triangles one raster pass lists for its tile stage (see largeTriangleWord()),
// and that stage's workgroup counts.
struct LargeTriangles {
    BufferRef<tc::uint> list;
    BufferRef<tc::DispatchIndirectCommand> arguments;
};

LargeTriangles largeTrianglesOf(FrameGraphBuilder &graph, std::string const &pass) {
    return LargeTriangles{graph.createBuffer<tc::uint>(pass + ": large triangles", kernels::largeTriangleWords()),
                          graph.createBuffer<tc::DispatchIndirectCommand>(pass + ": tile arguments", 1)};
}

// What a raster pass reads and writes besides its targets.
void declareRaster(PassBuilder &pass, VisibleSet const &drawn, LargeTriangles const &large) {
    pass.read(drawn.visible);
    pass.read(drawn.arguments);
    pass.write(large.list);
    pass.write(large.arguments);
}

struct RasterInputs {
    VisibleSet drawn;
    LargeTriangles large;
    core::mat4 viewProjection{1.0f};
    core::ivec2 size{0};
};

// The bindings and uniforms of LR_TRIANGLE_RASTERIZER.
template <class K>
void bindRasterizer(PassContext &context, K &kernel, RasterInputs const &inputs) {
    auto &device = context.device();
    auto const &scene = context.scene();
    bindMaterialTextures(device, kernel, scene.textures);
    bind(device, kernel.instances, *scene.instances);
    bind(device, kernel.vertices, *scene.vertices);
    bind(device, kernel.indices, *scene.indices);
    bind(device, kernel.materials, *scene.materials);
    bind(device, kernel.visible, context.buffer(inputs.drawn.visible));
    bind(device, kernel.large, context.buffer(inputs.large.list));
    set(device, kernel.viewProjection, inputs.viewProjection);
    set(device, kernel.size, inputs.size);
}

// Both stages of a rasterizer (see largeTriangleArea()): the triangle stage K and the tile
// stage T; `bindTargets(kernel)` binds what they add to LR_TRIANGLE_RASTERIZER.
template <class K, class T, class BindTargets>
void recordRasterStages(PassContext &context, RasterInputs const &inputs, BindTargets const &bindTargets) {
    auto &device = context.device();
    clearWords(context, inputs.large.list, 0, 2, 0u);
    auto triangles = K{};
    device.useKernel(triangles);
    bindRasterizer(context, triangles, inputs);
    bindTargets(triangles);
    context.dispatchIndirect(triangles, inputs.drawn.arguments);

    auto scan = kernels::ScanLargeTriangles{};
    device.useKernel(scan);
    bind(device, scan.large, context.buffer(inputs.large.list));
    bind(device, scan.tileArguments, context.buffer(inputs.large.arguments));
    context.dispatch(scan, tc::uvec3{1, 1, 1});

    auto tiles = T{};
    device.useKernel(tiles);
    bindRasterizer(context, tiles, inputs);
    bindTargets(tiles);
    context.dispatchIndirect(tiles, inputs.large.arguments);
}

// What RasterVisibility and RasterVisibilityTiles add.
struct VisibilityTargets {
    PassContext &context;
    WordRange depth;
    WordRange triangles;
    tc::uint mode = 0;
    tc::uint fragmentCapacity = 0;

    template <class K>
    void operator()(K &kernel) const {
        bindRange(context, kernel.depth, depth);
        bindRange(context, kernel.triangles, triangles);
        set(context.device(), kernel.mode, mode);
        set(context.device(), kernel.fragmentCapacity, fragmentCapacity);
    }
};

struct RasterRequest {
    RasterInputs inputs;
    tc::uint mode = 0;
    WordRange depth;
    WordRange triangles;
    tc::uint fragmentCapacity = 0;
};

void recordRaster(PassContext &context, RasterRequest const &request) {
    recordRasterStages<kernels::RasterVisibility, kernels::RasterVisibilityTiles>(
        context, request.inputs,
        VisibilityTargets{context, request.depth, request.triangles, request.mode, request.fragmentCapacity});
}

// What shading kernels of a view take besides the scene.
struct ShadingView {
    core::mat4 inverseViewProjection{1.0f};
    core::vec4 environment{0.0f};
    core::ivec2 extent{0};
    tc::uint lightCount = 0;
    ShadowOutputs shadows;
    LightGridOutputs lights;
};

ShadingView shadingViewOf(ViewContext const &view, ShadowOutputs const &shadows, LightGridOutputs const &lights) {
    return ShadingView{view.inverseViewProjection, tc::vec4{view.environment.radiance, 0.0f}, view.extent,
                       view.lightCount, shadows, lights};
}

// The view's light grid, assigned by the first feature that shades.
LightGridOutputs lightGridOf(FrameGraphBuilder &graph, ViewContext const &view) {
    auto &board = graph.blackboard();
    if (board.contains<LightGridOutputs>()) {
        return board.get<LightGridOutputs>();
    }
    auto const tiles = [](int const pixels) {
        return static_cast<tc::uint>((pixels + static_cast<int>(lightTileSize()) - 1) / static_cast<int>(lightTileSize()));
    };
    auto const shape = view.clusterLights ? tc::uvec4{tiles(view.extent.x), tiles(view.extent.y), lightSlices(), 1u}
                                          : tc::uvec4{1u, 1u, 1u, 0u};
    auto const lightCount = static_cast<tc::uint>(view.lightCount);
    auto const maskWords = static_cast<std::size_t>(shape.x * shape.y * shape.z) * lightMaskWords(lightCount);
    auto const grid = graph.createBuffer<tc::uint>(
        "light grid", lightGridHeaderWords() + std::size_t{lightRecordWords()} * lightCount + maskWords);
    auto const near = std::max(view.camera.nearPlane, 1.0e-6f);
    auto const depths =
        tc::vec2{near, static_cast<float>(lightSlices()) / std::max(std::log(view.camera.farPlane / near), 1.0e-3f)};
    auto &pass = graph.addPass("assign lights");
    pass.write(grid);
    pass.record([grid, shape, depths, lightCount, viewProjection = view.viewProjection, extent = view.extent,
                 invocations = std::max<std::size_t>({maskWords, lightCount, 1})](PassContext &context) {
        auto &device = context.device();
        auto kernel = kernels::AssignLights{};
        device.useKernel(kernel);
        bind(device, kernel.lights, *context.scene().lights);
        bind(device, kernel.grid, context.buffer(grid));
        set(device, kernel.viewProjection, viewProjection);
        set(device, kernel.size, extent);
        set(device, kernel.shape, shape);
        set(device, kernel.depths, depths);
        set(device, kernel.lightCount, lightCount);
        context.dispatch(kernel, tc::uvec3{groupsFor(invocations, 64), 1, 1});
    });
    board.put(LightGridOutputs{grid});
    return LightGridOutputs{grid};
}

SurfaceShadingResources surfaceResources(PassContext &context, ShadingView const &view,
                                         BufferRef<tc::uvec4> const visible) {
    auto resources = SurfaceShadingResources{};
    resources.scene = &context.scene();
    resources.visible = &context.buffer(visible);
    resources.shadow = &context.buffer(view.shadows.map);
    resources.lights = &context.buffer(view.lights.grid);
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

    // With occlusion culling, the clusters visible the frame before are drawn first, the
    // depth pyramid is built from them, and the others that may show after them (see
    // CullOccludedClusters).
    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const pixels = pixelsOf(view.extent);
        auto const triangles = graph.createBuffer<tc::uint>("visibility", pixels);
        auto const depth = graph.createBuffer<tc::uint>("depth", pixels);
        auto const totals = graph.createReadbackBuffer<tc::uvec4>("visible totals", 1);
        auto const singlePass = view.capabilities.singlePassVisibility;
        if (not singlePass) {
            clearSeparate(graph, depth, triangles, pixels);
        }
        auto const occlusion = view.cullClusters and view.cullOccluded;
        auto const historyCapacity = roundUpToPowerOfTwo(std::max<std::size_t>(view.clusterInstanceCount, 1));
        auto const history = occlusion ? std::optional<HistoryBuffer<tc::uint>>{graph.historyBuffer<tc::uint>(
                                             "cluster visibility", historyCapacity)}
                                       : std::nullopt;
        auto const early = cullAndCompact(graph, view,
                                          CullRequest{"cull instances", "compact visible instances", "cull clusters",
                                                      "compact visible clusters", view.viewProjection,
                                                      core::vec4{view.camera.position, 1.0f}, false},
                                          history ? &*history : nullptr);
        auto const outputs = VisibilityOutputs{early.drawn.instances, triangles, depth};
        auto late = std::optional<VisibleSet>{};
        auto const cullLate = [&](auto const source, auto const first) {
            auto const pyramid = graph.createBuffer<tc::uint>("depth pyramid", pyramidOf(view.extent).words);
            buildPyramid<decltype(first)>(graph, view, source, pyramid);
            late = cullOccluded(graph, view, early, pyramid, *history, historyCapacity);
        };
        if (singlePass) {
            auto const packed = clearPacked(graph, view);
            rasterPacked(graph, view, "rasterize visibility", early.drawn, packed);
            if (occlusion) {
                cullLate(packed, kernels::DownsamplePackedDepth{});
                rasterPacked(graph, view, "rasterize late visibility", *late, packed);
            }
            unpack(graph, view, packed, outputs);
        } else {
            rasterTargets(graph, view, "rasterize depth", early.drawn, outputs, kernels::rasterDepth());
            if (occlusion) {
                cullLate(depth, kernels::DownsampleDepth{});
                rasterTargets(graph, view, "rasterize late depth", *late, outputs, kernels::rasterDepth());
            }
            rasterTargets(graph, view, "rasterize triangles", early.drawn, outputs, kernels::rasterTriangles());
            if (late) {
                rasterTargets(graph, view, "rasterize late triangles", *late, outputs, kernels::rasterTriangles());
            }
        }

        auto const lateList = late ? late->visible : early.drawn.visible;
        auto &stats = graph.addPass("visibility statistics");
        stats.read(early.drawn.instances);
        stats.read(early.drawn.visible);
        stats.read(lateList);
        stats.write(totals);
        stats.record([instances = early.drawn.instances, visible = early.drawn.visible, lateList, totals,
                      lists = late ? 2u : 1u](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::CopyVisibleTotals{};
            device.useKernel(kernel);
            bind(device, kernel.instances, context.buffer(instances));
            bind(device, kernel.visible, context.buffer(visible));
            bind(device, kernel.late, context.buffer(lateList));
            bind(device, kernel.totals, context.buffer(totals));
            set(device, kernel.lists, lists);
            context.dispatch(kernel, tc::uvec3{1, 1, 1});
        });
        graph.readback<tc::uvec4>(totals, [](tc::uvec4 const *data, std::size_t, ViewResult &result) {
            result.stats.visibleInstances = data[0].x;
            result.stats.visibleClusters = data[0].y;
            result.stats.visibleTriangles = data[0].z;
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

    // Two of these, the depth first, then the triangles at the nearest depth.
    static void rasterTargets(FrameGraphBuilder &graph, ViewContext const &view, std::string const &name,
                              VisibleSet const &drawn, VisibilityOutputs const &outputs, tc::uint const mode) {
        auto const large = largeTrianglesOf(graph, name);
        auto &raster = graph.addPass(name);
        declareRaster(raster, drawn, large);
        raster.read(outputs.depth);
        if (mode == kernels::rasterDepth()) {
            raster.write(outputs.depth);
        } else {
            raster.read(outputs.triangles);
            raster.write(outputs.triangles);
        }
        auto request = RasterRequest{};
        request.inputs = RasterInputs{drawn, large, view.viewProjection, view.extent};
        request.mode = mode;
        request.depth = WordRange{outputs.depth};
        request.triangles = WordRange{outputs.triangles};
        raster.record([request](PassContext &context) { recordRaster(context, request); });
    }

    // Both at once with 64-bit atomics, into a buffer that unpack() turns into those that
    // rasterTargets() fills, so the passes after see the same.
    static BufferRef<tc::u64> clearPacked(FrameGraphBuilder &graph, ViewContext const &view) {
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
        return packed;
    }

    static void rasterPacked(FrameGraphBuilder &graph, ViewContext const &view, std::string const &name,
                             VisibleSet const &drawn, BufferRef<tc::u64> const packed) {
        auto const large = largeTrianglesOf(graph, name);
        auto &raster = graph.addPass(name);
        declareRaster(raster, drawn, large);
        raster.read(packed);
        raster.write(packed);
        raster.record([inputs = RasterInputs{drawn, large, view.viewProjection, view.extent},
                       packed](PassContext &context) {
            recordRasterStages<kernels::RasterPackedVisibility, kernels::RasterPackedVisibilityTiles>(
                context, inputs, [&](auto &kernel) { bind(context.device(), kernel.packed, context.buffer(packed)); });
        });
    }

    static void unpack(FrameGraphBuilder &graph, ViewContext const &view, BufferRef<tc::u64> const packed,
                       VisibilityOutputs const &outputs) {
        auto const pixels = static_cast<tc::uint>(pixelsOf(view.extent));
        auto &unpackPass = graph.addPass("unpack visibility");
        unpackPass.read(packed);
        unpackPass.write(outputs.depth);
        unpackPass.write(outputs.triangles);
        unpackPass.record([packed, depth = outputs.depth, triangles = outputs.triangles, pixels](PassContext &context) {
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
            auto const drawn =
                cullAndCompact(graph, view,
                               CullRequest{"cull shadow casters " + name, "compact shadow casters " + name,
                                           "cull shadow clusters " + name, "compact shadow clusters " + name,
                                           cascades[index].viewProjection,
                                           core::vec4{view.shadowLight->direction, 0.0f}, false})
                    .drawn;
            auto const large = largeTrianglesOf(graph, "rasterize shadow cascade " + name);
            auto &raster = graph.addPass("rasterize shadow cascade " + name);
            declareRaster(raster, drawn, large);
            raster.read(map);
            raster.write(map);
            auto rasterRequest = RasterRequest{};
            rasterRequest.inputs = RasterInputs{drawn, large, cascades[index].viewProjection,
                                                core::ivec2{static_cast<int>(resolution), static_cast<int>(resolution)}};
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
        auto const shadows = shadowsOf(graph);
        auto const shading = shadingViewOf(view, shadows, lightGridOf(graph, view));
        auto const lists = classify(graph, view, visibility);
        for (auto index = std::size_t{0}; index < view.materialModels.size(); ++index) {
            auto const &model = view.materialModels[index];
            auto &pass = graph.addPass("shade " + std::string{model->program()});
            pass.read(lists.pixels);
            pass.read(lists.arguments);
            pass.read(visibility.visible);
            pass.read(shading.shadows.map);
            pass.read(shading.lights.grid);
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
                                                      "cull transparent clusters", "compact transparent clusters",
                                                      view.viewProjection, core::vec4{view.camera.position, 1.0f},
                                                      true})
                              .drawn;
        auto const large = largeTrianglesOf(graph, "rasterize transparent");
        auto &raster = graph.addPass("rasterize transparent");
        declareRaster(raster, drawn, large);
        raster.read(visibility.depth);
        raster.read(lists);
        raster.write(lists);
        auto rasterRequest = RasterRequest{};
        rasterRequest.inputs = RasterInputs{drawn, large, view.viewProjection, view.extent};
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

        auto const shadows = shadowsOf(graph);
        auto const shading = shadingViewOf(view, shadows, lightGridOf(graph, view));
        auto &blend = graph.addPass("blend transparent");
        blend.read(drawn.instances);
        blend.read(lists);
        blend.read(opaque);
        blend.read(shading.shadows.map);
        blend.read(shading.lights.grid);
        blend.write(view.sceneColor);
        blend.record([shading, visible = drawn.instances, lists, opaque, color = view.sceneColor](PassContext &context) {
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
        auto inputs = TracerInputs{accumulation,
                                   view.sceneColor,
                                   view.extent,
                                   view.inverseViewProjection,
                                   tc::vec4{view.environment.radiance, 0.0f},
                                   static_cast<tc::uint>(view.lightCount),
                                   0u};
        pass.record([inputs, continues, total, counts = sampleCounts, viewId = view.view,
                     rayQueries = view.capabilities.rayQueries](PassContext &context) mutable {
            auto &device = context.device();
            auto const &scene = context.scene();
            if (rayQueries ? scene.topLevel == nullptr : scene.tlasNodes == nullptr) {
                throw std::logic_error{"lr::render: the path tracer ran before the scene had hierarchies"};
            }
            auto &count = (*counts)[viewId];
            auto const samples = continues and count.residency == scene.textures.residency ? count.samples : 0u;
            count = SampleCount{samples + 1, scene.textures.residency};
            *total = samples + 1;
            inputs.samples = samples;

            if (rayQueries) {
                auto kernel = kernels::PathTraceRayQuery{};
                device.useKernel(kernel);
                bindTracerInputs(context, kernel, inputs);
                bind(device, kernel.scene, *scene.topLevel);
                bind(device, kernel.meshes, *scene.meshes);
                bind(device, kernel.indices, *scene.indices);
                context.dispatch(kernel, groupsCovering(inputs.extent));
                return;
            }
            auto kernel = kernels::PathTrace{};
            device.useKernel(kernel);
            bindTracerInputs(context, kernel, inputs);
            bind(device, kernel.tlasNodes, *scene.tlasNodes);
            bind(device, kernel.blasNodes, *scene.blasNodes);
            bind(device, kernel.blasTriangles, *scene.blasTriangles);
            bind(device, kernel.blasRoots, *scene.blasRoots);
            set(device, kernel.tlasNodeCount, scene.tlasNodeCount);
            context.dispatch(kernel, groupsCovering(inputs.extent));
        });
        graph.onFinish([total](ViewResult &result) { result.stats.accumulatedSamples = *total; });
        graph.blackboard().put(PathTraceOutputs{accumulation.current, total});
    }

private:
    // What the bindings and uniforms of LR_PATH_TRACER_INPUTS take, besides the scene.
    struct TracerInputs {
        HistoryImage<AccumulationFormat> accumulation;
        ImageRef<HdrFormat> color;
        core::ivec2 extent;
        tc::mat4 inverseViewProjection;
        tc::vec4 environment; // radiance in xyz
        tc::uint lightCount;
        tc::uint samples; // averaged in accumulation.previous
    };

    template <class K>
    static void bindTracerInputs(PassContext &context, K &kernel, TracerInputs const &inputs) {
        auto &device = context.device();
        auto const &scene = context.scene();
        bind(device, kernel.previous, context.image(inputs.accumulation.previous));
        bind(device, kernel.current, context.image(inputs.accumulation.current));
        bind(device, kernel.color, context.image(inputs.color));
        bindMaterialTextures(device, kernel, scene.textures);
        bind(device, kernel.instances, *scene.instances);
        bind(device, kernel.vertices, *scene.vertices);
        bind(device, kernel.materials, *scene.materials);
        bind(device, kernel.lights, *scene.lights);
        set(device, kernel.inverseViewProjection, inputs.inverseViewProjection);
        set(device, kernel.environment, inputs.environment);
        set(device, kernel.size, inputs.extent);
        set(device, kernel.lightCount, inputs.lightCount);
        set(device, kernel.sampleIndex, inputs.samples);
    }

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

struct PathTraceDenoiseFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::PostProcess; }

    FeatureRequirements requirements() const noexcept override {
        auto requirements = FeatureRequirements{};
        requirements.previousView = true;
        return requirements;
    }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto &board = graph.blackboard();
        if (not view.temporal or not board.contains<VisibilityOutputs>() or not board.contains<PathTraceOutputs>()) {
            return;
        }
        auto const visibility = board.get<VisibilityOutputs>();
        auto const traced = board.get<PathTraceOutputs>();
        auto const pixels = pixelsOf(view.extent);
        auto const extent = view.extent;
        auto const motion = motionOf(graph, view);
        auto const guides = graph.history<AccumulationFormat>("denoise guides", extent);
        auto const albedo = graph.createImage<HdrFormat>("denoise albedo", extent);
        auto const emission = graph.createImage<HdrFormat>("denoise emission", extent);
        auto const illumination = graph.createBuffer<tc::vec4>("denoise illumination", pixels);
        auto const color = graph.historyBuffer<tc::vec4>("denoise colour", pixels);
        auto const moments = graph.historyBuffer<tc::vec2>("denoise moments", pixels);
        auto const filtered = std::array{graph.createBuffer<tc::vec4>("denoise filtered", pixels),
                                         graph.createBuffer<tc::vec4>("denoise filtered back", pixels)};
        auto const shadows = shadowsOf(graph);
        auto const shading = shadingViewOf(view, shadows, lightGridOf(graph, view));

        auto &guide = graph.addPass("denoise guides");
        guide.read(visibility.visible);
        guide.read(visibility.triangles);
        guide.read(shading.shadows.map);
        guide.read(shading.lights.grid);
        guide.write(guides.current);
        guide.write(albedo);
        guide.write(emission);
        guide.record([shading, visibility, guides, albedo, emission](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::PathTraceGuides{};
            device.useKernel(kernel);
            bindSurfaceShading(device, kernel, surfaceResources(context, shading, visibility.visible));
            bind(device, kernel.triangles, context.buffer(visibility.triangles));
            bind(device, kernel.normalDepth, context.image(guides.current));
            bind(device, kernel.albedo, context.image(albedo));
            bind(device, kernel.emission, context.image(emission));
            context.dispatch(kernel, groupsCovering(shading.extent));
        });

        auto &demodulate = graph.addPass("demodulate illumination");
        demodulate.read(view.sceneColor);
        demodulate.read(albedo);
        demodulate.read(emission);
        demodulate.read(guides.current);
        demodulate.write(illumination);
        demodulate.record([color = view.sceneColor, albedo, emission, guides, illumination,
                           extent](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::DemodulateIllumination{};
            device.useKernel(kernel);
            bind(device, kernel.color, context.image(color));
            bind(device, kernel.albedo, context.image(albedo));
            bind(device, kernel.emission, context.image(emission));
            bind(device, kernel.normalDepth, context.image(guides.current));
            bind(device, kernel.illumination, context.buffer(illumination));
            set(device, kernel.size, extent);
            context.dispatch(kernel, groupsCovering(extent));
        });

        auto &accumulate = graph.addPass("accumulate illumination");
        accumulate.read(guides.current);
        accumulate.read(guides.previous);
        accumulate.read(illumination);
        accumulate.read(motion);
        accumulate.read(color.previous);
        accumulate.read(moments.previous);
        accumulate.write(color.current);
        accumulate.write(moments.current);
        auto const historyValid = guides.valid and color.valid and moments.valid and view.temporal->continues;
        accumulate.record([guides, illumination, motion, color, moments, extent,
                           valid = historyValid ? 1u : 0u](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::AccumulateIllumination{};
            device.useKernel(kernel);
            bind(device, kernel.normalDepth, context.image(guides.current));
            bind(device, kernel.previousNormalDepth, context.image(guides.previous));
            bind(device, kernel.illumination, context.buffer(illumination));
            bind(device, kernel.motion, context.buffer(motion));
            bind(device, kernel.previousColor, context.buffer(color.previous));
            bind(device, kernel.previousMoments, context.buffer(moments.previous));
            bind(device, kernel.color, context.buffer(color.current));
            bind(device, kernel.moments, context.buffer(moments.current));
            set(device, kernel.size, extent);
            set(device, kernel.historyValid, valid);
            context.dispatch(kernel, groupsCovering(extent));
        });

        auto &variance = graph.addPass("estimate variance");
        variance.read(guides.current);
        variance.read(color.current);
        variance.read(moments.current);
        variance.write(filtered[0]);
        variance.record([guides, color, moments, target = filtered[0], extent](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::EstimateVariance{};
            device.useKernel(kernel);
            bind(device, kernel.normalDepth, context.image(guides.current));
            bind(device, kernel.color, context.buffer(color.current));
            bind(device, kernel.moments, context.buffer(moments.current));
            bind(device, kernel.filtered, context.buffer(target));
            set(device, kernel.size, extent);
            context.dispatch(kernel, groupsCovering(extent));
        });

        auto &filter = graph.addPass("filter illumination");
        filter.read(guides.current);
        filter.read(filtered[0]);
        filter.write(filtered[0]);
        filter.write(filtered[1]);
        filter.read(color.current);
        filter.write(color.current);
        filter.record([guides, filtered, color, extent](PassContext &context) {
            auto &device = context.device();
            for (auto iteration = 0; iteration < kDenoiseIterations; ++iteration) {
                auto const from = static_cast<std::size_t>(iteration % 2);
                auto kernel = kernels::FilterIllumination{};
                device.useKernel(kernel);
                bind(device, kernel.normalDepth, context.image(guides.current));
                bind(device, kernel.source, context.buffer(filtered[from]));
                bind(device, kernel.target, context.buffer(filtered[1 - from]));
                bind(device, kernel.color, context.buffer(color.current));
                set(device, kernel.size, extent);
                set(device, kernel.step, 1 << iteration);
                set(device, kernel.feedback, iteration == 0 ? 1u : 0u);
                context.dispatch(kernel, groupsCovering(extent));
            }
        });

        auto const denoised = filtered[static_cast<std::size_t>(kDenoiseIterations % 2)];
        auto &remodulate = graph.addPass("remodulate illumination");
        remodulate.read(denoised);
        remodulate.read(traced.mean);
        remodulate.read(albedo);
        remodulate.read(emission);
        remodulate.read(guides.current);
        remodulate.read(view.sceneColor);
        remodulate.write(view.sceneColor);
        remodulate.record([color = view.sceneColor, traced, albedo, emission, guides, denoised,
                           extent](PassContext &context) {
            auto const fade =
                (static_cast<float>(*traced.samples) - kDenoiseFadeFrom) / (kDenoiseFadeTo - kDenoiseFadeFrom);
            auto &device = context.device();
            auto kernel = kernels::RemodulateIllumination{};
            device.useKernel(kernel);
            bind(device, kernel.color, context.image(color));
            bind(device, kernel.mean, context.image(traced.mean));
            bind(device, kernel.albedo, context.image(albedo));
            bind(device, kernel.emission, context.image(emission));
            bind(device, kernel.normalDepth, context.image(guides.current));
            bind(device, kernel.filtered, context.buffer(denoised));
            set(device, kernel.size, extent);
            set(device, kernel.reference, std::clamp(fade, 0.0f, 1.0f));
            context.dispatch(kernel, groupsCovering(extent));
        });
    }
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
std::shared_ptr<RenderFeature> makePathTraceDenoiseFeature() { return std::make_shared<PathTraceDenoiseFeature>(); }

} // namespace lr::render
