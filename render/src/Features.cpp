#include <lr/render/Features.h>

#include "Bind.h"
#include "kernels/PathTrace.h"
#include "kernels/Post.h"
#include "kernels/Shading.h"
#include "kernels/Visibility.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lr::render {

namespace {

std::size_t pixelsOf(core::ivec2 const extent) {
    return static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y);
}

std::size_t roundUpToPowerOfTwo(std::size_t const value) {
    auto result = std::size_t{1};
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

void clear(PassContext &context, BufferRef<tc::uint> const target, std::size_t const count, tc::uint const value) {
    auto &device = context.device();
    auto kernel = kernels::ClearUint{};
    device.useKernel(kernel);
    bind(device, kernel.target, context.buffer(target));
    set(device, kernel.fill, value);
    set(device, kernel.count, static_cast<tc::uint>(count));
    context.dispatch(kernel, tc::uvec3{groupsFor(count, 64), 1, 1});
}

tc::uvec3 groupsCovering(core::ivec2 const extent) {
    return tc::uvec3{groupsFor(static_cast<std::size_t>(extent.x), 8), groupsFor(static_cast<std::size_t>(extent.y), 8),
                     1};
}

bool sameCamera(core::CameraDesc const &a, core::CameraDesc const &b) noexcept {
    return a.position == b.position and a.target == b.target and a.up == b.up and a.verticalFov == b.verticalFov
       and a.nearPlane == b.nearPlane and a.farPlane == b.farPlane;
}

struct VisibilityFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Visibility; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const pixels = pixelsOf(view.extent);
        auto const instanceCapacity = roundUpToPowerOfTwo(std::max<std::size_t>(view.instanceCount, 1));
        auto const outputs = VisibilityOutputs{graph.createBuffer<tc::uvec2>("visible instances", instanceCapacity + 1),
                                               graph.createBuffer<tc::uint>("visibility", pixels),
                                               graph.createBuffer<tc::uint>("depth", pixels)};
        auto const triangleCounts = graph.createBuffer<tc::uint>("triangle counts", instanceCapacity);
        auto const arguments = graph.createBuffer<tc::DispatchIndirectCommand>("raster arguments", 1);
        auto const totals = graph.createReadbackBuffer<tc::uvec2>("visible totals", 1);
        auto const instanceCount = static_cast<tc::uint>(view.instanceCount);

        auto &clearPass = graph.addPass("clear visibility");
        clearPass.write(outputs.depth);
        clearPass.write(outputs.triangles);
        clearPass.record([outputs, pixels](PassContext &context) {
            clear(context, outputs.depth, pixels, 0xFFFFFFFFu);
            clear(context, outputs.triangles, pixels, noTriangle());
        });

        auto &cull = graph.addPass("cull instances");
        cull.write(triangleCounts);
        cull.record([triangleCounts, instanceCount, viewProjection = view.viewProjection](PassContext &context) {
            if (instanceCount == 0) {
                return;
            }
            auto &device = context.device();
            auto const &scene = context.scene();
            auto kernel = kernels::CullInstances{};
            device.useKernel(kernel);
            bind(device, kernel.instances, *scene.instances);
            bind(device, kernel.meshes, *scene.meshes);
            bind(device, kernel.triangleCounts, context.buffer(triangleCounts));
            set(device, kernel.viewProjection, viewProjection);
            set(device, kernel.instanceCount, instanceCount);
            context.dispatch(kernel, tc::uvec3{groupsFor(instanceCount, 64), 1, 1});
        });

        auto &compact = graph.addPass("compact visible instances");
        compact.read(triangleCounts);
        compact.write(outputs.visible);
        compact.write(arguments);
        compact.record([triangleCounts, outputs, arguments, instanceCount](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::CompactVisible{};
            device.useKernel(kernel);
            bind(device, kernel.triangleCounts, context.buffer(triangleCounts));
            bind(device, kernel.visible, context.buffer(outputs.visible));
            bind(device, kernel.rasterArguments, context.buffer(arguments));
            set(device, kernel.instanceCount, instanceCount);
            context.dispatch(kernel, tc::uvec3{1, 1, 1});
        });

        for (auto const writeTriangles : {0u, 1u}) {
            auto &raster = graph.addPass(writeTriangles == 0u ? "rasterize depth" : "rasterize triangles");
            raster.read(outputs.visible);
            raster.read(arguments);
            if (writeTriangles == 0u) {
                raster.read(outputs.depth);
                raster.write(outputs.depth);
            } else {
                raster.read(outputs.depth);
                raster.read(outputs.triangles);
                raster.write(outputs.triangles);
            }
            raster.record([outputs, arguments, writeTriangles, viewProjection = view.viewProjection,
                           extent = view.extent](PassContext &context) {
                auto &device = context.device();
                auto const &scene = context.scene();
                auto kernel = kernels::RasterVisibility{};
                device.useKernel(kernel);
                bind(device, kernel.instances, *scene.instances);
                bind(device, kernel.meshes, *scene.meshes);
                bind(device, kernel.vertices, *scene.vertices);
                bind(device, kernel.indices, *scene.indices);
                bind(device, kernel.materials, *scene.materials);
                bind(device, kernel.visible, context.buffer(outputs.visible));
                bind(device, kernel.depth, context.buffer(outputs.depth));
                bind(device, kernel.triangles, context.buffer(outputs.triangles));
                set(device, kernel.viewProjection, viewProjection);
                set(device, kernel.size, extent);
                set(device, kernel.writeTriangles, writeTriangles);
                context.dispatchIndirect(kernel, arguments);
            });
        }

        auto &stats = graph.addPass("visibility statistics");
        stats.read(outputs.visible);
        stats.write(totals);
        stats.record([outputs, totals](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::CopyVisibleTotals{};
            device.useKernel(kernel);
            bind(device, kernel.visible, context.buffer(outputs.visible));
            bind(device, kernel.totals, context.buffer(totals));
            context.dispatch(kernel, tc::uvec3{1, 1, 1});
        });
        graph.readback<tc::uvec2>(totals, [](tc::uvec2 const *data, std::size_t, ViewResult &result) {
            result.stats.visibleInstances = data[0].x;
            result.stats.visibleTriangles = data[0].y;
        });

        graph.blackboard().put(outputs);
    }
};

struct ShadingFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Shading; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const visibility = graph.blackboard().get<VisibilityOutputs>();
        auto &pass = graph.addPass("shade visibility");
        pass.read(visibility.visible);
        pass.read(visibility.triangles);
        pass.write(view.sceneColor);
        pass.record([visibility, color = view.sceneColor, extent = view.extent,
                     inverseViewProjection = view.inverseViewProjection,
                     environment = tc::vec4{view.environment.radiance, 0.0f},
                     lightCount = static_cast<tc::uint>(view.lightCount)](PassContext &context) {
            auto &device = context.device();
            auto const &scene = context.scene();
            auto kernel = kernels::ShadeVisibility{};
            device.useKernel(kernel);
            bind(device, kernel.color, context.image(color));
            bind(device, kernel.instances, *scene.instances);
            bind(device, kernel.meshes, *scene.meshes);
            bind(device, kernel.vertices, *scene.vertices);
            bind(device, kernel.indices, *scene.indices);
            bind(device, kernel.materials, *scene.materials);
            bind(device, kernel.lights, *scene.lights);
            bind(device, kernel.visible, context.buffer(visibility.visible));
            bind(device, kernel.triangles, context.buffer(visibility.triangles));
            set(device, kernel.inverseViewProjection, inverseViewProjection);
            set(device, kernel.environment, environment);
            set(device, kernel.size, extent);
            set(device, kernel.lightCount, lightCount);
            context.dispatch(kernel, groupsCovering(extent));
        });
    }
};

struct PathTraceFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::Shading; }
    SceneRequirements requirements() const noexcept override { return SceneRequirements{true}; }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        auto const accumulation = graph.history<AccumulationFormat>("path trace accumulation", view.extent);
        auto const samples = samplesBefore(view, accumulation.valid);
        auto &pass = graph.addPass("path trace");
        pass.read(accumulation.previous);
        pass.write(accumulation.current);
        pass.write(view.sceneColor);
        pass.record([accumulation, samples, color = view.sceneColor, extent = view.extent,
                     inverseViewProjection = view.inverseViewProjection,
                     environment = tc::vec4{view.environment.radiance, 0.0f},
                     lightCount = static_cast<tc::uint>(view.lightCount)](PassContext &context) {
            auto &device = context.device();
            auto const &scene = context.scene();
            if (scene.tlasNodes == nullptr) {
                throw std::logic_error{"lr::render: the path tracer ran before the scene had hierarchies"};
            }
            auto kernel = kernels::PathTrace{};
            device.useKernel(kernel);
            bind(device, kernel.previous, context.image(accumulation.previous));
            bind(device, kernel.current, context.image(accumulation.current));
            bind(device, kernel.color, context.image(color));
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
        graph.onFinish([total = samples + 1](ViewResult &result) { result.stats.accumulatedSamples = total; });
    }

private:
    // What the samples averaged so far show.
    struct Accumulation {
        core::CameraDesc camera;
        core::ivec2 extent;
        std::uint64_t sceneVersion;
        std::uint32_t samples;
    };

    // The samples taken before this frame's, none when the view shows something else now.
    std::uint32_t samplesBefore(ViewContext const &view, bool const historyValid) {
        auto const found = accumulations.find(view.view);
        auto const continues = historyValid and found != accumulations.end()
                           and sameCamera(found->second.camera, view.camera) and found->second.extent == view.extent
                           and found->second.sceneVersion == view.sceneVersion;
        auto const before = continues ? found->second.samples : 0u;
        accumulations[view.view] = Accumulation{view.camera, view.extent, view.sceneVersion, before + 1};
        return before;
    }

    std::unordered_map<ViewId, Accumulation> accumulations;
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
        graph.readback<tc::uint>(view.output,
                                 [extent = view.extent](tc::uint const *data, std::size_t const count, ViewResult &result) {
                                     result.image = ImageRgba8{extent, std::vector<std::uint32_t>(data, data + count)};
                                 });
    }
};

} // namespace

std::shared_ptr<RenderFeature> makeVisibilityFeature() { return std::make_shared<VisibilityFeature>(); }
std::shared_ptr<RenderFeature> makeShadingFeature() { return std::make_shared<ShadingFeature>(); }
std::shared_ptr<RenderFeature> makeTonemapFeature() { return std::make_shared<TonemapFeature>(); }
std::shared_ptr<RenderFeature> makePathTraceFeature() { return std::make_shared<PathTraceFeature>(); }

} // namespace lr::render
