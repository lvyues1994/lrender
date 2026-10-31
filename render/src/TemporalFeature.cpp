#include <lr/render/Features.h>

#include "PassHelpers.h"
#include "kernels/Temporal.h"

#include <algorithm>
#include <array>

namespace lr::render {

namespace {

struct TemporalAntiAliasingFeature final : RenderFeature {
    FeatureStage stage() const noexcept override { return FeatureStage::PostProcess; }

    FeatureRequirements requirements() const noexcept override {
        auto requirements = FeatureRequirements{};
        requirements.temporalJitter = true;
        return requirements;
    }

    void setup(FrameGraphBuilder &graph, ViewContext const &view) override {
        if (not view.temporal or view.settings.antiAliasing != AntiAliasing::Taa) {
            return;
        }
        auto const &temporal = *view.temporal;
        auto const visibility = graph.blackboard().get<VisibilityOutputs>();
        auto const history = graph.history<HdrFormat>("temporal anti-aliasing", view.extent);
        auto const matrices = graph.createUploadBuffer<tc::mat4>("temporal matrices", 3);
        auto const motion = graph.createBuffer<tc::vec2>("motion", pixelsOf(view.extent));

        auto &motionPass = graph.addPass("temporal motion");
        motionPass.read(visibility.visible);
        motionPass.read(visibility.triangles);
        motionPass.write(matrices);
        motionPass.write(motion);
        auto const views =
            std::array<core::mat4, 3>{view.inverseViewProjection, temporal.viewProjection, temporal.previousViewProjection};
        motionPass.record([visibility, matrices, motion, views, extent = view.extent](PassContext &context) {
            auto &device = context.device();
            auto const &scene = context.scene();
            auto &matrixBuffer = context.buffer(matrices);
            std::copy(views.begin(), views.end(), matrixBuffer.data());
            device.uploadBuffer(matrixBuffer);
            auto kernel = kernels::TemporalMotion{};
            device.useKernel(kernel);
            bind(device, kernel.instances, *scene.instances);
            bind(device, kernel.vertices, *scene.vertices);
            bind(device, kernel.indices, *scene.indices);
            bind(device, kernel.visible, context.buffer(visibility.visible));
            bind(device, kernel.triangles, context.buffer(visibility.triangles));
            bind(device, kernel.matrices, matrixBuffer);
            bind(device, kernel.motion, context.buffer(motion));
            set(device, kernel.size, extent);
            context.dispatch(kernel, groupsCovering(extent));
        });

        auto &resolve = graph.addPass("temporal resolve");
        resolve.read(view.sceneColor);
        resolve.read(history.previous);
        resolve.read(motion);
        resolve.write(history.current);
        resolve.record([color = view.sceneColor, history, motion, extent = view.extent,
                        valid = (history.valid and temporal.continues) ? 1u : 0u](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::TemporalResolve{};
            device.useKernel(kernel);
            bind(device, kernel.color, context.image(color));
            bind(device, kernel.resolved, context.image(history.current));
            bind(device, kernel.history, context.image(history.previous));
            bind(device, kernel.motion, context.buffer(motion));
            set(device, kernel.size, extent);
            set(device, kernel.historyValid, valid);
            context.dispatch(kernel, groupsCovering(extent));
        });

        auto &show = graph.addPass("show resolved");
        show.read(history.current);
        show.write(view.sceneColor);
        show.record([color = view.sceneColor, resolved = history.current, extent = view.extent](PassContext &context) {
            auto &device = context.device();
            auto kernel = kernels::ShowResolved{};
            device.useKernel(kernel);
            bind(device, kernel.resolved, context.image(resolved));
            bind(device, kernel.color, context.image(color));
            set(device, kernel.size, extent);
            context.dispatch(kernel, groupsCovering(extent));
        });
        graph.blackboard().put(TemporalOutputs{motion});
    }
};

} // namespace

std::shared_ptr<RenderFeature> makeTemporalAntiAliasingFeature() {
    return std::make_shared<TemporalAntiAliasingFeature>();
}

} // namespace lr::render
