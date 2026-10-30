#include <lr/render/Renderer.h>

#include "DeviceScene.h"
#include "FrameGraphImpl.h"
#include "RenderWorld.h"

#include <lr/gpu/GpuQueue.h>
#include <lr/render/Features.h>
#include <lr/runtime/SerialThread.h>

#include <lexec/scopes/counting_scope.hpp>

#include <algorithm>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace lr::render {

namespace {

// Frames a view may go without using a history before it is released.
constexpr auto kHistoryKeepFrames = std::uint64_t{60};

struct TargetsImpl final : RenderTargets {
    RenderTargetId createImage(ImageTargetDesc const &desc) override {
        checkExtent(desc.extent);
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto const id = RenderTargetId{next++};
        extents[id] = desc.extent;
        return id;
    }

    void resize(RenderTargetId const id, core::ivec2 const extent) override {
        checkExtent(extent);
        auto const lock = std::lock_guard<std::mutex>{mutex};
        find(id) = extent;
    }

    void destroy(RenderTargetId const id) override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        find(id);
        extents.erase(id);
    }

    core::ivec2 extentOf(RenderTargetId const id) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return find(id);
    }

    bool exists(RenderTargetId const id) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return extents.count(id) != 0;
    }

private:
    static void checkExtent(core::ivec2 const extent) {
        if (extent.x <= 0 or extent.y <= 0) {
            throw std::invalid_argument{"lr::render: render targets need a positive extent"};
        }
    }

    core::ivec2 &find(RenderTargetId const id) {
        auto const found = extents.find(id);
        if (found == extents.end()) {
            throw std::invalid_argument{"lr::render: unknown render target " + std::to_string(id.value)};
        }
        return found->second;
    }

    std::mutex mutex;
    std::unordered_map<RenderTargetId, core::ivec2> extents;
    std::uint32_t next = 0;
};

using FeatureList = std::vector<std::shared_ptr<RenderFeature>>;

struct PipelinesImpl final : PipelineRegistry {
    PipelinesImpl() {
        pipelines.push_back(FeatureList{makeVisibilityFeature(), makeShadingFeature(), makeTonemapFeature()});
        pipelines.push_back(FeatureList{makePathTraceFeature(), makeTonemapFeature()});
    }

    PipelineId visibility() const noexcept override { return PipelineId{0}; }
    PipelineId pathTraceReference() const noexcept override { return PipelineId{1}; }

    PipelineId create(PipelineDesc desc) override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto features = featuresLocked(desc.base);
        for (auto &feature : desc.features) {
            if (feature == nullptr) {
                throw std::invalid_argument{"lr::render: a pipeline feature is null"};
            }
            features.push_back(std::move(feature));
        }
        std::stable_sort(features.begin(), features.end(), [](auto const &a, auto const &b) {
            return a->stage() < b->stage();
        });
        pipelines.push_back(std::move(features));
        return PipelineId{static_cast<std::uint32_t>(pipelines.size() - 1)};
    }

    // An invalid id selects the visibility pipeline.
    FeatureList featuresOf(PipelineId const id) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return featuresLocked(id.isValid() ? id : visibility());
    }

private:
    FeatureList featuresLocked(PipelineId const id) const {
        if (id.value >= pipelines.size()) {
            throw std::invalid_argument{"lr::render: unknown pipeline " + std::to_string(id.value)};
        }
        return pipelines[id.value];
    }

    std::mutex mutex;
    std::vector<FeatureList> pipelines;
};

// What submit() captures of a view: the target's extent and the pipeline's features as
// they are at the call.
struct ViewPlan {
    ViewDesc desc;
    core::ivec2 extent;
    FeatureList features;
};

struct PreparedView {
    ViewId view;
    RenderTargetId target;
    std::optional<CompiledGraph> graph;
    std::exception_ptr failure;
};

struct PreparedFrame {
    std::uint64_t index = 0;
    SceneUpdate scene;
    std::vector<PreparedView> views;
    std::vector<std::uint64_t> releasedHistories;
};

bool tracesRays(std::vector<ViewPlan> const &plans) {
    return std::any_of(plans.begin(), plans.end(), [](ViewPlan const &plan) {
        return std::any_of(plan.features.begin(), plan.features.end(),
                           [](auto const &feature) { return feature->requirements().rayTracing; });
    });
}

ViewContext makeViewContext(ViewPlan const &plan, RenderWorld const &world) {
    auto context = ViewContext{};
    context.view = plan.desc.id;
    context.extent = plan.extent;
    context.camera = plan.desc.camera;
    context.settings = plan.desc.settings;
    context.environment = world.environment();
    auto const aspect = static_cast<float>(plan.extent.x) / static_cast<float>(plan.extent.y);
    context.viewMatrix = core::viewMatrix(plan.desc.camera);
    context.projection = core::projectionMatrix(plan.desc.camera, aspect);
    context.viewProjection = context.projection * context.viewMatrix;
    context.inverseViewProjection = tc::inverse(context.viewProjection);
    context.instanceCount = world.instanceCount();
    context.lightCount = world.lightCount();
    context.sceneVersion = world.version();
    return context;
}

// Whether the timings are those of the runs' dispatches: the device lists them in
// execution order, the runs in recording order, and the two agree unless a pass
// dispatched behind PassContext's back.
bool timingsMatch(std::vector<GraphRun> const &runs, std::vector<tc::KernelTiming> const &timings) {
    auto next = std::size_t{0};
    for (auto const &run : runs) {
        for (auto const &dispatch : run.dispatches()) {
            if (next == timings.size() or timings[next].program != dispatch.program) {
                return false;
            }
            ++next;
        }
    }
    return next == timings.size();
}

struct RendererImpl final : Renderer {
    explicit RendererImpl(RendererConfig const &config)
        : gpu(gpu::makeGpuQueue(gpu::GpuQueueConfig{config.backend, config.deviceNameContains, config.runtime})),
          timers(config.gpuTimers and gpu->info().timers) {
        if (timers) {
            static_cast<void>(
                lexec::sync_wait(gpu->record([](gpu::GpuContext &context) { context.device->enableTimers(true); })));
        }
    }

    ~RendererImpl() override {
        static_cast<void>(lexec::sync_wait(scope.join()));
        static_cast<void>(lexec::sync_wait(gpu->record([this](gpu::GpuContext &) noexcept {
            pool.clear();
            historyImages.clear();
            deviceScene = DeviceScene{};
        })));
    }

    tc::DeviceInfo const &deviceInfo() const noexcept override { return gpu->info(); }
    RenderTargets &targets() noexcept override { return targetSet; }
    PipelineRegistry &pipelines() noexcept override { return pipelineSet; }

    FrameSender submit(FrameDesc frame) override {
        auto plans = std::vector<ViewPlan>{};
        plans.reserve(frame.views.size());
        for (auto const &view : frame.views) {
            plans.push_back(ViewPlan{view, targetSet.extentOf(view.target), pipelineSet.featuresOf(view.pipeline)});
        }
        auto const lock = std::lock_guard<std::mutex>{submitMutex};
        auto const index = nextFrame++;
        auto work =
            lexec::schedule(renderThread.scheduler())
            | lexec::then([this, index, delta = std::move(frame.delta), plans = std::move(plans)] {
                  return std::make_shared<PreparedFrame>(prepare(index, delta, plans));
              })
            | lexec::let_value([this](std::shared_ptr<PreparedFrame> &prepared) {
                  return gpu->record([this, prepared](gpu::GpuContext &context) { return execute(context, *prepared); });
              });
        return FrameSender{lexec::spawn_future(std::move(work), scope.get_token())};
    }

private:
    // On the render thread. A view whose graph fails to build fails the frame on the GPU
    // thread, after the scene update has been applied there, so both copies of the scene
    // stay in step.
    PreparedFrame prepare(std::uint64_t const index, core::RenderWorldDelta const &delta,
                          std::vector<ViewPlan> const &plans) {
        auto frame = PreparedFrame{index, world.apply(delta), {}, {}};
        if (tracesRays(plans)) {
            world.keepHierarchies(frame.scene);
        }
        for (auto const &plan : plans) {
            auto prepared = PreparedView{plan.desc.id, plan.desc.target, std::nullopt, nullptr};
            try {
                auto builder = GraphBuilder{GraphOwner{&histories, plan.desc.id, plan.desc.target, index}};
                auto context = makeViewContext(plan, world);
                auto const pixels = static_cast<std::size_t>(plan.extent.x) * static_cast<std::size_t>(plan.extent.y);
                context.sceneColor = builder.createImage<HdrFormat>("scene colour", plan.extent);
                context.output = builder.createReadbackBuffer<tc::uint>("output", pixels);
                for (auto const &feature : plan.features) {
                    feature->setup(builder, context);
                }
                prepared.graph = std::move(builder).compile();
            } catch (...) {
                prepared.failure = std::current_exception();
            }
            frame.views.push_back(std::move(prepared));
        }
        frame.releasedHistories =
            histories.expire(index, [this](RenderTargetId const target) { return targetSet.exists(target); });
        return frame;
    }

    // On the GPU thread.
    FrameResult execute(gpu::GpuContext &context, PreparedFrame &frame) {
        auto &device = *context.device;
        deviceScene.apply(device, frame.scene);
        historyImages.release(frame.releasedHistories);
        for (auto const &view : frame.views) {
            if (view.failure) {
                std::rethrow_exception(view.failure);
            }
        }
        auto const tables = deviceScene.tables();
        auto runs = std::vector<GraphRun>{};
        runs.reserve(frame.views.size());
        for (auto &view : frame.views) {
            runs.emplace_back(std::move(*view.graph), pool, historyImages, device);
            runs.back().record(device, tables);
        }
        auto const submission = device.submit();
        device.finish();
        auto const timings = timers ? device.takeTimings(submission) : std::vector<tc::KernelTiming>{};
        auto const timed = timers and timingsMatch(runs, timings);
        auto result = FrameResult{frame.index, {}};
        auto firstTiming = std::size_t{0};
        for (auto index = std::size_t{0}; index < runs.size(); ++index) {
            auto view = ViewResult{};
            view.view = frame.views[index].view;
            view.target = frame.views[index].target;
            if (timed) {
                view.passTimings = runs[index].passTimings(timings, firstTiming);
                firstTiming += runs[index].dispatches().size();
            }
            runs[index].finish(view);
            result.views.push_back(std::move(view));
        }
        return result;
    }

    std::unique_ptr<gpu::GpuQueue> gpu;
    bool timers;
    runtime::SerialThread renderThread;
    TargetsImpl targetSet;
    PipelinesImpl pipelineSet;
    RenderWorld world;                              // render thread
    FrameHistories histories{kHistoryKeepFrames};   // render thread
    DeviceScene deviceScene;                        // GPU thread
    ResourcePool pool;                              // GPU thread
    HistoryResources historyImages;                 // GPU thread
    std::mutex submitMutex;
    std::uint64_t nextFrame = 0;
    lexec::counting_scope scope;
};

} // namespace

std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config) { return std::make_unique<RendererImpl>(config); }

} // namespace lr::render
