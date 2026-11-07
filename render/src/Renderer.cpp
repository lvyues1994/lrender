#include <lr/render/Renderer.h>

#include "FrameGraphImpl.h"
#include "FrameQueue.h"
#include "KernelVariants.h"
#include "PreparedFrame.h"
#include "RenderWorld.h"
#include "Temporal.h"
#include "kernels/Shading.h"

#include <lr/gpu/GpuQueue.h>
#include <lr/render/Features.h>
#include <lr/render/MaterialModel.h>
#include <lr/runtime/SerialThread.h>

#include <lexec/scopes/counting_scope.hpp>
#include <stackfull/exec/sync.hpp>
#include <stackfull/sync/Completion.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace lr::render {

namespace {

// Frames a view may go without using a history before it is released.
constexpr auto kHistoryKeepFrames = std::uint64_t{60};
// A device generation no device has: the next frame uploads the whole scene.
constexpr auto kSceneOutOfStep = ~std::uint64_t{0};

struct TargetState {
    core::ivec2 extent{0};
    std::shared_ptr<Presenter> presenter; // window targets
    bool sharesBuffers = false;           // the presenter's SharedBuffer path
};

// Whether frames of `device` can go to `presenter` through its SharedBuffer path.
bool sharedPath(Presenter const &presenter, tc::DeviceInfo const &device) {
    return device.syncFiles and device.importsMemory and presenter.sharesBuffersWith(device);
}

struct TargetsImpl final : RenderTargets {
    explicit TargetsImpl(tc::DeviceInfo const &device_) : device(device_) {}

    RenderTargetId createWindow(WindowTargetDesc const &desc) override {
        if (desc.presenter == nullptr) {
            throw std::invalid_argument{"lr::render: a window target needs a presenter"};
        }
        return add(TargetState{desc.extent, desc.presenter, sharedPath(*desc.presenter, device)});
    }

    RenderTargetId createImage(ImageTargetDesc const &desc) override {
        return add(TargetState{desc.extent, nullptr, false});
    }

    void resize(RenderTargetId const id, core::ivec2 const extent) override {
        checkExtent(extent);
        auto const lock = std::lock_guard<std::mutex>{mutex};
        find(id).extent = extent;
    }

    void destroy(RenderTargetId const id) override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        find(id);
        targets.erase(id);
    }

    TargetState stateOf(RenderTargetId const id) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return find(id);
    }

    bool exists(RenderTargetId const id) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return targets.count(id) != 0;
    }

private:
    static void checkExtent(core::ivec2 const extent) {
        if (extent.x <= 0 or extent.y <= 0) {
            throw std::invalid_argument{"lr::render: render targets need a positive extent"};
        }
    }

    RenderTargetId add(TargetState state) {
        checkExtent(state.extent);
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto const id = RenderTargetId{next++};
        targets.emplace(id, std::move(state));
        return id;
    }

    TargetState &find(RenderTargetId const id) {
        auto const found = targets.find(id);
        if (found == targets.end()) {
            throw std::invalid_argument{"lr::render: unknown render target " + std::to_string(id.value)};
        }
        return found->second;
    }

    tc::DeviceInfo device;
    std::mutex mutex;
    std::unordered_map<RenderTargetId, TargetState> targets;
    std::uint32_t next = 0;
};

using FeatureList = std::vector<std::shared_ptr<RenderFeature>>;
using ModelList = std::vector<std::shared_ptr<MaterialModel>>;

struct PipelinesImpl final : PipelineRegistry {
    PipelinesImpl() {
        pipelines.push_back(FeatureList{makeVisibilityFeature(), makeShadowFeature(), makeShadingFeature(),
                                        makeTransparencyFeature(), makeTemporalAntiAliasingFeature(),
                                        makeTonemapFeature()});
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

// Why the device cannot run `program`; on the GPU thread. Without a device (it was lost
// and could not be recreated yet) only the limits of the first one tell.
std::optional<std::string> whyDeviceCannotRun(gpu::GpuQueue const &gpu, gpu::GpuContext const &context,
                                              std::string const &program) {
    if (context.device != nullptr) {
        return context.device->checkKernel(program);
    }
    auto const found = tc::ProgramRegistry::instance().find(program);
    if (not found) {
        return "kernel '" + program + "' was not transpiled";
    }
    return tc::exceedsLimits(*found, gpu.info().limits);
}

struct MaterialModelsImpl final : MaterialModels {
    explicit MaterialModelsImpl(gpu::GpuQueue &gpu_) : gpu(gpu_) {
        models.push_back(makeMaterialModel<kernels::MetallicRoughness>());
    }

    core::MaterialModelId add(std::shared_ptr<MaterialModel> model) override {
        if (model == nullptr) {
            throw std::invalid_argument{"lr::render: a material model is null"};
        }
        auto const program = std::string{model->program()};
        auto const checked = lexec::sync_wait(gpu.record(
            [this, program](gpu::GpuContext &context) { return whyDeviceCannotRun(gpu, context, program); }));
        if (auto const &reason = std::get<0>(checked.value())) {
            throw tc::DeviceError{"lr::render: the device cannot run material model kernel " + program + " (" + *reason
                                  + ")"};
        }
        auto const lock = std::lock_guard<std::mutex>{mutex};
        if (models.size() >= maxMaterialModels()) {
            throw std::length_error{"lr::render: a renderer holds at most " + std::to_string(maxMaterialModels())
                                    + " material models"};
        }
        models.push_back(std::move(model));
        return core::MaterialModelId{static_cast<std::uint32_t>(models.size() - 1)};
    }

    std::uint32_t count() const override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return static_cast<std::uint32_t>(models.size());
    }

    ModelList snapshot() const {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return models;
    }

    std::vector<std::string_view> programs() const {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto result = std::vector<std::string_view>{};
        for (auto const &model : models) {
            result.push_back(model->program());
        }
        return result;
    }

private:
    gpu::GpuQueue &gpu;
    mutable std::mutex mutex;
    ModelList models;
};

// What submit() captures of a view: its target and the pipeline's features as they are
// at the call.
struct ViewPlan {
    ViewDesc desc;
    TargetState target;
    FeatureList features;
};

bool tracesRays(std::vector<ViewPlan> const &plans) {
    return std::any_of(plans.begin(), plans.end(), [](ViewPlan const &plan) {
        return std::any_of(plan.features.begin(), plan.features.end(),
                           [](auto const &feature) { return feature->requirements().rayTracing; });
    });
}

bool jitters(ViewPlan const &plan) {
    return plan.desc.settings.antiAliasing == AntiAliasing::Taa
       and std::any_of(plan.features.begin(), plan.features.end(),
                       [](auto const &feature) { return feature->requirements().temporalJitter; });
}

RendererCapabilities chooseFor(gpu::GpuQueue &gpu, RendererConfig const &config) {
    auto const choices = VariantChoices{config.subgroups, config.singlePassVisibility, gpu.info().limits.subgroupSize,
                                        config.rayQueries};
    auto const chosen = lexec::sync_wait(gpu.record([choices](gpu::GpuContext &context) {
        return chooseCapabilities(*makeKernelCheck(*context.device), choices);
    }));
    return std::get<0>(chosen.value());
}

ViewContext makeViewContext(ViewPlan const &plan, RenderWorld const &world) {
    auto const extent = plan.target.extent;
    auto context = ViewContext{};
    context.view = plan.desc.id;
    context.extent = extent;
    context.camera = plan.desc.camera;
    context.settings = plan.desc.settings;
    context.environment = world.environment();
    auto const aspect = static_cast<float>(extent.x) / static_cast<float>(extent.y);
    context.viewMatrix = core::viewMatrix(plan.desc.camera);
    context.projection = core::projectionMatrix(plan.desc.camera, aspect);
    context.viewProjection = context.projection * context.viewMatrix;
    context.inverseViewProjection = tc::inverse(context.viewProjection);
    context.instanceCount = world.instanceCount();
    context.lightCount = world.lightCount();
    context.transparentInstanceCount = world.transparentInstanceCount();
    context.clusterInstanceCount = world.clusterInstanceCount();
    context.sceneBounds = world.bounds();
    context.shadowLight = world.shadowLight();
    context.sceneVersion = world.version();
    return context;
}

using OutputHandler = std::function<void(tc::uint const *data, std::size_t count, ViewResult &)>;

// Where the final pass's pixels go once the frame is done: into the ViewResult of an
// image target, to the presenter of a window target.
OutputHandler deliverOutput(ViewPlan const &plan, std::uint64_t const frameIndex) {
    auto const extent = plan.target.extent;
    if (plan.target.presenter == nullptr) {
        return [extent](tc::uint const *data, std::size_t const count, ViewResult &result) {
            result.image = ImageRgba8{extent, std::vector<std::uint32_t>(data, data + count)};
        };
    }
    auto const frame = PresentFrame{frameIndex, plan.desc.id, plan.desc.target, ImageRgba8View{extent, nullptr}};
    return [presenter = plan.target.presenter, frame](tc::uint const *data, std::size_t, ViewResult &) {
        auto shown = frame;
        shown.pixels.pixels = data;
        presenter->present(shown);
    };
}

std::exception_ptr asException(std::exception_ptr error) noexcept { return error; }

std::exception_ptr asException(std::error_code const error) {
    return std::make_exception_ptr(std::system_error{error, "lr::render: the frame was lost"});
}

std::exception_ptr frameStopped() noexcept {
    return std::make_exception_ptr(std::runtime_error{"lr::render: the frame was stopped before the GPU thread"});
}

std::uint32_t checkedFramesInFlight(RendererConfig const &config) {
    if (config.framesInFlight == 0) {
        throw std::invalid_argument{"lr::render: framesInFlight must be at least 1"};
    }
    return config.framesInFlight;
}

struct RendererImpl final : Renderer {
    explicit RendererImpl(RendererConfig const &config)
        : framesInFlight(checkedFramesInFlight(config)),
          gpu(gpu::makeGpuQueue(gpu::GpuQueueConfig{config.backend, config.deviceNameContains, config.runtime})),
          timers(config.gpuTimers and gpu->info().timers),
          cullClusters(config.cullClusters),
          cullOccluded(config.cullOccluded),
          chosen(chooseFor(*gpu, config)),
          models(*gpu),
          targetSet(gpu->info()),
          worldParallel(makeFramePoolFor(*config.runtime)),
          conversions(makeBackgroundConversions(*config.runtime)),
          world(textureLimitsOf(gpu->info().limits, config.maxTextureSize), *worldParallel, *conversions,
                config.textureConversions),
          frames(FrameQueueConfig{gpu.get(), &scope, &deviceGeneration, framesInFlight, timers,
                                  config.textureStreamingBytes, config.aliasGraphResources,
                                  [this](tc::Device &device) {
                                      verifyCapabilities(*makeKernelCheck(device), chosen, models.programs());
                                  }}) {
        if (timers) {
            static_cast<void>(
                lexec::sync_wait(gpu->record([](gpu::GpuContext &context) { context.device->enableTimers(true); })));
        }
    }

    ~RendererImpl() override {
        static_cast<void>(lexec::sync_wait(scope.join()));
        static_cast<void>(lexec::sync_wait(gpu->record([this](gpu::GpuContext &) noexcept { frames.clear(); })));
    }

    tc::DeviceInfo const &deviceInfo() const noexcept override { return gpu->info(); }
    RendererCapabilities const &capabilities() const noexcept override { return chosen; }
    RenderTargets &targets() noexcept override { return targetSet; }
    PipelineRegistry &pipelines() noexcept override { return pipelineSet; }
    MaterialModels &materialModels() noexcept override { return models; }

    // The frame's work runs whether or not its sender is started: the render thread and
    // the GPU thread see every frame, in order, and the sender only waits for the result.
    FrameSender submit(FrameDesc frame) override {
        auto plans = std::vector<ViewPlan>{};
        plans.reserve(frame.views.size());
        for (auto const &view : frame.views) {
            plans.push_back(ViewPlan{view, targetSet.stateOf(view.target), pipelineSet.featuresOf(view.pipeline)});
        }
        auto pending = PreparedFrame{};
        pending.submitted = std::chrono::steady_clock::now();
        auto const completion = pending.completion;
        auto const lock = std::lock_guard<std::mutex>{submitMutex};
        pending.index = nextFrame++;
        auto work =
            lexec::schedule(renderThread.scheduler())
            | lexec::then([this, pending = std::move(pending), delta = std::move(frame.delta),
                           plans = std::move(plans)]() mutable {
                  auto const start = std::chrono::steady_clock::now();
                  auto ready = prepare(std::move(pending), delta, plans);
                  ready.prepareTime = std::chrono::steady_clock::now() - start;
                  return ready;
              })
            | lexec::let_value([this](PreparedFrame &ready) {
                  return gpu->record([this, ready = std::move(ready)](gpu::GpuContext &context) mutable noexcept {
                      frames.accept(context, std::move(ready));
                  });
              })
            | lexec::upon_error([completion](std::exception_ptr const &failure) noexcept { completion.setException(failure); })
            | lexec::upon_stopped([completion]() noexcept { completion.setException(frameStopped()); });
        lexec::spawn(std::move(work), scope.get_token());
        return FrameSender{stackfull::exec::async_get(completion)
                           | lexec::let_error([](auto &error) { return lexec::just_error(asException(error)); })};
    }

    KernelReloadSender reloadKernels(std::string directory) override {
        return gpu->record([directory = std::move(directory)](gpu::GpuContext &) {
            return tc::ProgramRegistry::instance().reloadFrom(directory);
        });
    }

private:
    // On the render thread. A delta the world rejects fails the frame and changes nothing;
    // anything failing later leaves the world ahead of the GPU copy, which the next frame
    // then uploads whole.
    PreparedFrame prepare(PreparedFrame frame, core::RenderWorldDelta const &delta, std::vector<ViewPlan> const &plans) {
        frame.deviceGeneration = deviceGeneration.load(std::memory_order_acquire);
        // Before the delta's textures, which go to the new device.
        if (frame.deviceGeneration != sceneGeneration) {
            world.loseTextures();
        }
        auto const modelList = models.snapshot();
        world.allowMaterialModels(static_cast<std::uint32_t>(modelList.size()));
        try {
            frame.scene = world.apply(delta);
        } catch (...) {
            frame.failure = std::current_exception();
            return frame;
        }
        try {
            prepareViews(frame, plans, modelList);
        } catch (...) {
            sceneGeneration = kSceneOutOfStep;
            frame.failure = std::current_exception();
            frame.scene = SceneUpdate{};
            frame.views.clear();
        }
        return frame;
    }

    void prepareViews(PreparedFrame &frame, std::vector<ViewPlan> const &plans, ModelList const &modelList) {
        if (frame.deviceGeneration != sceneGeneration) {
            auto full = world.fullUpdate();
            full.pools = std::move(frame.scene.pools);
            full.removedTextures = std::move(frame.scene.removedTextures);
            full.textureUploads = std::move(frame.scene.textureUploads);
            frame.scene = std::move(full);
            histories.restart();
            temporalViews.restart();
        }
        frame.lostTextures = world.takeLostTextures();
        if (tracesRays(plans)) {
            if (chosen.rayQueries) {
                frame.tracesWithRayQueries = true;
            } else {
                world.keepHierarchies(frame.scene);
            }
        }
        for (auto const &plan : plans) {
            frame.views.push_back(prepareView(plan, frame.index, modelList));
        }
        frame.releasedHistories =
            histories.expire(frame.index, [this](RenderTargetId const target) { return targetSet.exists(target); });
        temporalViews.expire(frame.index);
        sceneGeneration = frame.deviceGeneration;
    }

    ViewContext viewContextOf(ViewPlan const &plan, std::uint64_t const frameIndex, ModelList const &modelList) {
        auto context = makeViewContext(plan, world);
        context.capabilities = chosen;
        context.cullClusters = cullClusters;
        context.cullOccluded = cullOccluded;
        context.materialModels = modelList;
        if (jitters(plan)) {
            auto const temporal = temporalViews.advance(TemporalRequest{plan.desc.id, frameIndex, context.extent,
                                                                        context.viewProjection, plan.desc.cameraCut});
            context.viewProjection = jitteredViewProjection(context.viewProjection, temporal.jitter, context.extent);
            context.inverseViewProjection = tc::inverse(context.viewProjection);
            context.temporal = temporal;
        }
        return context;
    }

    // A view whose graph fails to build fails the frame on the GPU thread.
    PreparedView prepareView(ViewPlan const &plan, std::uint64_t const frameIndex, ModelList const &modelList) {
        auto prepared = PreparedView{plan.desc.id, plan.desc.target, std::nullopt, nullptr, std::nullopt};
        try {
            auto builder = GraphBuilder{GraphOwner{&histories, plan.desc.id, plan.desc.target, frameIndex}};
            auto context = viewContextOf(plan, frameIndex, modelList);
            auto const extent = plan.target.extent;
            auto const pixels = static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y);
            auto const shared = plan.target.sharesBuffers;
            context.sceneColor = builder.createImage<HdrFormat>("scene colour", extent);
            context.output = shared ? builder.createExternalBuffer<tc::uint>("output", pixels)
                                    : builder.createReadbackBuffer<tc::uint>("output", pixels);
            for (auto const &feature : plan.features) {
                feature->setup(builder, context);
            }
            if (shared) {
                prepared.sharedOutput = SharedOutput{plan.target.presenter, extent, context.output.index};
            } else {
                builder.readback<tc::uint>(context.output, deliverOutput(plan, frameIndex));
            }
            prepared.graph = std::move(builder).compile();
            prepared.graph->describe = plan.desc.describeGraph;
        } catch (...) {
            prepared.failure = std::current_exception();
        }
        return prepared;
    }

    std::uint32_t framesInFlight;
    std::unique_ptr<gpu::GpuQueue> gpu;
    bool timers;
    bool cullClusters;
    bool cullOccluded;
    RendererCapabilities chosen;
    MaterialModelsImpl models;
    runtime::SerialThread renderThread;
    TargetsImpl targetSet;
    PipelinesImpl pipelineSet;
    std::unique_ptr<ParallelFor> worldParallel;        // the world's, on the frame pool
    std::unique_ptr<ImageConversions> conversions;     // the world's, on the background pool
    RenderWorld world;                                 // render thread
    FrameHistories histories{kHistoryKeepFrames};      // render thread
    TemporalViews temporalViews{kHistoryKeepFrames};   // render thread
    std::uint64_t sceneGeneration = 0;            // render thread: the device the GPU copy is for
    std::atomic<std::uint64_t> deviceGeneration{0};
    std::mutex submitMutex;
    std::uint64_t nextFrame = 0;
    lexec::counting_scope scope;
    FrameQueue frames; // GPU thread
};

} // namespace

std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config) { return std::make_unique<RendererImpl>(config); }

std::optional<std::string> builtInKernelDirectory() {
#ifdef LR_BUILT_IN_KERNEL_DIRECTORY
    return std::string{LR_BUILT_IN_KERNEL_DIRECTORY};
#else
    return std::nullopt;
#endif
}

} // namespace lr::render
