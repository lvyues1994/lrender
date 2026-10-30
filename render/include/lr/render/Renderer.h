#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/render/FrameGraph.h>
#include <lr/render/Presenter.h>
#include <lr/render/View.h>
#include <lr/runtime/Runtime.h>

#include <lexec/any_sender_of.hpp>
#include <lexec/execution.hpp>
#include <tc/device.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace lr::render {

struct ImageTargetDesc {
    core::ivec2 extent{256, 256};
};

struct WindowTargetDesc {
    core::ivec2 extent{256, 256};
    std::shared_ptr<Presenter> presenter;
};

// What views render into. Thread-safe; a change applies to frames submitted after it.
struct RenderTargets {
    virtual ~RenderTargets() = default;

    // Each frame's pixels go to the presenter; the view's ViewResult::image stays empty.
    virtual RenderTargetId createWindow(WindowTargetDesc const &desc) = 0;
    // Each frame's pixels come back in the view's ViewResult::image.
    virtual RenderTargetId createImage(ImageTargetDesc const &desc) = 0;
    // The views of the target start their history (the path tracer's accumulation) over.
    virtual void resize(RenderTargetId id, core::ivec2 extent) = 0;
    virtual void destroy(RenderTargetId id) = 0;
};

struct PipelineDesc {
    PipelineId base; // the pipeline whose features come first
    std::vector<std::shared_ptr<RenderFeature>> features;
};

// Thread-safe.
struct PipelineRegistry {
    virtual ~PipelineRegistry() = default;

    // The real-time pipeline: visibility buffer, shading, tone mapping.
    virtual PipelineId visibility() const noexcept = 0;
    // The reference: a path tracer that adds a sample per pixel each frame of a view and
    // shows their mean, tone mapped (ViewStats::accumulatedSamples).
    virtual PipelineId pathTraceReference() const noexcept = 0;
    virtual PipelineId create(PipelineDesc desc) = 0;
};

struct FrameDesc {
    core::RenderWorldDelta delta;
    std::vector<ViewDesc> views;
};

struct FrameResult {
    std::uint64_t frameIndex = 0;
    std::vector<ViewResult> views;
    // From submit() until the frame was finished, after the device.
    std::chrono::duration<double, std::milli> wallTime{0.0};
};

using FrameSender = lexec::any_sender_of<lexec::set_value_t(FrameResult), lexec::set_error_t(std::exception_ptr),
                                         lexec::set_stopped_t()>;

struct RendererConfig {
    std::string backend = "cpu";    // one of tc::availableBackends()
    std::string deviceNameContains; // optional, e.g. "llvmpipe"
    runtime::Runtime *runtime = nullptr;
    // Times every pass on the device (ViewResult::passTimings) where DeviceInfo::timers
    // allows. Timed dispatches run one at a time, so leave it off but for profiling.
    bool gpuTimers = false;
    // Frames recorded for the device and not finished yet, at most; later frames wait,
    // prepared, until an earlier one finishes.
    std::uint32_t framesInFlight = 2;
};

struct Renderer {
    virtual ~Renderer() = default;

    virtual tc::DeviceInfo const &deviceInfo() const noexcept = 0;
    virtual RenderTargets &targets() noexcept = 0;
    virtual PipelineRegistry &pipelines() noexcept = 0;

    // Returns at once; any thread may call it. Frames are numbered and processed in the
    // order of the calls, whether or not their senders are started, and the senders
    // complete in that order once the device has finished each frame, on the GPU thread
    // (move off it before waiting for anything). A tc::DeviceError fails the frame it
    // happens in, those in flight and those prepared before the device was recreated;
    // the next frame uploads the whole scene again.
    virtual FrameSender submit(FrameDesc frame) = 0;
};

// Throws tc::DeviceError when the device cannot be created.
std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config);

} // namespace lr::render
