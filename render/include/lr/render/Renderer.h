#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/render/FrameGraph.h>
#include <lr/render/View.h>
#include <lr/runtime/Runtime.h>

#include <lexec/any_sender_of.hpp>
#include <lexec/execution.hpp>
#include <tc/device.hpp>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace lr::render {

struct ImageTargetDesc {
    core::ivec2 extent{256, 256};
};

// What views render into. Thread-safe; a change applies to frames submitted after it.
struct RenderTargets {
    virtual ~RenderTargets() = default;

    // Each frame's pixels come back in the view's ViewResult::image.
    virtual RenderTargetId createImage(ImageTargetDesc const &desc) = 0;
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

    virtual PipelineId visibility() const noexcept = 0;
    virtual PipelineId create(PipelineDesc desc) = 0;
};

struct FrameDesc {
    core::RenderWorldDelta delta;
    std::vector<ViewDesc> views;
};

struct FrameResult {
    std::uint64_t frameIndex = 0;
    std::vector<ViewResult> views;
};

using FrameSender = lexec::any_sender_of<lexec::set_value_t(FrameResult), lexec::set_error_t(std::exception_ptr),
                                         lexec::set_stopped_t()>;

struct RendererConfig {
    std::string backend = "cpu";    // one of tc::availableBackends()
    std::string deviceNameContains; // optional, e.g. "llvmpipe"
    runtime::Runtime *runtime = nullptr;
};

struct Renderer {
    virtual ~Renderer() = default;

    virtual tc::DeviceInfo const &deviceInfo() const noexcept = 0;
    virtual RenderTargets &targets() noexcept = 0;
    virtual PipelineRegistry &pipelines() noexcept = 0;

    // Returns at once; any thread may call it. Frames are numbered and processed in the
    // order of the calls. The sender completes once the device has finished the frame.
    virtual FrameSender submit(FrameDesc frame) = 0;
};

// Throws tc::DeviceError when the device cannot be created.
std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config);

} // namespace lr::render
