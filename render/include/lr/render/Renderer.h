#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/render/Capabilities.h>
#include <lr/render/FrameGraph.h>
#include <lr/render/MaterialModel.h>
#include <lr/render/Presenter.h>
#include <lr/render/View.h>
#include <lr/runtime/Runtime.h>

#include <lexec/any_sender_of.hpp>
#include <lexec/execution.hpp>
#include <tc/device.hpp>
#include <tc/program.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
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

// When the textures the renderer decodes or resamples (compressed images in formats the
// device has no arrays of, images that are not power-of-two squares) upload. Their
// conversions run on the background pool; images that go to a texture pool as they are
// upload with the frame they come in.
enum class ConversionDelivery : std::uint8_t {
    // With the first frame after their conversion finished: no frame waits for one.
    AsSoonAsReady,
    // With the frame after the one they came in, which waits for the conversion if it has
    // not finished: how long conversions take does not change which frame shows them.
    Deterministic,
};

struct PipelineDesc {
    PipelineId base; // the pipeline whose features come first
    std::vector<std::shared_ptr<RenderFeature>> features;
};

// Thread-safe.
struct PipelineRegistry {
    virtual ~PipelineRegistry() = default;

    // The real-time pipeline: visibility buffer, shadow map, shading, transparency, tone
    // mapping.
    virtual PipelineId visibility() const noexcept = 0;
    // The reference: a path tracer that adds a sample per pixel each frame of a view and
    // shows their mean, tone mapped (ViewStats::accumulatedSamples).
    virtual PipelineId pathTraceReference() const noexcept = 0;
    // The same path tracer, denoised while the view or the scene changes (see
    // makePathTraceDenoiseFeature()): once the mean holds enough samples, it shows the mean
    // as the reference does.
    virtual PipelineId pathTraceInteractive() const noexcept = 0;
    virtual PipelineId create(PipelineDesc desc) = 0;
};

struct FrameDesc {
    core::RenderWorldDelta delta;
    std::vector<ViewDesc> views;
};

// Device memory the scene takes, in bytes: its tables, as large as they have grown, and
// the texture pools with their mip levels.
struct SceneMemory {
    std::uint64_t geometry = 0;    // vertices and indices
    std::uint64_t clusters = 0;
    std::uint64_t objects = 0;     // meshes, instances, materials and lights
    std::uint64_t hierarchies = 0; // the hierarchies of path tracing without ray queries
    std::uint64_t textures = 0;
};

struct FrameResult {
    std::uint64_t frameIndex = 0;
    std::vector<ViewResult> views;
    // From submit() until the frame was finished, after the device.
    std::chrono::duration<double, std::milli> wallTime{0.0};
    // Host time within wallTime: on the render thread, applying the delta and building the
    // views' graphs; on the GPU thread, recording the frame for the device and submitting
    // it, and finishing it once the device is done (readbacks, timings, presenting).
    std::chrono::duration<double, std::milli> prepareTime{0.0};
    std::chrono::duration<double, std::milli> recordTime{0.0};
    std::chrono::duration<double, std::milli> finishTime{0.0};
    // As the GPU thread held the scene when it finished the frame.
    SceneMemory sceneMemory;
    // Textures the device held when it was lost: upsert them again. Until then materials
    // use their factors alone. Frames that fail leave the list to the next one that does not.
    std::vector<core::TextureId> lostTextures;
};

using FrameSender = lexec::any_sender_of<lexec::set_value_t(FrameResult), lexec::set_error_t(std::exception_ptr),
                                         lexec::set_stopped_t()>;

// tc's report: the programs replaced, and those rejected with the reason.
using KernelReloadSender = lexec::any_sender_of<lexec::set_value_t(tc::ReloadReport),
                                                lexec::set_error_t(std::exception_ptr), lexec::set_stopped_t()>;

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
    // Texture data copied to the device per frame, at most; new textures stream in over
    // as many frames as that takes, and a single mip level larger than this goes alone.
    std::size_t textureStreamingBytes = std::size_t{8} << 20;
    // Larger textures are scaled down on the host (also to what the device allows).
    std::uint32_t maxTextureSize = 2048;
    ConversionDelivery textureConversions = ConversionDelivery::AsSoonAsReady;
    // A view's frame graph places its device buffers and images (images where the device
    // places them, DeviceInfo::placedImages) in one heap, those whose passes do not overlap
    // in the same memory; otherwise each gets memory of its own (ViewStats::graphBytes).
    bool aliasGraphResources = true;
    // Raster passes draw only the clusters of the visible instances that lie in view and,
    // for single-sided materials, do not face away; the pixels are the same either way.
    bool cullClusters = true;
    // With cullClusters, the visibility pipeline draws the clusters visible the frame before
    // first, builds a depth pyramid from them and draws only the others that may show in
    // front of what is there; the pixels are the same either way.
    bool cullOccluded = true;
    // Shading lights each point with the lights of its cell of a grid over the view (see
    // lightGridHeaderWords()) instead of every light; the pixels are the same either way.
    bool clusterLights = true;
    // Optional kernel variants (RendererCapabilities), each used where the device runs it
    // unless turned off here.
    VariantUse subgroups = VariantUse::Auto;
    VariantUse singlePassVisibility = VariantUse::Auto;
    VariantUse rayQueries = VariantUse::Auto;
};

struct Renderer {
    virtual ~Renderer() = default;

    virtual tc::DeviceInfo const &deviceInfo() const noexcept = 0;
    // The kernel variants chosen for the device; the same after the device is recreated.
    virtual RendererCapabilities const &capabilities() const noexcept = 0;
    virtual RenderTargets &targets() noexcept = 0;
    virtual PipelineRegistry &pipelines() noexcept = 0;
    virtual MaterialModels &materialModels() noexcept = 0;

    // Returns at once; any thread may call it. Frames are numbered and processed in the
    // order of the calls, whether or not their senders are started, and the senders
    // complete in that order once the device has finished each frame, on the GPU thread
    // (move off it before waiting for anything). A tc::DeviceError fails the frame it
    // happens in, those in flight and those prepared before the device was recreated;
    // the next frame uploads the whole scene again.
    virtual FrameSender submit(FrameDesc frame) = 0;

    // Development: on the GPU thread, between the frames submitted before and after the
    // call, replaces the programs of a tc_add_kernels glsl directory (such as
    // builtInKernelDirectory()) with what the build last wrote there, and completes with
    // the report there. GPU devices use them from their next frame; programs whose
    // interface changed are rejected, and the CPU backend keeps running compiled C++.
    // Release builds (LRENDER_KERNEL_RELOAD off) reject everything.
    virtual KernelReloadSender reloadKernels(std::string directory) = 0;
};

// Throws tc::DeviceError when the device cannot be created, or cannot run one of the
// kernels every frame may need (the message names them).
std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config);

// Where the build writes lrender's own kernels for reloadKernels(); nullopt in builds
// without LRENDER_KERNEL_RELOAD.
std::optional<std::string> builtInKernelDirectory();

} // namespace lr::render
