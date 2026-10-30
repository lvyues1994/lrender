#pragma once

#include <lr/core/Id.h>
#include <lr/core/Math.h>
#include <lr/core/Scene.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lr::render {

using core::PipelineId;
using core::RenderTargetId;
using core::ViewId;

enum class ToneMapping : std::uint8_t {
    Clamp, // exposure, then clamp to [0, 1]
    Aces,  // exposure, then the ACES filmic curve
};

struct ViewSettings {
    float exposure = 1.0f;
    ToneMapping toneMapping = ToneMapping::Aces;
};

// A view keeps history (the path tracer's accumulation) by its id across frames, until
// it goes unrendered for a while or its target is destroyed.
struct ViewDesc {
    ViewId id;
    RenderTargetId target;
    PipelineId pipeline; // an invalid id selects the visibility pipeline
    core::CameraDesc camera;
    ViewSettings settings;
};

// sRGB-encoded RGBA8, one std::uint32_t per pixel with red in the low byte, rows from the top.
struct ImageRgba8 {
    core::ivec2 extent{0};
    std::vector<std::uint32_t> pixels;
};

// Pixels laid out as in ImageRgba8, in memory someone else owns.
struct ImageRgba8View {
    core::ivec2 extent{0};
    std::uint32_t const *pixels = nullptr; // extent.x * extent.y of them
};

inline ImageRgba8View imageView(ImageRgba8 const &image) noexcept {
    return ImageRgba8View{image.extent, image.pixels.data()};
}

struct ViewStats {
    std::uint32_t visibleInstances = 0;
    std::uint32_t visibleTriangles = 0;
    // The samples per pixel the path tracer has averaged since its last reset; 0 for
    // pipelines that do not accumulate.
    std::uint32_t accumulatedSamples = 0;
};

// Device time of one pass of a view's frame graph.
struct PassTiming {
    std::string pass;
    std::uint32_t dispatches = 0;
    // The sum over the dispatches; nullopt when the device reported a clock change
    // during one of them.
    std::optional<double> milliseconds;
};

struct ViewResult {
    ViewId view;
    RenderTargetId target;
    std::optional<ImageRgba8> image; // image targets
    ViewStats stats;
    // Every pass that ran, in order, when RendererConfig::gpuTimers is on and the device
    // has timers; empty otherwise.
    std::vector<PassTiming> passTimings;
};

} // namespace lr::render
