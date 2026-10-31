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

// The visibility pipeline's cascaded shadow map for the shadowed directional light.
struct ShadowSettings {
    std::uint32_t cascades = 3;      // 1 to 4, splitting the distance from the near plane
    std::uint32_t resolution = 1024; // texels on a side of each cascade
    float distance = 40.0f;          // from the camera; nothing farther receives shadows
};

enum class AntiAliasing : std::uint8_t {
    None,
    // Temporal: the visibility pipeline moves its projection by a different sub-pixel
    // offset every frame and blends each frame into the view's history, reprojected along
    // the motion of the surfaces. The path tracer ignores it; it averages anyway.
    Taa,
};

struct ViewSettings {
    float exposure = 1.0f;
    ToneMapping toneMapping = ToneMapping::Aces;
    ShadowSettings shadows;
    AntiAliasing antiAliasing = AntiAliasing::None;
};

// A view keeps history (the path tracer's accumulation, temporal anti-aliasing) by its id
// across frames, until it goes unrendered for a while or its target is destroyed.
struct ViewDesc {
    ViewId id;
    RenderTargetId target;
    PipelineId pipeline; // an invalid id selects the visibility pipeline
    core::CameraDesc camera;
    ViewSettings settings;
    // The camera jumped since the view's previous frame: temporal anti-aliasing starts over
    // instead of reprojecting what the view showed before.
    bool cameraCut = false;
    // ViewResult::graph describes this frame's graph.
    bool describeGraph = false;
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
    // Device memory the frame graph's resources of the frame took, history aside: the
    // heap those with disjoint lifetimes share and the others. `unaliasedBytes` is what
    // they would take each in its own memory.
    std::uint64_t graphBytes = 0;
    std::uint64_t unaliasedBytes = 0;
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
    // With ViewDesc::describeGraph: the frame's graph in Graphviz's DOT language, the
    // passes that ran in order and the resources they read and write, with their
    // lifetimes, sizes and places in the shared heap, and the passes' device times.
    std::optional<std::string> graph;
};

} // namespace lr::render
