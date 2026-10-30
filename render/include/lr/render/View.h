#pragma once

#include <lr/core/Id.h>
#include <lr/core/Math.h>
#include <lr/core/Scene.h>

#include <cstdint>
#include <optional>
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

struct ViewStats {
    std::uint32_t visibleInstances = 0;
    std::uint32_t visibleTriangles = 0;
};

struct ViewResult {
    ViewId view;
    RenderTargetId target;
    std::optional<ImageRgba8> image; // image targets
    ViewStats stats;
};

} // namespace lr::render
