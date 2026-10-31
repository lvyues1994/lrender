#pragma once

#include <lr/render/FrameGraph.h>

#include <cstdint>
#include <unordered_map>

namespace lr::render {

// Temporal anti-aliasing cycles through this many sub-pixel offsets. With the history's
// blend (temporalFrames()), 16 put a straight edge's pixels some 0.035 from their
// coverage on average, 8 some 0.06.
inline constexpr std::uint32_t kTemporalJitters = 16;

// The offset of a view's `sample`th frame since its history started over, in pixels right
// and down: Halton (2, 3) point 1 + sample % kTemporalJitters, less a half.
core::vec2 temporalJitter(std::uint32_t sample);

// The view-projection that puts points `jitter` pixels right and down of where
// `viewProjection` puts them on a target of `extent`.
core::mat4 jitteredViewProjection(core::mat4 const &viewProjection, core::vec2 jitter, core::ivec2 extent);

struct TemporalRequest {
    ViewId view;
    std::uint64_t frame = 0;
    core::ivec2 extent{0};
    core::mat4 viewProjection{1.0f}; // without jitter
    bool cameraCut = false;
};

// What the render thread keeps of each view whose frames jitter.
struct TemporalViews {
    // Frames a view may go without jittering before it is forgotten.
    explicit TemporalViews(std::uint64_t const keepFrames_) noexcept : keepFrames(keepFrames_) {}

    // The view's jitter and history state for the requested frame, which it then remembers.
    TemporalView advance(TemporalRequest const &request);
    // Every view starts over, as after the device was lost.
    void restart() noexcept { records.clear(); }
    void expire(std::uint64_t frame);

private:
    struct Record {
        std::uint64_t frame;
        core::ivec2 extent;
        core::mat4 viewProjection;
        std::uint32_t sample;
    };

    std::uint64_t keepFrames;
    std::unordered_map<ViewId, Record> records;
};

} // namespace lr::render
