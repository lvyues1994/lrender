#pragma once

#include "FrameGraphImpl.h"
#include "SceneUpdate.h"

#include <lr/render/Renderer.h>

#include <stackfull/sync/Completion.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <optional>
#include <vector>

namespace lr::render {

struct PreparedView {
    ViewId view;
    RenderTargetId target;
    std::optional<CompiledGraph> graph;
    std::exception_ptr failure; // the graph could not be built
};

// A frame as the render thread hands it to the GPU thread.
struct PreparedFrame {
    std::uint64_t index = 0;
    std::chrono::steady_clock::time_point submitted;
    std::chrono::duration<double, std::milli> prepareTime{0.0};
    // The device generation the scene update was made for: one the GPU thread has
    // replaced cannot apply it.
    std::uint64_t deviceGeneration = 0;
    // The frame failed before anything was made of it.
    std::exception_ptr failure;
    SceneUpdate scene;
    // A view traces rays with ray queries: the GPU thread brings the acceleration
    // structures up to date before recording it.
    bool tracesWithRayQueries = false;
    std::vector<PreparedView> views;
    std::vector<std::uint64_t> releasedHistories;
    // Textures a lost device took with it, for FrameResult::lostTextures.
    std::vector<core::TextureId> lostTextures;
    // What the frame's FrameSender waits for.
    stackfull::sync::Completion<FrameResult> completion;
};

} // namespace lr::render
