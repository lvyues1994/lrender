#pragma once

#include <lr/render/FrameGraph.h>

#include <memory>

namespace lr::render {

// What the visibility feature puts on the blackboard.
struct VisibilityOutputs {
    // [0] = (visible instances, visible triangles), then (instance, first visible triangle)
    // per visible instance in instance order.
    BufferRef<tc::uvec2> visible;
    // Per pixel: the lowest visible triangle at the nearest depth, or noTriangle().
    BufferRef<tc::uint> triangles;
    // Per pixel: floatBitsToUint of the nearest clip depth, 0xFFFFFFFF where empty.
    BufferRef<tc::uint> depth;
};

// The features of the visibility pipeline, in order.
std::shared_ptr<RenderFeature> makeVisibilityFeature();
std::shared_ptr<RenderFeature> makeShadingFeature();
std::shared_ptr<RenderFeature> makeTonemapFeature();

} // namespace lr::render
