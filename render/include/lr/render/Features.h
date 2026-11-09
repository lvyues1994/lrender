#pragma once

#include <lr/render/FrameGraph.h>

#include <memory>

namespace lr::render {

// What the visibility feature puts on the blackboard.
struct VisibilityOutputs {
    // [0] = (visible instances, visible triangles, 0, 0), then (instance, first visible
    // triangle, its mesh's first index, its mesh's first vertex) per visible instance in
    // instance order. Instances whose material blends are left out.
    BufferRef<tc::uvec4> visible;
    // Per pixel: the lowest visible triangle at the nearest depth, or noTriangle().
    BufferRef<tc::uint> triangles;
    // Per pixel: floatBitsToUint of the nearest clip depth, 0xFFFFFFFF where empty.
    BufferRef<tc::uint> depth;
};

// What the shadow feature puts on the blackboard: the shadow map of the scene's shadowed
// directional light (see shadowHeaderWords()), with no cascades when no light casts
// shadows.
struct ShadowOutputs {
    BufferRef<tc::uint> map;
    std::uint32_t light = 0xFFFFFFFFu; // its index in the light table, or none
};

// What the first feature that shades puts on the blackboard: the view's light grid (see
// lightGridHeaderWords()), which every kernel that expands LR_SURFACE_SHADING reads.
struct LightGridOutputs {
    BufferRef<tc::uint> grid;
};

// What the first feature that needs it puts on the blackboard (the temporal anti-aliasing
// feature, the path tracer's denoiser).
struct TemporalOutputs {
    // Per pixel, in pixels right and down: where the surface it shows is on the screen
    // less where it was in the previous frame, both without the frames' jitters.
    BufferRef<tc::vec2> motion;
};

// The features of the visibility pipeline, in order. The shadow feature draws cascades
// fitted to the view frustum up to ViewSettings::shadows.distance, each snapped to its
// texel grid so that shadows stay put while the camera moves. The shading feature sorts
// the pixels by the material model of what they show and runs each model's kernel over
// its own. The transparency feature collects the fragments of blended instances in
// per-pixel lists, keeps the nearest eight of each pixel and blends them in depth order
// over the opaque colour, shading them with the metallic-roughness model whatever model
// their materials name. The temporal anti-aliasing feature (a PostProcess) runs for views
// with ViewSettings::antiAliasing Taa and blends each frame into the view's history.
std::shared_ptr<RenderFeature> makeVisibilityFeature();
std::shared_ptr<RenderFeature> makeShadowFeature();
std::shared_ptr<RenderFeature> makeShadingFeature();
std::shared_ptr<RenderFeature> makeTransparencyFeature();
std::shared_ptr<RenderFeature> makeTemporalAntiAliasingFeature();
std::shared_ptr<RenderFeature> makeTonemapFeature();

// What the path tracing feature puts on the blackboard: the view's mean, and the samples it
// holds, set on the GPU thread as the feature's pass is recorded, before the passes after it.
struct PathTraceOutputs {
    ImageRef<AccumulationFormat> mean;
    std::shared_ptr<std::uint32_t const> samples;
};

// The path tracing pipeline's shading: a Shading-stage feature that adds one sample per
// pixel to the view's running mean and writes the mean to the scene colour. The mean
// starts over when the camera, the extent or the scene changes, or textures finish
// streaming in.
std::shared_ptr<RenderFeature> makePathTraceFeature();

// A PostProcess feature that denoises the path tracer's mean, for pipelines that also have
// the visibility feature: spatiotemporal variance-guided filtering of the illumination,
// with guides from the visibility buffer (blended surfaces are not in it, so what shows
// through them is filtered by what is behind). While the mean holds fewer than 16
// samples the view shows the denoised colour; from 16 to 64 it fades into the mean, so a
// view that stays still ends with the reference's colour.
std::shared_ptr<RenderFeature> makePathTraceDenoiseFeature();

} // namespace lr::render
