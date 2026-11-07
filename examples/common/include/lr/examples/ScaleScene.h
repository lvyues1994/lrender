#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>

#include <cstdint>

namespace lr::examples {

struct ScaleSceneDesc {
    std::uint32_t instances = 100000; // on a square grid, two metres apart
    std::uint32_t lights = 256;       // point lights with a range, spread over the grid
    // Segments around each sphere: their triangles grow with its square.
    std::uint32_t sphereSegments = 16;
    std::uint32_t seed = 1;
};

struct ScaleScene {
    core::RenderWorldDelta delta; // creates the whole scene
    core::CameraDesc camera;      // low, from a corner of the grid across it
    std::uint64_t triangles = 0;  // of the instances on the grid
};

// For measuring how the renderer scales: a ground plane under a grid of boxes, pillars and
// spheres of four meshes and sixteen untextured materials, each turned and scaled, the
// point lights, a sun that casts shadows and a sky. The same description gives the same
// scene.
ScaleScene makeScaleScene(ScaleSceneDesc const &desc);

} // namespace lr::examples
