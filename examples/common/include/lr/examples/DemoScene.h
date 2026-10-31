#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>

namespace lr::examples {

struct DemoScene {
    core::RenderWorldDelta delta; // creates the whole scene
    core::CameraDesc camera;
};

// A checkered ground plane with a textured, normal-mapped box, a gold, a glowing and a
// glass-like (blended) sphere and a lattice cut out by alpha, a sun that casts shadows,
// a point light and a sky. The textures are made here.
DemoScene makeDemoScene();

} // namespace lr::examples
