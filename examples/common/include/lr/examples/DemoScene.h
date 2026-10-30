#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>

namespace lr::examples {

struct DemoScene {
    core::RenderWorldDelta delta; // creates the whole scene
    core::CameraDesc camera;
};

// A ground plane with a box and spheres of different materials, a sun, a point light and
// a sky.
DemoScene makeDemoScene();

} // namespace lr::examples
