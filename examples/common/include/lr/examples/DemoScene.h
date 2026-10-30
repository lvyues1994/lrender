#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>
#include <lr/render/View.h>

#include <string>

namespace lr::examples {

struct DemoScene {
    core::RenderWorldDelta delta; // creates the whole scene
    core::CameraDesc camera;
};

// A ground plane with a box and spheres of different materials, a sun, a point light and
// a sky.
DemoScene makeDemoScene();

// Binary PPM (P6); returns false when the file cannot be written.
bool writePpm(render::ImageRgba8 const &image, std::string const &path);

} // namespace lr::examples
