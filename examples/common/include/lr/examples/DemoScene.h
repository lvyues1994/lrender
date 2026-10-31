#pragma once

#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>
#include <lr/scene/World.h>

namespace lr::examples {

struct DemoEntities {
    scene::EntityId camera;
    core::CameraDesc view;   // where the camera starts and what it looks at
    scene::EntityId lantern; // the glowing sphere with the lamp above it
};

// A checkered ground plane with a textured, normal-mapped box, a gold, a glowing and a
// glass-like (blended) sphere and a lattice cut out by alpha, a sun that casts shadows,
// a point light and a sky. The textures are made here.
DemoEntities addDemoScene(scene::World &world);

struct DemoScene {
    core::RenderWorldDelta delta; // creates the whole scene
    core::CameraDesc camera;
};

// The demo scene of a world of its own, as one delta.
DemoScene makeDemoScene();

} // namespace lr::examples
