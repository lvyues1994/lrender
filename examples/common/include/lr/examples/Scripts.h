#pragma once

#include <lr/asset/AssetLibrary.h>
#include <lr/core/Scene.h>
#include <lr/scene/ScriptTask.h>
#include <lr/scene/World.h>

namespace lr::examples {

// A camera on a sphere around its target, which the viewer's mouse and steerCamera() turn.
struct Orbit {
    core::vec3 target{0.0f};
    float yaw = 0.0f;   // about +y, from +z
    float pitch = 0.0f; // above the target's horizon
    float distance = 5.0f;
    float spin = 0.15f; // yaw per second of simulation time
};

// The orbit of a camera at `camera`'s position around its target.
Orbit orbitOf(core::CameraDesc const &camera);
// A camera's transform on the orbit, looking at the target.
core::Transform cameraTransform(Orbit const &orbit);
scene::CameraLens lensOf(core::CameraDesc const &camera);

// Every frame, turns the orbit by its spin and puts the camera on it.
scene::ScriptTask steerCamera(scene::World &world, Orbit &orbit, scene::EntityId camera);
// Bobs the entity up and down and turns it, as long as it lives.
scene::ScriptTask bob(scene::World &world, scene::EntityId entity);
// Shows the glTF file's scene, framed by the orbit and the camera's lens the first time,
// and loads it again whenever its files change.
scene::ScriptTask showGltf(scene::World &world, asset::AssetLibrary &assets, asset::AssetPath path, Orbit &orbit,
                           scene::EntityId camera);
// The script of docs/architecture.md: loads a mesh, puts it into the world and circles
// the camera around it.
scene::ScriptTask orbitCamera(scene::World &world, asset::AssetLibrary &assets, scene::EntityId camera,
                              asset::AssetPath path);

} // namespace lr::examples
