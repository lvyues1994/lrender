#include <lr/examples/Scripts.h>

#include <lr/examples/GltfScene.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <vector>

namespace lr::examples {

namespace {

using scene::nextFrame;
using scene::ScriptTask;

constexpr auto kUp = core::vec3{0.0f, 1.0f, 0.0f};

float secondsOf(scene::SimDuration const duration) { return std::chrono::duration<float>{duration}.count(); }

core::Transform looking(core::vec3 const &eye, core::vec3 const &target) {
    auto transform = core::Transform{};
    transform.translation = eye;
    transform.rotation = core::Rotation::lookingAlong(target - eye, kUp);
    return transform;
}

core::Transform bobbed(core::Transform rest, float const seconds) {
    rest.translation.y += 0.15f * (1.0f - std::cos(seconds * 2.0f));
    rest.rotation = core::Rotation::axisAngle(kUp, seconds * 0.8f);
    return rest;
}

// A circle above the box, all of it in view from there.
core::Transform orbitAround(core::Aabb const &bounds, float const angle) {
    auto const framing = framingCamera(bounds);
    auto const offset = framing.position - framing.target;
    auto const flat = tc::length(core::vec3{offset.x, 0.0f, offset.z});
    auto const eye = framing.target + core::vec3{flat * std::sin(angle), offset.y, flat * std::cos(angle)};
    return looking(eye, framing.target);
}

// Puts the loaded scene in the world in place of the one shown before, framing the first.
std::optional<scene::EntityId> show(scene::World &world, asset::AssetPath const &path,
                                    asset::AssetResult<asset::SceneHandle> const &loaded,
                                    std::optional<scene::EntityId> const shown, Orbit &orbit, scene::EntityId const camera) {
    if (not loaded) {
        std::fprintf(stderr, "loading %s: %s\n", path.c_str(), loaded.error().message.c_str());
        return shown;
    }
    auto const &scene = **loaded;
    for (auto const &warning : scene.warnings) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), warning.c_str());
    }
    if (shown) {
        world.destroyEntity(*shown);
        std::printf("reloaded %s\n", path.c_str());
    } else {
        auto const view = cameraFor(scene);
        auto const spin = orbit.spin;
        orbit = orbitOf(view);
        orbit.spin = spin;
        world.setCamera(camera, lensOf(view));
        world.setEnvironment(skyForScenes());
        std::printf("%s: %zu meshes, %zu instances, %zu materials, %zu textures, %zu lights\n", path.c_str(),
                    scene.meshes.size(), scene.instances.size(), scene.materials.size(), scene.textures.size(),
                    scene.lights.size());
    }
    return spawnScene(scene, world);
}

} // namespace

Orbit orbitOf(core::CameraDesc const &camera) {
    auto const offset = camera.position - camera.target;
    auto const distance = tc::length(offset);
    auto orbit = Orbit{};
    orbit.target = camera.target;
    orbit.yaw = std::atan2(offset.x, offset.z);
    orbit.pitch = std::asin(offset.y / distance);
    orbit.distance = distance;
    return orbit;
}

core::Transform cameraTransform(Orbit const &orbit) {
    auto const flat = std::cos(orbit.pitch);
    auto const direction = core::vec3{flat * std::sin(orbit.yaw), std::sin(orbit.pitch), flat * std::cos(orbit.yaw)};
    return looking(orbit.target + direction * orbit.distance, orbit.target);
}

scene::CameraLens lensOf(core::CameraDesc const &camera) {
    auto lens = scene::CameraLens{};
    lens.verticalFov = camera.verticalFov;
    lens.nearPlane = camera.nearPlane;
    lens.farPlane = camera.farPlane;
    return lens;
}

auto steerCamera(scene::World &world, Orbit &orbit, scene::EntityId const camera)
    CO2_BEG(ScriptTask, (world, orbit, camera), scene::SimDuration elapsed{};) {
    for (;;) {
        world.setTransform(camera, cameraTransform(orbit));
        CO2_AWAIT_SET(elapsed, nextFrame());
        orbit.yaw += orbit.spin * secondsOf(elapsed);
    }
}
CO2_END

auto bob(scene::World &world, scene::EntityId const entity)
    CO2_BEG(ScriptTask, (world, entity), core::Transform rest; float seconds{}; scene::SimDuration elapsed{};) {
    rest = world.transformOf(entity);
    while (world.isAlive(entity)) {
        world.setTransform(entity, bobbed(rest, seconds));
        CO2_AWAIT_SET(elapsed, nextFrame());
        seconds += secondsOf(elapsed);
    }
}
CO2_END

auto showGltf(scene::World &world, asset::AssetLibrary &assets, asset::AssetPath const path, Orbit &orbit,
              scene::EntityId const camera)
    CO2_BEG(ScriptTask, (world, assets, path, orbit, camera), asset::AssetResult<asset::SceneHandle> loaded;
            std::optional<scene::EntityId> shown; std::vector<asset::AssetChange> changes;) {
    for (;;) {
        CO2_AWAIT_SET(loaded, assets.loadScene(path));
        shown = show(world, path, loaded, shown, orbit, camera);
        changes.clear();
        while (changes.empty()) {
            CO2_AWAIT(nextFrame());
            assets.drainChanges(changes);
        }
    }
}
CO2_END

auto orbitCamera(scene::World &world, asset::AssetLibrary &assets, scene::EntityId const camera,
                 asset::AssetPath const path)
    CO2_BEG(ScriptTask, (world, assets, camera, path), asset::AssetResult<core::MeshHandle> mesh; core::Aabb bounds;
            float angle{};) {
    CO2_AWAIT_SET(mesh, assets.loadMesh(path));
    if (mesh) {
        world.spawnInstance(*mesh, core::Transform::identity());
        bounds = core::boundsOf(**mesh);
        world.setCamera(camera, lensOf(framingCamera(bounds)));
    } else {
        std::fprintf(stderr, "loading %s: %s\n", path.c_str(), mesh.error().message.c_str());
    }
    for (angle = 0.0f;; angle += 0.01f) {
        world.setTransform(camera, orbitAround(bounds, angle));
        CO2_AWAIT(nextFrame());
    }
}
CO2_END

} // namespace lr::examples
