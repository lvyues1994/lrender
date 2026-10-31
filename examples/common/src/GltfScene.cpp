#include <lr/examples/GltfScene.h>

#include <lexec/execution.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace lr::examples {

namespace {

constexpr auto kFramingFov = 0.8f;
constexpr auto kSky = core::vec3{0.25f, 0.3f, 0.4f};

} // namespace

ViewableScene viewableScene(asset::SceneData const &scene, asset::IdAllocator &ids) {
    auto added = asset::sceneDelta(scene, ids);
    auto result = ViewableScene{std::move(added.delta), std::move(added.removal), framingCamera(scene.bounds)};
    result.delta.environment = core::EnvironmentDesc{kSky};
    if (scene.lights.empty()) {
        auto sun = core::DirectionalLight{};
        sun.direction = core::vec3{-0.4f, -1.0f, -0.6f};
        sun.illuminance = 2.5f;
        sun.castsShadows = true;
        auto const id = ids.newLight();
        result.delta.lights.push_back(core::LightUpsert{id, sun});
        result.removal.lights.push_back(id);
    }
    if (not scene.cameras.empty()) {
        result.camera = scene.cameras.front().camera;
    }
    return result;
}

core::CameraDesc framingCamera(core::Aabb const &bounds) {
    auto const centre = (bounds.min + bounds.max) * 0.5f;
    auto const radius = std::max(tc::length(bounds.max - bounds.min) * 0.5f, 1e-3f);
    auto const distance = radius / std::sin(kFramingFov * 0.5f) * 1.1f;
    auto camera = core::CameraDesc{};
    camera.target = centre;
    camera.position = centre + tc::normalize(core::vec3{0.35f, 0.45f, 1.0f}) * distance;
    camera.verticalFov = kFramingFov;
    camera.nearPlane = std::max(distance - radius * 2.0f, distance * 1e-3f);
    camera.farPlane = distance + radius * 4.0f;
    return camera;
}

asset::SceneHandle loadSceneNow(asset::AssetLibrary &assets, std::string const &path) {
    auto loaded = lexec::sync_wait(assets.loadScene(path));
    if (not loaded) {
        throw std::runtime_error{"loading " + path + " was stopped"};
    }
    auto scene = std::get<0>(std::move(*loaded));
    if (not scene) {
        throw std::runtime_error{scene.error().message};
    }
    return *scene;
}

} // namespace lr::examples
