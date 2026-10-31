#include <lr/examples/GltfScene.h>

#include <lexec/execution.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace lr::examples {

namespace {

constexpr auto kFramingFov = 0.8f;
constexpr auto kSky = core::vec3{0.25f, 0.3f, 0.4f};

core::DirectionalLight defaultSun() {
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.6f};
    sun.illuminance = 2.5f;
    sun.castsShadows = true;
    return sun;
}

std::optional<core::TextureId> remapped(std::optional<core::TextureId> const local,
                                        std::vector<core::TextureId> const &textures) {
    if (not local or local->value >= textures.size()) {
        return std::nullopt;
    }
    return textures[local->value];
}

// The scene's materials, which hold its textures, under the world's ids.
std::vector<core::MaterialId> addMaterials(asset::SceneData const &scene, scene::WorldResources &resources) {
    auto textures = std::vector<core::TextureId>{};
    for (auto const &texture : scene.textures) {
        textures.push_back(resources.addTexture(texture.image, texture.sampler));
    }
    auto materials = std::vector<core::MaterialId>{};
    for (auto material : scene.materials) {
        material.baseColorTexture = remapped(material.baseColorTexture, textures);
        material.metallicRoughnessTexture = remapped(material.metallicRoughnessTexture, textures);
        material.emissiveTexture = remapped(material.emissiveTexture, textures);
        material.normalTexture = remapped(material.normalTexture, textures);
        materials.push_back(resources.addMaterial(material));
    }
    for (auto const texture : textures) {
        resources.releaseTexture(texture);
    }
    return materials;
}

} // namespace

ViewableScene viewableScene(asset::SceneData const &scene, asset::IdAllocator &ids) {
    auto added = asset::sceneDelta(scene, ids);
    auto result = ViewableScene{std::move(added.delta), std::move(added.removal), cameraFor(scene)};
    result.delta.environment = skyForScenes();
    if (scene.lights.empty()) {
        auto const id = ids.newLight();
        result.delta.lights.push_back(core::LightUpsert{id, defaultSun()});
        result.removal.lights.push_back(id);
    }
    return result;
}

scene::EntityId spawnScene(asset::SceneData const &scene, scene::World &world) {
    auto &resources = world.resources();
    auto const materials = addMaterials(scene, resources);
    auto meshes = std::vector<core::MeshId>{};
    for (auto const &mesh : scene.meshes) {
        meshes.push_back(resources.addMesh(mesh.mesh));
    }
    auto const root = world.createEntity(core::Transform::identity(), std::nullopt);
    for (auto const &instance : scene.instances) {
        auto const entity = world.createEntity(core::decomposed(instance.objectToWorld), root);
        world.setRenderable(entity,
                            scene::Renderable{meshes[instance.mesh], materials[scene.meshes[instance.mesh].material]});
    }
    for (auto const &light : scene.lights) {
        world.setLight(world.createEntity(core::Transform::identity(), root), light);
    }
    if (scene.lights.empty()) {
        world.setLight(world.createEntity(core::Transform::identity(), root), defaultSun());
    }
    for (auto const mesh : meshes) {
        resources.releaseMesh(mesh);
    }
    for (auto const material : materials) {
        resources.releaseMaterial(material);
    }
    return root;
}

core::CameraDesc cameraFor(asset::SceneData const &scene) {
    return scene.cameras.empty() ? framingCamera(scene.bounds) : scene.cameras.front().camera;
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

core::EnvironmentDesc skyForScenes() { return core::EnvironmentDesc{kSky}; }

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
