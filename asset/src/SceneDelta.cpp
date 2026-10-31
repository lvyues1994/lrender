#include <lr/asset/SceneDelta.h>

#include <optional>
#include <vector>

namespace lr::asset {

namespace {

struct SequentialIds final : IdAllocator {
    explicit SequentialIds(std::uint32_t const first)
        : meshes(first), textures(first), materials(first), instances(first), lights(first) {}

    core::MeshId newMesh() override { return core::MeshId{meshes++}; }
    core::TextureId newTexture() override { return core::TextureId{textures++}; }
    core::MaterialId newMaterial() override { return core::MaterialId{materials++}; }
    core::InstanceId newInstance() override { return core::InstanceId{instances++}; }
    core::LightId newLight() override { return core::LightId{lights++}; }

private:
    std::uint32_t meshes;
    std::uint32_t textures;
    std::uint32_t materials;
    std::uint32_t instances;
    std::uint32_t lights;
};

std::optional<core::TextureId> remapped(std::optional<core::TextureId> const local,
                                        std::vector<core::TextureId> const &textures) {
    if (not local or local->value >= textures.size()) {
        return std::nullopt;
    }
    return textures[local->value];
}

} // namespace

std::unique_ptr<IdAllocator> makeSequentialIds(std::uint32_t const first) {
    return std::make_unique<SequentialIds>(first);
}

SceneDelta sceneDelta(SceneData const &scene, IdAllocator &ids) {
    auto result = SceneDelta{};
    auto &delta = result.delta;
    auto &removal = result.removal;
    auto meshes = std::vector<core::MeshId>{};
    for (auto const &mesh : scene.meshes) {
        auto const id = meshes.emplace_back(ids.newMesh());
        delta.meshes.push_back(core::MeshUpsert{id, mesh.mesh});
        removal.meshes.push_back(id);
    }
    auto textures = std::vector<core::TextureId>{};
    for (auto const &texture : scene.textures) {
        auto const id = textures.emplace_back(ids.newTexture());
        delta.textures.push_back(core::TextureUpsert{id, texture.image, texture.sampler});
        removal.textures.push_back(id);
    }
    auto materials = std::vector<core::MaterialId>{};
    for (auto material : scene.materials) {
        material.baseColorTexture = remapped(material.baseColorTexture, textures);
        material.metallicRoughnessTexture = remapped(material.metallicRoughnessTexture, textures);
        material.emissiveTexture = remapped(material.emissiveTexture, textures);
        material.normalTexture = remapped(material.normalTexture, textures);
        auto const id = materials.emplace_back(ids.newMaterial());
        delta.materials.push_back(core::MaterialUpsert{id, material});
        removal.materials.push_back(id);
    }
    for (auto const &instance : scene.instances) {
        auto const id = ids.newInstance();
        auto const material = materials[scene.meshes[instance.mesh].material];
        delta.instances.push_back(core::InstanceUpsert{id, meshes[instance.mesh], material, instance.objectToWorld});
        removal.instances.push_back(id);
    }
    for (auto const &light : scene.lights) {
        auto const id = ids.newLight();
        delta.lights.push_back(core::LightUpsert{id, light});
        removal.lights.push_back(id);
    }
    return result;
}

} // namespace lr::asset
