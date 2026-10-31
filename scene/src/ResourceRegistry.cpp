#include "ResourceRegistry.h"

#include <array>
#include <stdexcept>
#include <string>

namespace lr::scene {

namespace {

std::array<std::optional<core::TextureId>, 4> texturesOf(core::MaterialDesc const &material) noexcept {
    return {material.baseColorTexture, material.metallicRoughnessTexture, material.emissiveTexture,
            material.normalTexture};
}

[[noreturn]] void throwUnknown(char const *const kind, std::uint32_t const id) {
    throw std::invalid_argument{std::string{"lr::scene: the world has no "} + kind + " " + std::to_string(id)};
}

} // namespace

core::MeshId ResourceRegistry::addMesh(core::MeshHandle const &mesh) {
    if (mesh == nullptr) {
        throw std::invalid_argument{"lr::scene: a mesh handle is empty"};
    }
    auto const found = meshIds.find(mesh.get());
    if (found != meshIds.end()) {
        meshes.retain(found->second);
        return found->second;
    }
    auto const id = meshes.insert(mesh);
    meshIds.emplace(mesh.get(), id);
    return id;
}

void ResourceRegistry::replaceMesh(core::MeshId const id, core::MeshHandle const &mesh) {
    checkMesh(id);
    if (mesh == nullptr) {
        throw std::invalid_argument{"lr::scene: a mesh handle is empty"};
    }
    auto const old = meshIds.find(meshes[id].get());
    if (old != meshIds.end() and old->second == id) {
        meshIds.erase(old);
    }
    meshIds.emplace(mesh.get(), id);
    meshes.modify(id) = mesh;
}

void ResourceRegistry::releaseMesh(core::MeshId const id) {
    checkMesh(id);
    auto const mesh = meshes[id];
    if (meshes.release(id)) {
        auto const found = meshIds.find(mesh.get());
        if (found != meshIds.end() and found->second == id) {
            meshIds.erase(found);
        }
    }
}

core::TextureId ResourceRegistry::addTexture(core::ImageHandle const &image, core::TextureSampler const &sampler) {
    if (image == nullptr) {
        throw std::invalid_argument{"lr::scene: an image handle is empty"};
    }
    auto const key = keyOf(image, sampler);
    auto const found = textureIds.find(key);
    if (found != textureIds.end()) {
        textures.retain(found->second);
        return found->second;
    }
    auto const id = textures.insert(TextureData{image, sampler});
    textureIds.emplace(key, id);
    return id;
}

void ResourceRegistry::replaceTexture(core::TextureId const id, core::ImageHandle const &image,
                                      core::TextureSampler const &sampler) {
    checkTexture(id);
    if (image == nullptr) {
        throw std::invalid_argument{"lr::scene: an image handle is empty"};
    }
    auto const &current = textures[id];
    auto const old = textureIds.find(keyOf(current.image, current.sampler));
    if (old != textureIds.end() and old->second == id) {
        textureIds.erase(old);
    }
    textureIds.emplace(keyOf(image, sampler), id);
    textures.modify(id) = TextureData{image, sampler};
}

void ResourceRegistry::releaseTexture(core::TextureId const id) {
    checkTexture(id);
    auto const key = keyOf(textures[id].image, textures[id].sampler);
    if (textures.release(id)) {
        auto const found = textureIds.find(key);
        if (found != textureIds.end() and found->second == id) {
            textureIds.erase(found);
        }
    }
}

void ResourceRegistry::resubmitTextures(std::vector<core::TextureId> const &lost) {
    for (auto const id : lost) {
        if (textures.contains(id)) {
            textures.markChanged(id);
        }
    }
}

core::MaterialId ResourceRegistry::addMaterial(core::MaterialDesc const &material) {
    checkTextures(material);
    auto const id = materials.insert(material);
    retainTextures(material);
    return id;
}

void ResourceRegistry::replaceMaterial(core::MaterialId const id, core::MaterialDesc const &material) {
    checkMaterial(id);
    checkTextures(material);
    retainTextures(material);
    auto const old = materials[id];
    materials.modify(id) = material;
    releaseTextures(old);
}

void ResourceRegistry::releaseMaterial(core::MaterialId const id) {
    checkMaterial(id);
    auto const material = materials[id];
    if (materials.release(id)) {
        releaseTextures(material);
    }
}

core::MaterialId ResourceRegistry::defaultMaterial() {
    if (not defaultId) {
        defaultId = materials.insert(core::MaterialDesc{});
    }
    return *defaultId;
}

ResourceCounts ResourceRegistry::counts() const noexcept {
    return ResourceCounts{meshes.size(), textures.size(), materials.size()};
}

void ResourceRegistry::checkMesh(core::MeshId const id) const {
    if (not meshes.contains(id)) {
        throwUnknown("mesh", id.value);
    }
}

void ResourceRegistry::checkMaterial(core::MaterialId const id) const {
    if (not materials.contains(id)) {
        throwUnknown("material", id.value);
    }
}

void ResourceRegistry::extractInto(core::RenderWorldDelta &delta) {
    meshes.extract([&delta](core::MeshId const id, core::MeshHandle const &mesh) { delta.meshes.push_back({id, mesh}); },
                   delta.removed.meshes);
    textures.extract(
        [&delta](core::TextureId const id, TextureData const &texture) {
            delta.textures.push_back({id, texture.image, texture.sampler});
        },
        delta.removed.textures);
    materials.extract(
        [&delta](core::MaterialId const id, core::MaterialDesc const &material) {
            delta.materials.push_back({id, material});
        },
        delta.removed.materials);
}

ResourceRegistry::TextureKey ResourceRegistry::keyOf(core::ImageHandle const &image,
                                                     core::TextureSampler const &sampler) noexcept {
    return TextureKey{image.get(), sampler.wrapU, sampler.wrapV};
}

void ResourceRegistry::checkTexture(core::TextureId const id) const {
    if (not textures.contains(id)) {
        throwUnknown("texture", id.value);
    }
}

void ResourceRegistry::checkTextures(core::MaterialDesc const &material) const {
    for (auto const texture : texturesOf(material)) {
        if (texture) {
            checkTexture(*texture);
        }
    }
}

void ResourceRegistry::retainTextures(core::MaterialDesc const &material) noexcept {
    for (auto const texture : texturesOf(material)) {
        if (texture) {
            textures.retain(*texture);
        }
    }
}

void ResourceRegistry::releaseTextures(core::MaterialDesc const &material) {
    for (auto const texture : texturesOf(material)) {
        if (texture) {
            releaseTexture(*texture);
        }
    }
}

} // namespace lr::scene
