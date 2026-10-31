#pragma once

#include "TrackedTable.h"

#include <lr/scene/WorldResources.h>

#include <map>
#include <optional>
#include <tuple>
#include <unordered_map>

namespace lr::scene {

struct TextureData {
    core::ImageHandle image;
    core::TextureSampler sampler;
};

// The World's resources. Besides WorldResources, the world retains what its renderables
// use and extracts the resources' changes into its deltas.
struct ResourceRegistry final : WorldResources {
    core::MeshId addMesh(core::MeshHandle const &mesh) override;
    void replaceMesh(core::MeshId id, core::MeshHandle const &mesh) override;
    void releaseMesh(core::MeshId id) override;
    core::TextureId addTexture(core::ImageHandle const &image, core::TextureSampler const &sampler) override;
    void replaceTexture(core::TextureId id, core::ImageHandle const &image, core::TextureSampler const &sampler) override;
    void releaseTexture(core::TextureId id) override;
    void resubmitTextures(std::vector<core::TextureId> const &lost) override;
    core::MaterialId addMaterial(core::MaterialDesc const &material) override;
    void replaceMaterial(core::MaterialId id, core::MaterialDesc const &material) override;
    void releaseMaterial(core::MaterialId id) override;
    core::MaterialId defaultMaterial() override;
    ResourceCounts counts() const noexcept override;

    // Both throw std::invalid_argument for ids that are not this world's.
    void checkMesh(core::MeshId id) const;
    void checkMaterial(core::MaterialId id) const;
    void retain(core::MeshId const id) noexcept { meshes.retain(id); }
    void retain(core::MaterialId const id) noexcept { materials.retain(id); }

    void extractInto(core::RenderWorldDelta &delta);

private:
    struct TextureKey {
        core::ImageData const *image;
        core::TextureWrap wrapU;
        core::TextureWrap wrapV;

        friend bool operator<(TextureKey const &a, TextureKey const &b) noexcept {
            return std::tie(a.image, a.wrapU, a.wrapV) < std::tie(b.image, b.wrapU, b.wrapV);
        }
    };

    static TextureKey keyOf(core::ImageHandle const &image, core::TextureSampler const &sampler) noexcept;
    void checkTexture(core::TextureId id) const;
    void checkTextures(core::MaterialDesc const &material) const;
    void retainTextures(core::MaterialDesc const &material) noexcept;
    void releaseTextures(core::MaterialDesc const &material);

    TrackedTable<core::MeshId, core::MeshHandle> meshes;
    TrackedTable<core::TextureId, TextureData> textures;
    TrackedTable<core::MaterialId, core::MaterialDesc> materials;
    std::unordered_map<core::MeshData const *, core::MeshId> meshIds;
    std::map<TextureKey, core::TextureId> textureIds;
    std::optional<core::MaterialId> defaultId;
};

} // namespace lr::scene
