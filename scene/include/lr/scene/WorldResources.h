#pragma once

#include <lr/core/Id.h>
#include <lr/core/Image.h>
#include <lr/core/Mesh.h>
#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>

#include <cstddef>
#include <vector>

namespace lr::scene {

struct ResourceCounts {
    std::size_t meshes = 0;
    std::size_t textures = 0;
    std::size_t materials = 0;
};

// A World's meshes, textures and materials under the ids its renderables and materials
// refer to. Every add gives the caller a reference to release; a renderable also holds its
// mesh and material, a material its textures. The render world gets a resource in the
// first delta after it is added or replaced, and loses it in the first delta after its
// last reference goes. Unknown ids throw std::invalid_argument.
struct WorldResources {
    virtual ~WorldResources() = default;

    // The same handle again gives the same id, with one more reference.
    virtual core::MeshId addMesh(core::MeshHandle const &mesh) = 0;
    // Hot reloading: the id stays, the next delta upserts the new data.
    virtual void replaceMesh(core::MeshId id, core::MeshHandle const &mesh) = 0;
    virtual void releaseMesh(core::MeshId id) = 0;

    // The same image with the same sampler gives the same id, with one more reference.
    virtual core::TextureId addTexture(core::ImageHandle const &image, core::TextureSampler const &sampler) = 0;
    virtual void replaceTexture(core::TextureId id, core::ImageHandle const &image,
                                core::TextureSampler const &sampler) = 0;
    virtual void releaseTexture(core::TextureId id) = 0;
    // The textures a lost device held (render::FrameResult::lostTextures) go into the
    // next delta again; those released since are left out.
    virtual void resubmitTextures(std::vector<core::TextureId> const &lost) = 0;

    // The material's textures must be this world's.
    virtual core::MaterialId addMaterial(core::MaterialDesc const &material) = 0;
    virtual void replaceMaterial(core::MaterialId id, core::MaterialDesc const &material) = 0;
    virtual void releaseMaterial(core::MaterialId id) = 0;
    // A MaterialDesc{} the world adds on first use and holds on to.
    virtual core::MaterialId defaultMaterial() = 0;

    virtual ResourceCounts counts() const noexcept = 0;
};

} // namespace lr::scene
