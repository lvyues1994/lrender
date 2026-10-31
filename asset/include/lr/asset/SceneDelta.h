#pragma once

#include <lr/asset/Scene.h>

#include <lr/core/Id.h>
#include <lr/core/RenderWorldDelta.h>

#include <cstdint>
#include <memory>

namespace lr::asset {

// The ids a scene's elements get in the render world, which the host (or a scene
// module) decides; assets only produce handles.
struct IdAllocator {
    virtual ~IdAllocator() = default;

    virtual core::MeshId newMesh() = 0;
    virtual core::TextureId newTexture() = 0;
    virtual core::MaterialId newMaterial() = 0;
    virtual core::InstanceId newInstance() = 0;
    virtual core::LightId newLight() = 0;
};

// Counts each kind of id up from `first` on its own.
std::unique_ptr<IdAllocator> makeSequentialIds(std::uint32_t first = 0);

struct SceneDelta {
    core::RenderWorldDelta delta; // adds the scene; no environment
    core::Removals removal;       // removes what `delta` added
};

// Gives every mesh, texture, material, instance and light of the scene a new id. Pure
// apart from the allocator, whose ids it takes in that order.
SceneDelta sceneDelta(SceneData const &scene, IdAllocator &ids);

} // namespace lr::asset
