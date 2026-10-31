#pragma once

#include <lr/scene/WorldResources.h>

#include <lr/core/Id.h>
#include <lr/core/Math.h>
#include <lr/core/Mesh.h>
#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>
#include <lr/runtime/Runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace lr::scene {

using core::EntityId;

// An entity drawn with a mesh and a material; it becomes one instance of the render world.
struct Renderable {
    core::MeshId mesh;
    core::MaterialId material;
};

// A camera's projection. The camera sits at its entity's origin, looks down the entity's
// -z axis with its +y axis up.
struct CameraLens {
    float verticalFov = 0.8f; // radians
    float nearPlane = 0.05f;
    float farPlane = 500.0f;
};

struct WorldConfig {
    // Optional; outlives the world. With it, propagating at least `parallelTransforms`
    // world transforms runs on its frame pool, which extractDelta() waits for, so the
    // world's thread must not be one of the pool's.
    runtime::Runtime *runtime = nullptr;
    std::size_t parallelTransforms = 4096;
};

struct WorldStats {
    std::uint64_t propagatedTransforms = 0; // world transforms computed by extractDelta()
    std::uint64_t parallelPropagations = 0; // extractDelta() calls that used the frame pool
};

// The simulation's scene: entities with a local transform under an optional parent, and
// components: a renderable, a light, a camera. Use it on one thread, the host thread that
// ticks the scripts; ids of destroyed entities throw std::invalid_argument.
//
// extractDelta() hands the render world what changed since the previous call, and only
// that: an instance per renderable entity (added, moved, its mesh or material changed,
// removed), lights, resources and the environment.
struct World {
    virtual ~World() = default;

    // `parent` empty: a root.
    virtual EntityId createEntity(core::Transform const &local, std::optional<EntityId> parent) = 0;
    // Destroys the entity's descendants with it.
    virtual void destroyEntity(EntityId entity) = 0;
    virtual bool isAlive(EntityId entity) const noexcept = 0;
    virtual std::size_t entityCount() const noexcept = 0;

    // An entity's world transform is its parent's world transform times its local one.
    // Reparenting keeps the local transform, so the entity moves with its new parent; an
    // entity cannot become its own descendant.
    virtual void setParent(EntityId entity, std::optional<EntityId> parent) = 0;
    virtual std::optional<EntityId> parentOf(EntityId entity) const = 0;
    virtual void setTransform(EntityId entity, core::Transform const &local) = 0;
    virtual core::Transform transformOf(EntityId entity) const = 0;
    virtual core::mat4 worldMatrix(EntityId entity) const = 0;

    virtual void setRenderable(EntityId entity, Renderable const &renderable) = 0;
    virtual void removeRenderable(EntityId entity) = 0;
    virtual std::optional<Renderable> renderableOf(EntityId entity) const = 0;
    // In the entity's space: a point light at its position, a directional light along its
    // direction, moved with the entity.
    virtual void setLight(EntityId entity, core::LightDesc const &light) = 0;
    virtual void removeLight(EntityId entity) = 0;
    virtual void setCamera(EntityId entity, CameraLens const &lens) = 0;
    // Throws std::invalid_argument when the entity has no camera.
    virtual core::CameraDesc cameraOf(EntityId entity) const = 0;
    virtual void setEnvironment(core::EnvironmentDesc const &environment) = 0;

    virtual WorldResources &resources() noexcept = 0;

    virtual core::RenderWorldDelta extractDelta() = 0;
    virtual WorldStats stats() const noexcept = 0;

    // A new root drawn with `mesh`, which the world adds, and the default material.
    EntityId spawnInstance(core::MeshHandle const &mesh, core::Transform const &local);
    EntityId spawnInstance(core::MeshHandle const &mesh, core::Transform const &local, core::MaterialId material);
};

std::unique_ptr<World> makeWorld(WorldConfig const &config);

} // namespace lr::scene
