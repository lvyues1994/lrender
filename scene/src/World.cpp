#include <lr/scene/World.h>

#include "Hierarchy.h"
#include "ResourceRegistry.h"
#include "TrackedTable.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lr::scene {

namespace {

struct InstanceData {
    std::uint32_t entity;
    Renderable renderable;
};

struct LightData {
    std::uint32_t entity;
    core::LightDesc light;
};

core::LightDesc inWorld(core::LightDesc light, core::mat4 const &world) noexcept {
    if (auto *const directional = std::get_if<core::DirectionalLight>(&light)) {
        directional->direction = (world * core::vec4{directional->direction, 0.0f}).xyz();
    } else if (auto *const point = std::get_if<core::PointLight>(&light)) {
        point->position = (world * core::vec4{point->position, 1.0f}).xyz();
    }
    return light;
}

std::string describe(EntityId const entity) {
    return "lr::scene: entity " + std::to_string(entity.index) + " (generation " + std::to_string(entity.generation)
         + ")";
}

struct WorldImpl final : World {
    explicit WorldImpl(WorldConfig const &config) : propagation{config.runtime, config.parallelTransforms} {}

    EntityId createEntity(core::Transform const &local, std::optional<EntityId> const parent) override {
        auto const parentSlot = parent ? slotOf(*parent) : kNoEntity;
        auto const entity = hierarchy.create(local, parentSlot);
        instanceOf.resize(hierarchy.slotCount());
        lightOf.resize(hierarchy.slotCount());
        lensOf.resize(hierarchy.slotCount());
        return hierarchy.idOf(entity);
    }

    void destroyEntity(EntityId const entity) override {
        for (auto const destroyed : hierarchy.destroy(slotOf(entity))) {
            clearRenderable(destroyed);
            clearLight(destroyed);
            lensOf[destroyed].reset();
        }
    }

    bool isAlive(EntityId const entity) const noexcept override { return hierarchy.isAlive(entity); }
    std::size_t entityCount() const noexcept override { return hierarchy.size(); }

    void setParent(EntityId const entity, std::optional<EntityId> const parent) override {
        auto const slot = slotOf(entity);
        auto const parentSlot = parent ? slotOf(*parent) : kNoEntity;
        if (parentSlot != kNoEntity and hierarchy.isAncestor(slot, parentSlot)) {
            throw std::invalid_argument{describe(entity) + " cannot become a descendant of itself"};
        }
        hierarchy.setParent(slot, parentSlot);
    }

    std::optional<EntityId> parentOf(EntityId const entity) const override {
        auto const parent = hierarchy.parentOf(slotOf(entity));
        if (parent == kNoEntity) {
            return std::nullopt;
        }
        return hierarchy.idOf(parent);
    }

    void setTransform(EntityId const entity, core::Transform const &local) override {
        hierarchy.setLocal(slotOf(entity), local);
    }

    core::Transform transformOf(EntityId const entity) const override { return hierarchy.localOf(slotOf(entity)); }
    core::mat4 worldMatrix(EntityId const entity) const override { return hierarchy.worldOf(slotOf(entity)); }

    void setRenderable(EntityId const entity, Renderable const &renderable) override {
        auto const slot = slotOf(entity);
        registry.checkMesh(renderable.mesh);
        registry.checkMaterial(renderable.material);
        registry.retain(renderable.mesh);
        registry.retain(renderable.material);
        auto const id = instanceOf[slot];
        if (not id.isValid()) {
            instanceOf[slot] = instances.insert(InstanceData{slot, renderable});
            return;
        }
        auto const old = instances[id].renderable;
        instances.modify(id).renderable = renderable;
        registry.releaseMesh(old.mesh);
        registry.releaseMaterial(old.material);
    }

    void removeRenderable(EntityId const entity) override { clearRenderable(slotOf(entity)); }

    std::optional<Renderable> renderableOf(EntityId const entity) const override {
        auto const id = instanceOf[slotOf(entity)];
        if (not id.isValid()) {
            return std::nullopt;
        }
        return instances[id].renderable;
    }

    void setLight(EntityId const entity, core::LightDesc const &light) override {
        auto const slot = slotOf(entity);
        auto const id = lightOf[slot];
        if (id.isValid()) {
            lights.modify(id).light = light;
        } else {
            lightOf[slot] = lights.insert(LightData{slot, light});
        }
    }

    void removeLight(EntityId const entity) override { clearLight(slotOf(entity)); }

    void setCamera(EntityId const entity, CameraLens const &lens) override { lensOf[slotOf(entity)] = lens; }

    core::CameraDesc cameraOf(EntityId const entity) const override {
        auto const slot = slotOf(entity);
        auto const &lens = lensOf[slot];
        if (not lens) {
            throw std::invalid_argument{describe(entity) + " has no camera"};
        }
        auto const world = hierarchy.worldOf(slot);
        auto camera = core::CameraDesc{};
        camera.position = world[3].xyz();
        camera.target = camera.position - tc::normalize(world[2].xyz());
        camera.up = tc::normalize(world[1].xyz());
        camera.verticalFov = lens->verticalFov;
        camera.nearPlane = lens->nearPlane;
        camera.farPlane = lens->farPlane;
        return camera;
    }

    void setEnvironment(core::EnvironmentDesc const &environment_) override { environment = environment_; }

    WorldResources &resources() noexcept override { return registry; }

    core::RenderWorldDelta extractDelta() override {
        moved.clear();
        auto const propagated = hierarchy.propagate(propagation, moved);
        statistics.propagatedTransforms += propagated.transforms;
        statistics.parallelPropagations += propagated.parallel ? 1u : 0u;
        for (auto const entity : moved) {
            if (instanceOf[entity].isValid()) {
                instances.markChanged(instanceOf[entity]);
            }
            if (lightOf[entity].isValid()) {
                lights.markChanged(lightOf[entity]);
            }
        }
        auto delta = core::RenderWorldDelta{};
        registry.extractInto(delta);
        instances.extract(
            [this, &delta](core::InstanceId const id, InstanceData const &instance) {
                delta.instances.push_back({id, instance.renderable.mesh, instance.renderable.material,
                                           hierarchy.worldOf(instance.entity)});
            },
            delta.removed.instances);
        lights.extract(
            [this, &delta](core::LightId const id, LightData const &light) {
                delta.lights.push_back({id, inWorld(light.light, hierarchy.worldOf(light.entity))});
            },
            delta.removed.lights);
        delta.environment = std::exchange(environment, std::nullopt);
        return delta;
    }

    WorldStats stats() const noexcept override { return statistics; }

private:
    std::uint32_t slotOf(EntityId const entity) const {
        if (not hierarchy.isAlive(entity)) {
            throw std::invalid_argument{describe(entity) + " is not alive"};
        }
        return entity.index;
    }

    void clearRenderable(std::uint32_t const slot) {
        auto const id = std::exchange(instanceOf[slot], core::InstanceId{});
        if (not id.isValid()) {
            return;
        }
        auto const renderable = instances[id].renderable;
        instances.erase(id);
        registry.releaseMesh(renderable.mesh);
        registry.releaseMaterial(renderable.material);
    }

    void clearLight(std::uint32_t const slot) {
        auto const id = std::exchange(lightOf[slot], core::LightId{});
        if (id.isValid()) {
            lights.erase(id);
        }
    }

    Hierarchy hierarchy;
    ResourceRegistry registry;
    TrackedTable<core::InstanceId, InstanceData> instances;
    TrackedTable<core::LightId, LightData> lights;
    // By entity slot: invalid ids and nothing where the entity has no such component.
    std::vector<core::InstanceId> instanceOf;
    std::vector<core::LightId> lightOf;
    std::vector<std::optional<CameraLens>> lensOf;
    std::optional<core::EnvironmentDesc> environment;
    PropagationConfig propagation;
    WorldStats statistics;
    std::vector<std::uint32_t> moved;
};

} // namespace

EntityId World::spawnInstance(core::MeshHandle const &mesh, core::Transform const &local) {
    return spawnInstance(mesh, local, resources().defaultMaterial());
}

EntityId World::spawnInstance(core::MeshHandle const &mesh, core::Transform const &local,
                              core::MaterialId const material) {
    auto &registry = resources();
    auto const meshId = registry.addMesh(mesh);
    auto entity = EntityId{};
    try {
        entity = createEntity(local, std::nullopt);
        setRenderable(entity, Renderable{meshId, material});
    } catch (...) {
        if (isAlive(entity)) {
            destroyEntity(entity);
        }
        registry.releaseMesh(meshId);
        throw;
    }
    registry.releaseMesh(meshId);
    return entity;
}

std::unique_ptr<World> makeWorld(WorldConfig const &config) { return std::make_unique<WorldImpl>(config); }

} // namespace lr::scene
