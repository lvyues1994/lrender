#pragma once

#include "SceneUpdate.h"

#include <lr/core/RenderWorldDelta.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace lr::render {

// The render thread's copy of the scene. Each frame applies a RenderWorldDelta and turns
// the changes into the SceneUpdate the GPU thread applies to its tables. Instances and
// lights stay dense (removal moves the last one into the gap) because kernels loop over
// them; meshes and materials keep their slots. Geometry and hierarchies of replaced or
// removed meshes are not reclaimed yet.
struct RenderWorld {
    // Throws std::invalid_argument, leaving the world unchanged, when the delta refers to
    // ids it does not know or carries malformed meshes.
    SceneUpdate apply(core::RenderWorldDelta const &delta);

    // Everything the world holds, for a GPU copy that starts out empty (after the device
    // was lost): geometry from the meshes' handles, every slot, and the hierarchies, built
    // anew, when they are kept.
    SceneUpdate fullUpdate();

    // From now on keeps a hierarchy over the triangles of each mesh and one over the
    // instances: builds them into `update`, and later apply() calls rebuild what their
    // deltas change.
    void keepHierarchies(SceneUpdate &update);
    bool keepsHierarchies() const noexcept { return hierarchies; }

    std::uint32_t instanceCount() const noexcept { return static_cast<std::uint32_t>(instances.size()); }
    std::uint32_t lightCount() const noexcept { return static_cast<std::uint32_t>(lights.size()); }
    core::EnvironmentDesc const &environment() const noexcept { return sky; }
    // Counts the deltas that changed meshes, materials, instances, lights or the sky.
    std::uint64_t version() const noexcept { return changes; }

private:
    struct MeshRecord {
        std::uint32_t slot;
        GpuMesh gpu;
        core::MeshHandle data;
        // While hierarchies are kept: the root of the mesh's, and the bounds of its triangles.
        std::uint32_t blasRoot;
        core::Aabb blasBounds;
    };
    struct MaterialRecord {
        std::uint32_t slot;
        core::MaterialDesc desc;
    };
    struct InstanceRecord {
        core::InstanceId id;
        core::MeshId mesh;
        core::MaterialId material;
        core::mat4 objectToWorld;
    };
    struct LightRecord {
        core::LightId id;
        core::LightDesc desc;
    };

    void validate(core::RenderWorldDelta const &delta) const;
    std::uint32_t takeSlot(std::vector<std::uint32_t> &freeSlots, std::uint32_t &end);
    void upsertMesh(core::MeshUpsert const &upsert, SceneUpdate &update);
    void upsertMaterial(core::MaterialUpsert const &upsert, SceneUpdate &update);
    void writeInstance(std::uint32_t index, SceneUpdate &update) const;
    void writeLight(std::uint32_t index, SceneUpdate &update) const;
    void removeInstance(core::InstanceId id, SceneUpdate &update);
    void removeLight(core::LightId id, SceneUpdate &update);
    void updateHierarchies(core::RenderWorldDelta const &delta, SceneUpdate &update);
    void buildBlas(MeshRecord &record, SceneUpdate &update);
    void buildTlas(SceneUpdate &update);
    SceneCapacity capacity() const noexcept;

    std::unordered_map<core::MeshId, MeshRecord> meshes;
    std::vector<std::uint32_t> freeMeshSlots;
    std::uint32_t meshSlotEnd = 0;
    std::uint32_t vertexEnd = 0;
    std::uint32_t indexEnd = 0;

    std::unordered_map<core::MaterialId, MaterialRecord> materials;
    std::vector<std::uint32_t> freeMaterialSlots;
    std::uint32_t materialSlotEnd = 0;

    std::vector<InstanceRecord> instances;
    std::unordered_map<core::InstanceId, std::uint32_t> instanceIndex;

    std::vector<LightRecord> lights;
    std::unordered_map<core::LightId, std::uint32_t> lightIndex;

    core::EnvironmentDesc sky;
    std::uint64_t changes = 0;

    bool hierarchies = false;
    std::uint32_t blasNodeEnd = 0;
    std::uint32_t blasTriangleEnd = 0;
    std::uint32_t tlasNodeCount = 0;
};

} // namespace lr::render
