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
// them; meshes and materials keep their slots. Geometry of replaced or removed meshes is
// not reclaimed yet.
struct RenderWorld {
    // Throws std::invalid_argument, leaving the world unchanged, when the delta refers to
    // ids it does not know or carries malformed meshes.
    SceneUpdate apply(core::RenderWorldDelta const &delta);

    std::uint32_t instanceCount() const noexcept { return static_cast<std::uint32_t>(instances.size()); }
    std::uint32_t lightCount() const noexcept { return static_cast<std::uint32_t>(lights.size()); }
    core::EnvironmentDesc const &environment() const noexcept { return sky; }

private:
    struct MeshRecord {
        std::uint32_t slot;
        GpuMesh gpu;
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
};

} // namespace lr::render
