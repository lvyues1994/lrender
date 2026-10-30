#pragma once

#include <lr/core/Id.h>
#include <lr/core/Math.h>
#include <lr/core/Mesh.h>
#include <lr/core/Scene.h>

#include <optional>
#include <vector>

namespace lr::core {

struct MeshUpsert {
    MeshId id;
    MeshHandle mesh;
};

struct MaterialUpsert {
    MaterialId id;
    MaterialDesc material;
};

struct InstanceUpsert {
    InstanceId id;
    MeshId mesh;
    MaterialId material;
    mat4 objectToWorld{1.0f};
};

struct LightUpsert {
    LightId id;
    LightDesc light;
};

struct Removals {
    std::vector<MeshId> meshes;
    std::vector<MaterialId> materials;
    std::vector<InstanceId> instances;
    std::vector<LightId> lights;
};

// What changed since the previous frame; anything unchanged is left out. An upsert of a
// known id replaces it. Removals apply after the upserts of the same delta.
struct RenderWorldDelta {
    std::vector<MeshUpsert> meshes;
    std::vector<MaterialUpsert> materials;
    std::vector<InstanceUpsert> instances;
    std::vector<LightUpsert> lights;
    std::optional<EnvironmentDesc> environment;
    Removals removed;
};

} // namespace lr::core
