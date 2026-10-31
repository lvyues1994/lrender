#pragma once

#include <lr/core/Id.h>
#include <lr/core/Image.h>
#include <lr/core/Math.h>
#include <lr/core/Mesh.h>
#include <lr/core/Scene.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace lr::core {

struct MeshUpsert {
    MeshId id;
    MeshHandle mesh;
};

enum class TextureWrap : std::uint8_t {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
};

struct TextureSampler {
    TextureWrap wrapU = TextureWrap::Repeat;
    TextureWrap wrapV = TextureWrap::Repeat;
};

// The renderer keeps no copy of the image: once the GPU has it, the device holds the only
// one, and after a lost device the frame's result lists the textures to upsert again.
struct TextureUpsert {
    TextureId id;
    ImageHandle image;
    TextureSampler sampler;
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
    std::vector<TextureId> textures;
    std::vector<MaterialId> materials;
    std::vector<InstanceId> instances;
    std::vector<LightId> lights;
};

// What changed since the previous frame; anything unchanged is left out. An upsert of a
// known id replaces it. Removals apply after the upserts of the same delta. Materials may
// refer to textures the world does not have (yet).
struct RenderWorldDelta {
    std::vector<MeshUpsert> meshes;
    std::vector<TextureUpsert> textures;
    std::vector<MaterialUpsert> materials;
    std::vector<InstanceUpsert> instances;
    std::vector<LightUpsert> lights;
    std::optional<EnvironmentDesc> environment;
    Removals removed;
};

} // namespace lr::core
