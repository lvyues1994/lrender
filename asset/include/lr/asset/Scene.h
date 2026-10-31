#pragma once

#include <lr/core/Image.h>
#include <lr/core/Math.h>
#include <lr/core/Mesh.h>
#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lr::asset {

// One glTF primitive and the material it is drawn with.
struct SceneMesh {
    core::MeshHandle mesh;
    std::uint32_t material = 0; // into SceneData::materials
};

struct SceneTexture {
    core::ImageHandle image;
    core::TextureSampler sampler;
};

struct SceneInstance {
    std::uint32_t mesh = 0; // into SceneData::meshes
    core::mat4 objectToWorld{1.0f};
};

struct SceneCamera {
    core::CameraDesc camera;          // world space
    std::optional<float> aspectRatio; // width over height, where the file gives one
};

// A glTF file's default scene, its node hierarchy flattened into world-space instances,
// lights and cameras. The TextureIds in `materials` index `textures`; sceneDelta() maps
// them to the ids it hands out.
struct SceneData {
    std::vector<SceneMesh> meshes;
    std::vector<SceneTexture> textures; // an image used both as colour and as data is here twice
    std::vector<core::MaterialDesc> materials;
    std::vector<SceneInstance> instances;
    std::vector<core::LightDesc> lights;
    std::vector<SceneCamera> cameras;
    core::Aabb bounds;                 // of the instances; all zero without any
    std::vector<std::string> warnings; // what of the file was left out or approximated
};

using SceneHandle = std::shared_ptr<SceneData const>;

} // namespace lr::asset
