#pragma once

#include "ImageFiles.h"

#include <lr/asset/AssetResult.h>
#include <lr/asset/AssetSource.h>
#include <lr/asset/Scene.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace lr::asset {

enum class GltfParts : std::uint8_t {
    Geometry,   // meshes and instances; materials without textures
    Everything, // with textures, lights and cameras
};

// A glTF file parsed on the background pool, waiting for the files it refers to, which
// the IO task reads before buildScene() runs on the background pool again.
struct GltfDocument {
    virtual ~GltfDocument() = default;

    // External buffers, then the external images textures use (with Everything).
    virtual std::vector<AssetPath> externalBuffers() const = 0;
    virtual std::vector<AssetPath> externalImages() const = 0;
    virtual void provideBuffer(std::size_t index, Bytes bytes) = 0;
    virtual void provideImage(std::size_t index, Bytes bytes) = 0;

    // Once every external file of `parts` is provided.
    virtual AssetResult<SceneData> buildScene(ImagePolicy const &policy, GltfParts parts) = 0;
};

// A .gltf or .glb file found at `path`.
AssetResult<std::unique_ptr<GltfDocument>> parseGltf(Bytes bytes, AssetPath const &path);

// Every instance's mesh in scene space, front faces still counter-clockwise; empty
// without instances.
core::MeshData mergedMesh(SceneData const &scene);

} // namespace lr::asset
