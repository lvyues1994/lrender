#pragma once

#include <lr/core/Id.h>
#include <lr/core/Image.h>
#include <lr/render/GpuTypes.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace lr::render {

template <class T>
struct Indexed {
    std::uint32_t index;
    T value;
};

struct GeometryUpload {
    std::uint32_t firstVertex = 0;
    std::vector<GpuVertex> vertices;
    std::uint32_t firstIndex = 0;
    std::vector<tc::uint> indices;
};

// A material and the textures it refers to (base colour, metallic-roughness, emissive,
// normal; invalid ids for none). The GPU thread fills in GpuMaterial::textures with the
// textures that have finished streaming in.
struct MaterialEncoding {
    GpuMaterial gpu;
    std::array<core::TextureId, 4> textures;
};

// A texture pool to create in a slot, or to grow there: a grown colour pool keeps its
// layers' contents, and the layer count only grows.
struct PoolChange {
    std::uint32_t slot = 0;
    std::optional<tc::CompressedFormat> compressed; // nullopt: an RGBA8 pool
    std::uint32_t size = 0;                         // texels on a side
    std::uint32_t layers = 0;
};

// Levels [0, levels) of `image` for the layer `ref` names. Colour pools get level 0 only
// and generate the rest.
struct TextureUpload {
    core::TextureId id;
    tc::uint ref = 0;
    core::ImageHandle image;
    std::uint32_t levels = 1;
};

// Elements each table must hold. The hierarchy tables stay 0, and are not created, until
// RenderWorld keeps hierarchies.
struct SceneCapacity {
    std::uint32_t vertices = 1;
    std::uint32_t indices = 1;
    std::uint32_t meshes = 1;
    std::uint32_t instances = 1;
    std::uint32_t materials = 1;
    std::uint32_t lights = 1;
    std::uint32_t blasNodes = 0;
    std::uint32_t blasTriangles = 0;
    std::uint32_t tlasNodes = 0;
};

// What the GPU copy of the scene must change for one frame: built on the render thread,
// applied on the GPU thread.
struct SceneUpdate {
    SceneCapacity capacity;
    std::vector<GeometryUpload> geometry;
    std::vector<Indexed<GpuMesh>> meshes;
    std::vector<Indexed<GpuInstance>> instances;
    std::vector<Indexed<MaterialEncoding>> materials;
    std::vector<Indexed<GpuLight>> lights;
    std::uint32_t instanceCount = 0;
    std::uint32_t lightCount = 0;

    // New mesh hierarchies (blasRoots is indexed by mesh slot) and, when instances
    // changed, the whole instance hierarchy.
    std::vector<Indexed<GpuBvhNode>> blasNodes;
    std::vector<Indexed<GpuBvhTriangle>> blasTriangles;
    std::vector<Indexed<tc::uint>> blasRoots;
    std::vector<Indexed<GpuBvhNode>> tlasNodes;
    std::uint32_t tlasNodeCount = 0;

    // Applied in this order: pools, removed textures, then uploads, which replace what
    // their ids had.
    std::vector<PoolChange> pools;
    std::vector<core::TextureId> removedTextures;
    std::vector<TextureUpload> textureUploads;
};

} // namespace lr::render
