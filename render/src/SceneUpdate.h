#pragma once

#include <lr/render/GpuTypes.h>

#include <cstdint>
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
    std::vector<Indexed<GpuMaterial>> materials;
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
};

} // namespace lr::render
