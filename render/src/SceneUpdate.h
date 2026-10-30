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

struct SceneCapacity {
    std::uint32_t vertices = 1;
    std::uint32_t indices = 1;
    std::uint32_t meshes = 1;
    std::uint32_t instances = 1;
    std::uint32_t materials = 1;
    std::uint32_t lights = 1;
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
};

} // namespace lr::render
