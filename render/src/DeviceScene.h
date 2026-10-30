#pragma once

#include "SceneUpdate.h"

#include <lr/render/GpuTypes.h>

#include <tc/device.hpp>

#include <memory>

namespace lr::render {

// The GPU thread's copy of the scene: device tables kept in step with RenderWorld by
// applying its SceneUpdates in frame order. Grown tables are uploaded whole; otherwise
// only the ranges an update writes move.
struct DeviceScene {
    void apply(tc::Device &device, SceneUpdate const &update);
    SceneTables tables() const noexcept;

private:
    void applyHierarchies(tc::Device &device, SceneUpdate const &update);

    std::unique_ptr<tc::TableResource<GpuVertex, VertexLayout>> vertices;
    std::unique_ptr<tc::BufferResource<tc::uint>> indices;
    std::unique_ptr<tc::TableResource<GpuMesh>> meshes;
    std::unique_ptr<tc::TableResource<GpuInstance, InstanceLayout>> instances;
    std::unique_ptr<tc::TableResource<GpuMaterial>> materials;
    std::unique_ptr<tc::TableResource<GpuLight>> lights;
    tc::uint instanceCount = 0;
    tc::uint lightCount = 0;

    std::unique_ptr<tc::TableResource<GpuBvhNode>> blasNodes;
    std::unique_ptr<tc::TableResource<GpuBvhTriangle>> blasTriangles;
    std::unique_ptr<tc::BufferResource<tc::uint>> blasRoots;
    std::unique_ptr<tc::TableResource<GpuBvhNode>> tlasNodes;
    tc::uint tlasNodeCount = 0;
};

} // namespace lr::render
