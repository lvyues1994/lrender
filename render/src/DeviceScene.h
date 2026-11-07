#pragma once

#include "RetireList.h"
#include "SceneUpdate.h"
#include "TexturePool.h"

#include <lr/render/GpuTypes.h>
#include <lr/render/Renderer.h>

#include <tc/device.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace lr::render {

// The GPU thread's copy of the scene: device tables kept in step with RenderWorld by
// applying its SceneUpdates in frame order, their moves first. Resized tables (grown, or
// shrunk once more than four times as large as they need to be) are uploaded whole and
// the ones they replace go to `retired`; otherwise only the ranges an update moves to or
// writes go up. Uploads
// copy the host tables at the call, so an update may be applied while earlier frames
// still read the tables on the device. Materials refer to the textures that are resident
// in `textures`, and are written again whenever that changes.
struct DeviceScene {
    void apply(tc::Device &device, SceneUpdate const &update, TexturePool &textures, RetireList &retired);
    SceneTables tables() const noexcept;
    // What the tables take on the device (SceneMemory, textures aside).
    SceneMemory memory() const noexcept;

private:
    void applyHierarchies(tc::Device &device, SceneUpdate const &update, RetireList &retired);
    void applyMaterials(tc::Device &device, SceneUpdate const &update, TexturePool &textures, bool grown);

    std::unique_ptr<tc::TableResource<GpuVertex, VertexLayout>> vertices;
    std::unique_ptr<tc::BufferResource<tc::uint>> indices;
    std::unique_ptr<tc::TableResource<GpuCluster>> clusters;
    std::unique_ptr<tc::TableResource<GpuMesh>> meshes;
    std::unique_ptr<tc::TableResource<GpuInstance, InstanceLayout>> instances;
    std::unique_ptr<tc::TableResource<GpuMaterial>> materials;
    std::unique_ptr<tc::TableResource<GpuLight>> lights;
    tc::uint instanceCount = 0;
    tc::uint lightCount = 0;
    std::vector<std::optional<MaterialEncoding>> materialSources; // by slot

    std::unique_ptr<tc::TableResource<GpuBvhNode>> blasNodes;
    std::unique_ptr<tc::TableResource<GpuBvhTriangle>> blasTriangles;
    std::unique_ptr<tc::BufferResource<tc::uint>> blasRoots;
    std::unique_ptr<tc::TableResource<GpuBvhNode>> tlasNodes;
    tc::uint tlasNodeCount = 0;
};

} // namespace lr::render
