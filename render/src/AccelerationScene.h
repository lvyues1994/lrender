#pragma once

#include "RetireList.h"
#include "SceneUpdate.h"

#include <lr/render/GpuTypes.h>

#include <tc/device.hpp>

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace lr::render {

// The GPU thread's acceleration structures for ray queries (RendererCapabilities::
// rayQueries): a bottom level per mesh with triangles, built on the device from corners
// CopyPositions takes out of the vertex table, and a top level over every instance, which
// WriteTopLevelInstances fills on the device. It follows the SceneUpdates DeviceScene
// applies and builds only for frames that trace rays, so a scene nothing traces costs
// nothing. Top levels do not keep the bottom levels they were built with alive, and frames
// still in flight may trace them, so what is replaced goes to the retire list.
struct AccelerationScene {
    // Every frame, after DeviceScene::apply.
    void apply(SceneUpdate const &update);
    // Before the passes of a frame that traces rays; `tables` are DeviceScene's.
    void build(tc::Device &device, SceneTables const &tables, RetireList &retired);
    // Null before the first build().
    tc::TopLevelAS const *topLevel() const noexcept;

private:
    void buildPlaceholder(tc::Device &device, RetireList &retired);
    void buildBottomLevels(tc::Device &device, SceneTables const &tables, RetireList &retired);
    void buildTopLevel(tc::Device &device, SceneTables const &tables, RetireList &retired);

    std::set<std::uint32_t> staleMeshes;   // slots written since their bottom levels were built
    std::set<std::uint32_t> removedMeshes; // slots whose meshes went away since
    bool staleTopLevel = true;
    std::uint32_t builtInstances = 0;

    std::vector<tc::BottomLevelAS> bottomLevels;                // by mesh slot; empty without triangles
    std::unique_ptr<tc::BufferResource<tc::uvec2>> references; // reference() of each, (0, 0) for none
    tc::BottomLevelAS placeholder;                              // one triangle, for meshes without any
    tc::TopLevelAS top;
    std::shared_ptr<tc::DeviceBuffer<tc::TopLevelInstance>> instances; // what `top` is built from
};

} // namespace lr::render
