#include "AccelerationScene.h"

#include "kernels/PathTrace.h"

#include <lr/render/Bind.h>

#include <algorithm>
#include <utility>

namespace lr::render {

namespace {

// Workgroups one dispatch of a kernel takes along x on every device.
constexpr auto kMaxGroups = std::uint32_t{65535};

std::uint32_t capacityFor(std::uint32_t const count) noexcept {
    auto capacity = std::uint32_t{1};
    while (capacity < count) {
        capacity *= 2;
    }
    return capacity;
}

template <class T>
void retire(RetireList &retired, T &&resource) {
    retired.push_back(std::make_shared<std::decay_t<T>>(std::forward<T>(resource)));
}

} // namespace

void AccelerationScene::apply(SceneUpdate const &update) {
    for (auto const &write : update.meshes) {
        staleMeshes.insert(write.index);
        removedMeshes.erase(write.index);
    }
    for (auto const slot : update.removedMeshes) {
        staleMeshes.erase(slot);
        removedMeshes.insert(slot);
    }
    if (not update.meshes.empty() or not update.removedMeshes.empty() or not update.instances.empty()
        or not update.materials.empty()) {
        staleTopLevel = true;
    }
}

void AccelerationScene::build(tc::Device &device, SceneTables const &tables, RetireList &retired) {
    if (not placeholder) {
        buildPlaceholder(device, retired);
    }
    buildBottomLevels(device, tables, retired);
    buildTopLevel(device, tables, retired);
}

tc::TopLevelAS const *AccelerationScene::topLevel() const noexcept { return top ? &top : nullptr; }

void AccelerationScene::buildPlaceholder(tc::Device &device, RetireList &retired) {
    auto corners = std::make_shared<tc::BufferResource<tc::vec4>>(3);
    (*corners)[0] = tc::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    (*corners)[1] = tc::vec4{1.0f, 0.0f, 0.0f, 1.0f};
    (*corners)[2] = tc::vec4{0.0f, 1.0f, 0.0f, 1.0f};
    auto indices = std::make_shared<tc::BufferResource<tc::uint>>(3);
    (*indices)[1] = 1u;
    (*indices)[2] = 2u;
    device.uploadBuffer(*corners);
    device.uploadBuffer(*indices);
    placeholder = device.createBottomLevel({tc::TriangleGeometry::of(*corners, 0, 3, *indices, 0, 1)});
    retired.push_back(std::move(corners));
    retired.push_back(std::move(indices));
}

// The stale meshes' corners go into one buffer, which lives until their builds have run.
void AccelerationScene::buildBottomLevels(tc::Device &device, SceneTables const &tables, RetireList &retired) {
    auto const slots = tables.meshes->size();
    auto whole = false;
    if (references == nullptr or references->size() < slots) {
        auto grown = std::make_unique<tc::BufferResource<tc::uvec2>>(static_cast<std::int32_t>(slots));
        if (references != nullptr) {
            std::copy(references->begin(), references->end(), grown->begin());
            retired.push_back(std::shared_ptr<void>{std::move(references)});
        }
        references = std::move(grown);
        bottomLevels.resize(slots);
        whole = true;
    }
    for (auto const slot : removedMeshes) {
        if (bottomLevels[slot]) {
            retire(retired, std::exchange(bottomLevels[slot], tc::BottomLevelAS{}));
        }
        (*references)[static_cast<std::int32_t>(slot)] = tc::uvec2{0u, 0u};
    }

    auto vertices = std::uint32_t{0};
    for (auto const slot : staleMeshes) {
        auto const mesh = tables.meshes->get(slot);
        vertices += mesh.triangleCount > 0 ? mesh.vertexCount : 0u;
    }
    auto corners = std::shared_ptr<tc::DeviceBuffer<tc::vec4>>{};
    if (vertices > 0) {
        corners = std::make_shared<tc::DeviceBuffer<tc::vec4>>(device.createDeviceBuffer<tc::vec4>(
            static_cast<std::int32_t>(vertices)));
        auto kernel = kernels::CopyPositions{};
        device.useKernel(kernel);
        bind(device, kernel.vertices, *tables.vertices);
        bind(device, kernel.positions, *corners);
        auto target = std::uint32_t{0};
        for (auto const slot : staleMeshes) {
            auto const mesh = tables.meshes->get(slot);
            if (mesh.triangleCount == 0) {
                continue;
            }
            for (auto done = std::uint32_t{0}; done < mesh.vertexCount; done += kMaxGroups * 64) {
                auto const count = std::min(mesh.vertexCount - done, kMaxGroups * 64);
                set(device, kernel.first, mesh.firstVertex + done);
                set(device, kernel.count, count);
                set(device, kernel.target, target + done);
                device.execute(kernel, tc::uvec3{groupsFor(count, 64), 1, 1});
            }
            target += mesh.vertexCount;
        }
    }

    auto target = std::size_t{0};
    for (auto const slot : staleMeshes) {
        auto const mesh = tables.meshes->get(slot);
        if (bottomLevels[slot]) {
            retire(retired, std::exchange(bottomLevels[slot], tc::BottomLevelAS{}));
        }
        auto reference = tc::uvec2{0u, 0u};
        if (mesh.triangleCount > 0) {
            bottomLevels[slot] = device.createBottomLevel({tc::TriangleGeometry::of(
                *corners, target, mesh.vertexCount, *tables.indices, mesh.firstIndex, mesh.triangleCount)});
            reference = bottomLevels[slot].reference();
            target += mesh.vertexCount;
        }
        (*references)[static_cast<std::int32_t>(slot)] = reference;
    }
    if (whole or not staleMeshes.empty() or not removedMeshes.empty()) {
        device.uploadBuffer(*references);
    }
    if (corners != nullptr) {
        retired.push_back(std::move(corners));
    }
    staleMeshes.clear();
    removedMeshes.clear();
}

void AccelerationScene::buildTopLevel(tc::Device &device, SceneTables const &tables, RetireList &retired) {
    auto const count = tables.instanceCount;
    if (top and not staleTopLevel and count == builtInstances) {
        return;
    }
    auto const capacity = capacityFor(count);
    if (not top or top.capacity() < capacity) {
        if (top) {
            retire(retired, std::move(top));
            retired.push_back(std::move(instances));
        }
        top = device.createTopLevel(capacity);
        instances = std::make_shared<tc::DeviceBuffer<tc::TopLevelInstance>>(
            device.createDeviceBuffer<tc::TopLevelInstance>(static_cast<std::int32_t>(capacity)));
    }
    if (count > 0) {
        auto kernel = kernels::WriteTopLevelInstances{};
        device.useKernel(kernel);
        bind(device, kernel.instances, *tables.instances);
        bind(device, kernel.materials, *tables.materials);
        bind(device, kernel.references, *references);
        bind(device, kernel.topLevel, *instances);
        set(device, kernel.count, count);
        set(device, kernel.placeholder, placeholder.reference());
        for (auto first = std::uint32_t{0}; first < count; first += kMaxGroups * 64) {
            set(device, kernel.first, first);
            device.execute(kernel, tc::uvec3{groupsFor(std::min(count - first, kMaxGroups * 64), 64), 1, 1});
        }
    }
    device.buildTopLevel(top, *instances, 0, count);
    builtInstances = count;
    staleTopLevel = false;
}

} // namespace lr::render
