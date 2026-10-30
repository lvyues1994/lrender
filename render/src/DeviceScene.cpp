#include "DeviceScene.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace lr::render {

namespace {

struct Range {
    std::size_t first;
    std::size_t count;
};

// Ensures `table` holds at least `capacity` elements, keeping its contents. Returns
// whether it was (re)created, in which case it has to be uploaded whole.
template <class T, class Layout>
bool reserveTable(std::unique_ptr<tc::TableResource<T, Layout>> &table, std::size_t const capacity,
                  RetireList &retired) {
    if (table != nullptr and table->size() >= capacity) {
        return false;
    }
    auto grown = std::make_unique<tc::TableResource<T, Layout>>(capacity);
    if (table != nullptr) {
        for (auto index = std::size_t{0}; index < table->size(); ++index) {
            grown->set(index, table->get(index));
        }
        retired.push_back(std::shared_ptr<void>{std::move(table)});
    }
    table = std::move(grown);
    return true;
}

bool reserveBuffer(std::unique_ptr<tc::BufferResource<tc::uint>> &buffer, std::size_t const capacity,
                   RetireList &retired) {
    if (buffer != nullptr and buffer->size() >= capacity) {
        return false;
    }
    auto grown = std::make_unique<tc::BufferResource<tc::uint>>(static_cast<std::int32_t>(capacity));
    if (buffer != nullptr) {
        std::copy(buffer->begin(), buffer->end(), grown->begin());
        retired.push_back(std::shared_ptr<void>{std::move(buffer)});
    }
    buffer = std::move(grown);
    return true;
}

// Sorted, merged ranges of the written indices.
template <class T>
std::vector<Range> rangesOf(std::vector<Indexed<T>> const &writes) {
    auto indices = std::vector<std::size_t>{};
    indices.reserve(writes.size());
    for (auto const &write : writes) {
        indices.push_back(write.index);
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    auto ranges = std::vector<Range>{};
    for (auto const index : indices) {
        if (not ranges.empty() and ranges.back().first + ranges.back().count == index) {
            ranges.back().count += 1;
        } else {
            ranges.push_back(Range{index, 1});
        }
    }
    return ranges;
}

template <class T, class Layout>
void writeTable(tc::Device &device, tc::TableResource<T, Layout> &table, std::vector<Indexed<T>> const &writes,
                bool const whole) {
    for (auto const &write : writes) {
        table.set(write.index, write.value);
    }
    if (whole) {
        device.uploadTable(table);
        return;
    }
    for (auto const range : rangesOf(writes)) {
        device.uploadTable(table, range.first, range.count);
    }
}

void writeBuffer(tc::Device &device, tc::BufferResource<tc::uint> &buffer, std::vector<Indexed<tc::uint>> const &writes,
                 bool const whole) {
    for (auto const &write : writes) {
        buffer[static_cast<std::int32_t>(write.index)] = write.value;
    }
    if (whole) {
        device.uploadBuffer(buffer);
        return;
    }
    for (auto const range : rangesOf(writes)) {
        device.uploadBuffer(buffer, range.first, range.count);
    }
}

} // namespace

void DeviceScene::applyHierarchies(tc::Device &device, SceneUpdate const &update, RetireList &retired) {
    auto const &capacity = update.capacity;
    if (capacity.blasNodes == 0) {
        return;
    }
    auto const newBlasNodes = reserveTable(blasNodes, capacity.blasNodes, retired);
    auto const newBlasTriangles = reserveTable(blasTriangles, capacity.blasTriangles, retired);
    auto const newBlasRoots = reserveBuffer(blasRoots, capacity.meshes, retired);
    auto const newTlasNodes = reserveTable(tlasNodes, capacity.tlasNodes, retired);
    writeTable(device, *blasNodes, update.blasNodes, newBlasNodes);
    writeTable(device, *blasTriangles, update.blasTriangles, newBlasTriangles);
    writeBuffer(device, *blasRoots, update.blasRoots, newBlasRoots);
    writeTable(device, *tlasNodes, update.tlasNodes, newTlasNodes);
    tlasNodeCount = update.tlasNodeCount;
}

void DeviceScene::apply(tc::Device &device, SceneUpdate const &update, RetireList &retired) {
    auto const &capacity = update.capacity;
    auto const newVertices = reserveTable(vertices, capacity.vertices, retired);
    auto const newIndices = reserveBuffer(indices, capacity.indices, retired);
    auto const newMeshes = reserveTable(meshes, capacity.meshes, retired);
    auto const newInstances = reserveTable(instances, capacity.instances, retired);
    auto const newMaterials = reserveTable(materials, capacity.materials, retired);
    auto const newLights = reserveTable(lights, capacity.lights, retired);

    for (auto const &geometry : update.geometry) {
        for (auto index = std::size_t{0}; index < geometry.vertices.size(); ++index) {
            vertices->set(geometry.firstVertex + index, geometry.vertices[index]);
        }
        std::copy(geometry.indices.begin(), geometry.indices.end(), indices->begin() + geometry.firstIndex);
    }
    if (newVertices) {
        device.uploadTable(*vertices);
    }
    if (newIndices) {
        device.uploadBuffer(*indices);
    }
    for (auto const &geometry : update.geometry) {
        if (not newVertices and not geometry.vertices.empty()) {
            device.uploadTable(*vertices, geometry.firstVertex, geometry.vertices.size());
        }
        if (not newIndices and not geometry.indices.empty()) {
            device.uploadBuffer(*indices, geometry.firstIndex, geometry.indices.size());
        }
    }

    writeTable(device, *meshes, update.meshes, newMeshes);
    writeTable(device, *instances, update.instances, newInstances);
    writeTable(device, *materials, update.materials, newMaterials);
    writeTable(device, *lights, update.lights, newLights);
    instanceCount = update.instanceCount;
    lightCount = update.lightCount;
    applyHierarchies(device, update, retired);
}

SceneTables DeviceScene::tables() const noexcept {
    return SceneTables{vertices.get(),  indices.get(),       meshes.get(),    instances.get(),
                       materials.get(), lights.get(),        instanceCount,   lightCount,
                       blasNodes.get(), blasTriangles.get(), blasRoots.get(), tlasNodes.get(),
                       tlasNodeCount};
}

} // namespace lr::render
