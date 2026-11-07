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

// A table of `size` elements is kept for `capacity` until it is more than four times as
// large, so that it does not shrink and grow back again and again.
constexpr auto kShrinkFactor = std::size_t{4};

bool keeps(std::size_t const size, std::size_t const capacity) noexcept {
    return size >= capacity and size <= kShrinkFactor * capacity;
}

// Ensures `table` holds `capacity` elements, keeping those of its contents that fit.
// Returns whether it was (re)created, in which case it has to be uploaded whole.
template <class T, class Layout>
bool reserveTable(std::unique_ptr<tc::TableResource<T, Layout>> &table, std::size_t const capacity,
                  RetireList &retired) {
    if (table != nullptr and keeps(table->size(), capacity)) {
        return false;
    }
    auto resized = std::make_unique<tc::TableResource<T, Layout>>(capacity);
    if (table != nullptr) {
        for (auto index = std::size_t{0}; index < std::min(table->size(), capacity); ++index) {
            resized->set(index, table->get(index));
        }
        retired.push_back(std::shared_ptr<void>{std::move(table)});
    }
    table = std::move(resized);
    return true;
}

template <class T, class Layout>
std::uint64_t bytesOf(std::unique_ptr<tc::TableResource<T, Layout>> const &table) noexcept {
    return table == nullptr ? 0 : tc::TableResource<T, Layout>::wordsFor(table->size()) * sizeof(std::uint32_t);
}

std::uint64_t bytesOf(std::unique_ptr<tc::BufferResource<tc::uint>> const &buffer) noexcept {
    return buffer == nullptr ? 0 : buffer->size() * sizeof(tc::uint);
}

bool reserveBuffer(std::unique_ptr<tc::BufferResource<tc::uint>> &buffer, std::size_t const capacity,
                   RetireList &retired) {
    if (buffer != nullptr and keeps(buffer->size(), capacity)) {
        return false;
    }
    auto resized = std::make_unique<tc::BufferResource<tc::uint>>(static_cast<std::int32_t>(capacity));
    if (buffer != nullptr) {
        std::copy_n(buffer->begin(), std::min(buffer->size(), capacity), resized->begin());
        retired.push_back(std::shared_ptr<void>{std::move(buffer)});
    }
    buffer = std::move(resized);
    return true;
}

// Moves the ranges of `moves` in turn, each element after the one before, as `rebase` has
// it: they go to lower places (GeometryMoves), so none overwrites what is still to move.
template <class T, class Layout, class Move, class Rebase>
void moveTable(tc::TableResource<T, Layout> &table, std::vector<Move> const &moves, Rebase const &rebase) {
    for (auto const &move : moves) {
        auto const &range = rangeOf(move);
        for (auto index = std::uint32_t{0}; index < range.count; ++index) {
            table.set(range.to + index, rebase(move, table.get(range.from + index)));
        }
    }
}

template <class T, class Layout>
void moveTable(tc::TableResource<T, Layout> &table, std::vector<RangeMove> const &moves) {
    moveTable(table, moves, [](RangeMove const &, T const &value) { return value; });
}

void moveBuffer(tc::BufferResource<tc::uint> &buffer, std::vector<RangeMove> const &moves) {
    for (auto const &move : moves) {
        auto const first = buffer.begin() + move.from;
        std::copy(first, first + move.count, buffer.begin() + move.to);
    }
}

// Uploads where `moves` went, unless the whole table is uploaded anyway.
template <class Move, class Upload>
void uploadMoved(std::vector<Move> const &moves, bool const whole, Upload const &upload) {
    if (whole) {
        return;
    }
    for (auto const &move : moves) {
        upload(rangeOf(move).to, rangeOf(move).count);
    }
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

GpuMaterial resolved(MaterialEncoding const &source, TexturePool const &textures) {
    auto const refOf = [&](std::size_t const index) {
        auto const id = source.textures[index];
        return id.isValid() ? textures.residentRef(id) : noTexture();
    };
    auto material = source.gpu;
    material.textures = tc::uvec4{refOf(0), refOf(1), refOf(2), refOf(3)};
    return material;
}

} // namespace

void DeviceScene::applyMaterials(tc::Device &device, SceneUpdate const &update, TexturePool &textures,
                                 bool const grown) {
    if (materialSources.size() != materials->size()) {
        materialSources.resize(materials->size());
    }
    for (auto const &write : update.materials) {
        materialSources[write.index] = write.value;
    }
    auto writes = std::vector<Indexed<GpuMaterial>>{};
    if (textures.takeResidencyChange() or grown) {
        for (auto index = std::size_t{0}; index < materialSources.size(); ++index) {
            if (materialSources[index]) {
                writes.push_back(Indexed<GpuMaterial>{static_cast<std::uint32_t>(index),
                                                      resolved(*materialSources[index], textures)});
            }
        }
        writeTable(device, *materials, writes, true);
        return;
    }
    for (auto const &write : update.materials) {
        writes.push_back(Indexed<GpuMaterial>{write.index, resolved(write.value, textures)});
    }
    writeTable(device, *materials, writes, false);
}

void DeviceScene::applyHierarchies(tc::Device &device, SceneUpdate const &update, RetireList &retired) {
    auto const &capacity = update.capacity;
    if (capacity.blasNodes == 0) {
        return;
    }
    auto const &moves = update.moves;
    if (blasNodes != nullptr) {
        moveTable(*blasNodes, moves.blasNodes, [](NodeMove const &move, GpuBvhNode node) {
            node.first = node.count == 0 ? node.first - move.nodes.from + move.nodes.to
                                         : node.first - move.fromTriangle + move.toTriangle;
            return node;
        });
    }
    if (blasTriangles != nullptr) {
        moveTable(*blasTriangles, moves.blasTriangles, [](TriangleMove const &move, GpuBvhTriangle triangle) {
            triangle.corners = triangle.corners - tc::uvec3{move.fromVertex} + tc::uvec3{move.toVertex};
            return triangle;
        });
    }
    auto const newBlasNodes = reserveTable(blasNodes, capacity.blasNodes, retired);
    auto const newBlasTriangles = reserveTable(blasTriangles, capacity.blasTriangles, retired);
    auto const newBlasRoots = reserveBuffer(blasRoots, capacity.meshes, retired);
    auto const newTlasNodes = reserveTable(tlasNodes, capacity.tlasNodes, retired);
    uploadMoved(moves.blasNodes, newBlasNodes,
                [&](std::uint32_t const first, std::uint32_t const count) { device.uploadTable(*blasNodes, first, count); });
    uploadMoved(moves.blasTriangles, newBlasTriangles, [&](std::uint32_t const first, std::uint32_t const count) {
        device.uploadTable(*blasTriangles, first, count);
    });
    writeTable(device, *blasNodes, update.blasNodes, newBlasNodes);
    writeTable(device, *blasTriangles, update.blasTriangles, newBlasTriangles);
    writeBuffer(device, *blasRoots, update.blasRoots, newBlasRoots);
    writeTable(device, *tlasNodes, update.tlasNodes, newTlasNodes);
    tlasNodeCount = update.tlasNodeCount;
}

void DeviceScene::apply(tc::Device &device, SceneUpdate const &update, TexturePool &textures, RetireList &retired) {
    auto const &capacity = update.capacity;
    auto const &moves = update.moves;
    if (vertices != nullptr) {
        moveTable(*vertices, moves.vertices);
        moveBuffer(*indices, moves.indices);
        moveTable(*clusters, moves.clusters);
    }
    auto const newVertices = reserveTable(vertices, capacity.vertices, retired);
    auto const newIndices = reserveBuffer(indices, capacity.indices, retired);
    auto const newClusters = reserveTable(clusters, capacity.clusters, retired);
    auto const newMeshes = reserveTable(meshes, capacity.meshes, retired);
    auto const newInstances = reserveTable(instances, capacity.instances, retired);
    auto const newMaterials = reserveTable(materials, capacity.materials, retired);
    auto const newLights = reserveTable(lights, capacity.lights, retired);

    for (auto const &geometry : update.geometry) {
        for (auto index = std::size_t{0}; index < geometry.vertices.size(); ++index) {
            vertices->set(geometry.firstVertex + index, geometry.vertices[index]);
        }
        std::copy(geometry.indices.begin(), geometry.indices.end(), indices->begin() + geometry.firstIndex);
        for (auto index = std::size_t{0}; index < geometry.clusters.size(); ++index) {
            clusters->set(geometry.firstCluster + index, geometry.clusters[index]);
        }
    }
    if (newVertices) {
        device.uploadTable(*vertices);
    }
    if (newIndices) {
        device.uploadBuffer(*indices);
    }
    if (newClusters) {
        device.uploadTable(*clusters);
    }
    uploadMoved(moves.vertices, newVertices,
                [&](std::uint32_t const first, std::uint32_t const count) { device.uploadTable(*vertices, first, count); });
    uploadMoved(moves.indices, newIndices,
                [&](std::uint32_t const first, std::uint32_t const count) { device.uploadBuffer(*indices, first, count); });
    uploadMoved(moves.clusters, newClusters,
                [&](std::uint32_t const first, std::uint32_t const count) { device.uploadTable(*clusters, first, count); });
    for (auto const &geometry : update.geometry) {
        if (not newVertices and not geometry.vertices.empty()) {
            device.uploadTable(*vertices, geometry.firstVertex, geometry.vertices.size());
        }
        if (not newIndices and not geometry.indices.empty()) {
            device.uploadBuffer(*indices, geometry.firstIndex, geometry.indices.size());
        }
        if (not newClusters and not geometry.clusters.empty()) {
            device.uploadTable(*clusters, geometry.firstCluster, geometry.clusters.size());
        }
    }

    writeTable(device, *meshes, update.meshes, newMeshes);
    writeTable(device, *instances, update.instances, newInstances);
    applyMaterials(device, update, textures, newMaterials);
    writeTable(device, *lights, update.lights, newLights);
    instanceCount = update.instanceCount;
    lightCount = update.lightCount;
    applyHierarchies(device, update, retired);
}

SceneTables DeviceScene::tables() const noexcept {
    return SceneTables{vertices.get(),  indices.get(),   meshes.get(),        clusters.get(),
                       instances.get(), materials.get(), lights.get(),        instanceCount,
                       lightCount,      blasNodes.get(), blasTriangles.get(), blasRoots.get(),
                       tlasNodes.get(), tlasNodeCount,   nullptr,             MaterialTextures{}};
}

SceneMemory DeviceScene::memory() const noexcept {
    auto memory = SceneMemory{};
    memory.geometry = bytesOf(vertices) + bytesOf(indices);
    memory.clusters = bytesOf(clusters);
    memory.objects = bytesOf(meshes) + bytesOf(instances) + bytesOf(materials) + bytesOf(lights);
    memory.hierarchies = bytesOf(blasNodes) + bytesOf(blasTriangles) + bytesOf(blasRoots) + bytesOf(tlasNodes);
    return memory;
}

} // namespace lr::render
