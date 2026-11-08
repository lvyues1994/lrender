#include "RenderWorld.h"

#include "Clusters.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace lr::render {

namespace {

std::uint32_t roundUpToPowerOfTwo(std::uint32_t const value) noexcept {
    auto result = std::uint32_t{1};
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

void validateMesh(core::MeshUpsert const &upsert) {
    auto const fail = [&upsert](char const *what) {
        throw std::invalid_argument{"lr::render: mesh " + std::to_string(upsert.id.value) + " " + what};
    };
    if (upsert.mesh == nullptr) {
        fail("has no data");
    }
    auto const &mesh = *upsert.mesh;
    if (mesh.normals.size() != mesh.positions.size()) {
        fail("needs one normal per position");
    }
    if (not mesh.uvs.empty() and mesh.uvs.size() != mesh.positions.size()) {
        fail("needs one uv per position, or none");
    }
    if (mesh.indices.size() % 3 != 0) {
        fail("has an index count that is not a multiple of 3");
    }
    auto const vertexCount = mesh.positions.size();
    if (std::any_of(mesh.indices.begin(), mesh.indices.end(),
                    [vertexCount](std::uint32_t const index) { return index >= vertexCount; })) {
        fail("has an index past its last vertex");
    }
}

void validateImage(core::TextureUpsert const &upsert) {
    auto const fail = [&upsert](char const *what) {
        throw std::invalid_argument{"lr::render: texture " + std::to_string(upsert.id.value) + " " + what};
    };
    if (upsert.image == nullptr) {
        fail("has no image");
    }
    auto const &image = *upsert.image;
    if (image.extent.x <= 0 or image.extent.y <= 0) {
        fail("has an empty extent");
    }
    if (image.compressed and static_cast<std::uint32_t>(*image.compressed) >= tc::kCompressedFormatCount) {
        fail("has an unknown compressed format");
    }
    if (image.levels.empty() or image.levels.size() > core::fullLevelCount(image.extent)) {
        fail("needs between one level and a full mip chain");
    }
    for (auto level = std::uint32_t{0}; level < image.levels.size(); ++level) {
        if (image.levels[level].size() != core::levelBytes(image, level)) {
            fail("has a level whose size does not match its extent and format");
        }
    }
}

GpuLight encodeLight(core::LightDesc const &desc) {
    struct Encoder {
        GpuLight operator()(core::DirectionalLight const &light) const {
            return GpuLight{lightDirectional(), light.color * light.illuminance, tc::normalize(light.direction), 0.0f};
        }
        GpuLight operator()(core::PointLight const &light) const {
            return GpuLight{lightPoint(), light.color * light.intensity, light.position, light.range};
        }
    };
    return std::visit(Encoder{}, desc);
}

tc::uint materialFlags(core::MaterialDesc const &desc) noexcept {
    auto const alpha = desc.alphaMode == core::AlphaMode::Mask    ? materialAlphaMask()
                     : desc.alphaMode == core::AlphaMode::Blend ? materialAlphaBlend()
                                                                  : 0u;
    return (desc.doubleSided ? materialDoubleSided() : 0u) | alpha;
}

core::TextureId textureOrNone(std::optional<core::TextureId> const &texture) noexcept {
    return texture.value_or(core::TextureId{});
}

MaterialEncoding encodeMaterial(core::MaterialDesc const &desc) {
    auto const &uv = desc.uvTransform;
    auto const cosine = std::cos(uv.rotation);
    auto const sine = std::sin(uv.rotation);
    auto gpu = GpuMaterial{};
    gpu.baseColor = desc.baseColor;
    gpu.emissive = desc.emissive;
    gpu.metallic = desc.metallic;
    gpu.roughness = desc.roughness;
    gpu.alphaCutoff = desc.alphaCutoff;
    gpu.flags = materialFlags(desc);
    gpu.normalScale = desc.normalScale;
    gpu.textures = tc::uvec4{noTexture()};
    gpu.uvRow0 = core::vec3{cosine * uv.scale.x, -sine * uv.scale.y, uv.offset.x};
    gpu.uvRow1 = core::vec3{sine * uv.scale.x, cosine * uv.scale.y, uv.offset.y};
    gpu.model = desc.model.value;
    gpu.parameters = desc.modelParameters;
    return MaterialEncoding{gpu,
                            {textureOrNone(desc.baseColorTexture), textureOrNone(desc.metallicRoughnessTexture),
                             textureOrNone(desc.emissiveTexture), textureOrNone(desc.normalTexture)}};
}

bool changesScene(core::RenderWorldDelta const &delta) noexcept {
    auto const &removed = delta.removed;
    return not delta.meshes.empty() or not delta.textures.empty() or not delta.materials.empty()
        or not delta.instances.empty() or not delta.lights.empty() or delta.environment.has_value()
        or not removed.meshes.empty() or not removed.textures.empty() or not removed.materials.empty()
        or not removed.instances.empty() or not removed.lights.empty();
}

GeometryUpload encodeGeometry(core::MeshData const &data, std::uint32_t const firstVertex,
                              std::uint32_t const firstIndex) {
    auto geometry = GeometryUpload{};
    geometry.firstVertex = firstVertex;
    geometry.firstIndex = firstIndex;
    geometry.vertices.reserve(data.positions.size());
    for (auto index = std::size_t{0}; index < data.positions.size(); ++index) {
        auto const uv = data.uvs.empty() ? core::vec2{0.0f} : data.uvs[index];
        geometry.vertices.push_back(GpuVertex{data.positions[index], tc::normalize(data.normals[index]), uv});
    }
    geometry.indices.assign(data.indices.begin(), data.indices.end());
    return geometry;
}

template <class Upserts>
auto idsOf(Upserts const &upserts) {
    auto ids = std::unordered_set<decltype(upserts.front().id)>{};
    for (auto const &upsert : upserts) {
        ids.insert(upsert.id);
    }
    return ids;
}

// Whether the world (a map or set by id) has `id`, or the delta upserts it.
template <class Id, class World>
bool known(Id const id, World const &world, std::unordered_set<Id> const &upserted) {
    return world.count(id) != 0 or upserted.count(id) != 0;
}

// Whether the world has each of `ids` or `upserts` upserts it; the upserted ids are only
// gathered for one the world does not have.
template <class Ids, class World, class Upserts>
bool allKnown(Ids const &ids, World const &world, Upserts const &upserts) {
    auto upserted = std::optional<decltype(idsOf(upserts))>{};
    for (auto const id : ids) {
        if (world.count(id) != 0) {
            continue;
        }
        if (not upserted) {
            upserted = idsOf(upserts);
        }
        if (upserted->count(id) == 0) {
            return false;
        }
    }
    return true;
}

std::vector<core::Aabb> triangleBounds(core::MeshData const &mesh) {
    auto bounds = std::vector<core::Aabb>{};
    bounds.reserve(mesh.indices.size() / 3);
    for (auto first = std::size_t{0}; first + 2 < mesh.indices.size(); first += 3) {
        auto const &a = mesh.positions[mesh.indices[first]];
        auto const &b = mesh.positions[mesh.indices[first + 1]];
        auto const &c = mesh.positions[mesh.indices[first + 2]];
        bounds.push_back(core::Aabb{tc::min(tc::min(a, b), c), tc::max(tc::max(a, b), c)});
    }
    return bounds;
}

} // namespace

void RenderWorld::validate(core::RenderWorldDelta const &delta) const {
    for (auto const &upsert : delta.meshes) {
        validateMesh(upsert);
    }
    for (auto const &upsert : delta.textures) {
        validateImage(upsert);
    }
    for (auto const &upsert : delta.materials) {
        if (upsert.material.model.value >= materialModels) {
            throw std::invalid_argument{"lr::render: material " + std::to_string(upsert.id.value)
                                        + " uses material model " + std::to_string(upsert.material.model.value)
                                        + ", which the renderer does not have"};
        }
    }
    auto const upsertedMeshes = idsOf(delta.meshes);
    auto const upsertedMaterials = idsOf(delta.materials);
    for (auto const &upsert : delta.instances) {
        if (not known(upsert.mesh, meshes, upsertedMeshes) or not known(upsert.material, materials, upsertedMaterials)) {
            throw std::invalid_argument{"lr::render: instance " + std::to_string(upsert.id.value)
                                        + " refers to a mesh or material the world does not have"};
        }
    }

    auto const &removed = delta.removed;
    if (not allKnown(removed.meshes, meshes, delta.meshes) or not allKnown(removed.textures, textureIds, delta.textures)
        or not allKnown(removed.materials, materials, delta.materials)
        or not allKnown(removed.instances, instanceIndex, delta.instances)
        or not allKnown(removed.lights, lightIndex, delta.lights)) {
        throw std::invalid_argument{"lr::render: the delta removes an object the world does not have"};
    }

    if (removed.meshes.empty() and removed.materials.empty()) {
        return;
    }
    // A mesh or material may only go once no remaining instance uses it.
    auto removedInstances = std::unordered_set<core::InstanceId>(removed.instances.begin(), removed.instances.end());
    auto usedMeshes = std::unordered_set<core::MeshId>{};
    auto usedMaterials = std::unordered_set<core::MaterialId>{};
    auto const use = [&](core::InstanceId const id, core::MeshId const mesh, core::MaterialId const material) {
        if (removedInstances.count(id) == 0) {
            usedMeshes.insert(mesh);
            usedMaterials.insert(material);
        }
    };
    auto upserted = std::unordered_set<core::InstanceId>{};
    for (auto const &upsert : delta.instances) {
        use(upsert.id, upsert.mesh, upsert.material);
        upserted.insert(upsert.id);
    }
    for (auto const &record : instances) {
        if (upserted.count(record.id) == 0) {
            use(record.id, record.mesh, record.material);
        }
    }
    for (auto const id : removed.meshes) {
        if (usedMeshes.count(id) != 0) {
            throw std::invalid_argument{"lr::render: mesh " + std::to_string(id.value) + " is removed while in use"};
        }
    }
    for (auto const id : removed.materials) {
        if (usedMaterials.count(id) != 0) {
            throw std::invalid_argument{"lr::render: material " + std::to_string(id.value) + " is removed while in use"};
        }
    }
}

std::uint32_t RenderWorld::takeSlot(std::vector<std::uint32_t> &freeSlots, std::uint32_t &end) {
    if (not freeSlots.empty()) {
        auto const slot = freeSlots.back();
        freeSlots.pop_back();
        return slot;
    }
    return end++;
}

RenderWorld::RenderWorld(TextureLimits const &limits)
    : ownParallel(makeSerialFor()), ownConversions(makeImmediateConversions()), parallel(ownParallel.get()),
      textures(limits, *ownConversions) {}

RenderWorld::RenderWorld(TextureLimits const &limits, ParallelFor &parallel_, ImageConversions &conversions,
                         ConversionDelivery const delivery)
    : parallel(&parallel_), textures(limits, conversions, delivery) {}

// Each mesh's vertices, indices and clusters take ranges of their tables, after a replaced
// mesh gives its own back; the encoding and the bounds of each are an item of `parallel`.
// Of the upserts of one id, the last one counts.
void RenderWorld::upsertMeshes(std::vector<core::MeshUpsert> const &all, SceneUpdate &update) {
    auto upserts = std::vector<core::MeshUpsert const *>{};
    auto last = std::unordered_map<core::MeshId, std::size_t>{};
    for (auto index = std::size_t{0}; index < all.size(); ++index) {
        last[all[index].id] = index;
    }
    for (auto index = std::size_t{0}; index < all.size(); ++index) {
        if (last.at(all[index].id) == index) {
            upserts.push_back(&all[index]);
        }
    }
    auto geometry = std::vector<GeometryUpload>(upserts.size());
    auto bounds = std::vector<core::Aabb>(upserts.size());
    auto clusters = std::vector<std::vector<GpuCluster>>(upserts.size());
    auto firsts = std::vector<std::pair<std::uint32_t, std::uint32_t>>{};
    firsts.reserve(upserts.size());
    for (auto const *upsert : upserts) {
        auto const found = meshes.find(upsert->id);
        if (found != meshes.end()) {
            releaseGeometry(found->second);
            releaseHierarchy(found->second);
        }
        firsts.emplace_back(vertexRanges.allocate(static_cast<std::uint32_t>(upsert->mesh->positions.size())),
                            indexRanges.allocate(static_cast<std::uint32_t>(upsert->mesh->indices.size())));
    }
    parallel->run(upserts.size(), [&](std::size_t const index) {
        auto const &data = *upserts[index]->mesh;
        geometry[index] = encodeGeometry(data, firsts[index].first, firsts[index].second);
        bounds[index] = core::boundsOf(data);
        clusters[index] = buildClusters(data);
    });
    for (auto index = std::size_t{0}; index < upserts.size(); ++index) {
        auto const &upsert = *upserts[index];
        auto const found = meshes.find(upsert.id);
        auto const slot = found != meshes.end() ? found->second.slot : takeSlot(freeMeshSlots, meshSlotEnd);
        auto const clusterCount = static_cast<std::uint32_t>(clusters[index].size());
        if (found != meshes.end()) {
            auto const &old = found->second.gpu;
            if (old.clusterCount != clusterCount or old.triangleCount * 3 != upsert.mesh->indices.size()
                or not(old.boundsMin == bounds[index].min) or not(old.boundsMax == bounds[index].max)) {
                markAllInstances();
            }
        }
        auto const firstCluster = clusterRanges.allocate(clusterCount);
        auto const gpu = GpuMesh{geometry[index].firstVertex,
                                 geometry[index].firstIndex,
                                 static_cast<std::uint32_t>(upsert.mesh->indices.size() / 3),
                                 bounds[index].min,
                                 bounds[index].max,
                                 static_cast<std::uint32_t>(geometry[index].vertices.size()),
                                 firstCluster,
                                 clusterCount};
        geometry[index].firstCluster = firstCluster;
        geometry[index].clusters = clusters[index];
        meshes[upsert.id] = MeshRecord{slot, gpu, upsert.mesh, noBvhNode(), core::Aabb{}, std::move(clusters[index])};
        update.geometry.push_back(std::move(geometry[index]));
        update.meshes.push_back(Indexed<GpuMesh>{slot, gpu});
    }
}

void RenderWorld::releaseGeometry(MeshRecord const &record) {
    vertexRanges.release(record.gpu.firstVertex, record.gpu.vertexCount);
    indexRanges.release(record.gpu.firstIndex, 3 * record.gpu.triangleCount);
    clusterRanges.release(record.gpu.firstCluster, record.gpu.clusterCount);
}

void RenderWorld::releaseHierarchy(MeshRecord &record) {
    if (record.blasRoot != noBvhNode()) {
        blasNodeRanges.release(record.blasRoot, record.blasNodeCount);
        blasTriangleRanges.release(record.blasTriangleFirst, record.blasTriangleCount);
    }
    record.blasRoot = noBvhNode();
    record.blasNodeCount = 0;
    record.blasTriangleCount = 0;
}

bool RenderWorld::fragmented() const noexcept {
    auto const wasteful = [](RangeAllocator const &ranges) { return ranges.wasted() > ranges.used(); };
    return wasteful(vertexRanges) or wasteful(indexRanges) or wasteful(clusterRanges)
        or wasteful(blasNodeRanges) or wasteful(blasTriangleRanges);
}

namespace {

// Places `count(record)` elements of each record one after the other from 0, in the order
// their `first(record)` has them now: the new firsts, and the moves to them.
template <class Record, class First, class Count>
std::vector<std::uint32_t> pack(std::vector<Record *> const &records, First const &first, Count const &count,
                                RangeAllocator &ranges, std::vector<RangeMove> &moves) {
    auto order = std::vector<std::size_t>(records.size());
    for (auto index = std::size_t{0}; index < order.size(); ++index) {
        order[index] = index;
    }
    std::sort(order.begin(), order.end(),
              [&](std::size_t const a, std::size_t const b) { return first(*records[a]) < first(*records[b]); });
    auto placed = std::vector<std::uint32_t>(records.size(), 0);
    auto next = std::uint32_t{0};
    for (auto const index : order) {
        auto const elements = count(*records[index]);
        if (elements == 0) {
            continue;
        }
        placed[index] = next;
        moves.push_back(RangeMove{first(*records[index]), next, elements});
        next += elements;
    }
    ranges.reset(next);
    return placed;
}

} // namespace

// Every mesh's geometry and hierarchy packed to the start of their tables, in the order they
// lie there.
void RenderWorld::compactGeometry(SceneUpdate &update) {
    auto records = std::vector<MeshRecord *>{};
    for (auto &[id, record] : meshes) {
        records.push_back(&record);
    }
    auto vertexMoves = std::vector<RangeMove>{};
    auto const vertices = pack(
        records, [](MeshRecord const &r) { return r.gpu.firstVertex; },
        [](MeshRecord const &r) { return r.gpu.vertexCount; }, vertexRanges, vertexMoves);
    auto indexMoves = std::vector<RangeMove>{};
    auto const indices = pack(
        records, [](MeshRecord const &r) { return r.gpu.firstIndex; },
        [](MeshRecord const &r) { return 3 * r.gpu.triangleCount; }, indexRanges, indexMoves);
    auto clusterMoves = std::vector<RangeMove>{};
    auto const firstClusters = pack(
        records, [](MeshRecord const &r) { return r.gpu.firstCluster; },
        [](MeshRecord const &r) { return r.gpu.clusterCount; }, clusterRanges, clusterMoves);
    auto nodeMoves = std::vector<RangeMove>{};
    auto const roots = pack(
        records, [](MeshRecord const &r) { return r.blasRoot; }, [](MeshRecord const &r) { return r.blasNodeCount; },
        blasNodeRanges, nodeMoves);
    auto triangleMoves = std::vector<RangeMove>{};
    auto const triangles = pack(
        records, [](MeshRecord const &r) { return r.blasTriangleFirst; },
        [](MeshRecord const &r) { return r.blasTriangleCount; }, blasTriangleRanges, triangleMoves);

    auto const moved = [](RangeMove const &move) { return move.from != move.to; };
    std::copy_if(vertexMoves.begin(), vertexMoves.end(), std::back_inserter(update.moves.vertices), moved);
    std::copy_if(indexMoves.begin(), indexMoves.end(), std::back_inserter(update.moves.indices), moved);
    std::copy_if(clusterMoves.begin(), clusterMoves.end(), std::back_inserter(update.moves.clusters), moved);
    auto recordOfNode = std::unordered_map<std::uint32_t, std::size_t>{};
    auto recordOfTriangle = std::unordered_map<std::uint32_t, std::size_t>{};
    for (auto index = std::size_t{0}; index < records.size(); ++index) {
        if (records[index]->blasNodeCount != 0) {
            recordOfNode[records[index]->blasRoot] = index;
            recordOfTriangle[records[index]->blasTriangleFirst] = index;
        }
    }
    for (auto const &move : nodeMoves) {
        auto const &record = *records[recordOfNode.at(move.from)];
        auto const toTriangle = triangles[recordOfNode.at(move.from)];
        if (move.from != move.to or record.blasTriangleFirst != toTriangle) {
            update.moves.blasNodes.push_back(NodeMove{move, record.blasTriangleFirst, toTriangle});
        }
    }
    for (auto const &move : triangleMoves) {
        auto const index = recordOfTriangle.at(move.from);
        auto const &record = *records[index];
        if (move.from != move.to or record.gpu.firstVertex != vertices[index]) {
            update.moves.blasTriangles.push_back(TriangleMove{move, record.gpu.firstVertex, vertices[index]});
        }
    }

    for (auto index = std::size_t{0}; index < records.size(); ++index) {
        auto &record = *records[index];
        auto gpu = record.gpu;
        gpu.firstVertex = record.gpu.vertexCount != 0 ? vertices[index] : 0u;
        gpu.firstIndex = record.gpu.triangleCount != 0 ? indices[index] : 0u;
        gpu.firstCluster = record.gpu.clusterCount != 0 ? firstClusters[index] : 0u;
        if (gpu.firstVertex != record.gpu.firstVertex or gpu.firstIndex != record.gpu.firstIndex
            or gpu.firstCluster != record.gpu.firstCluster) {
            record.gpu = gpu;
            update.meshes.push_back(Indexed<GpuMesh>{record.slot, gpu});
        }
        if (record.blasNodeCount != 0) {
            record.blasTriangleFirst = triangles[index];
            if (record.blasRoot != roots[index]) {
                record.blasRoot = roots[index];
                update.blasRoots.push_back(Indexed<tc::uint>{record.slot, record.blasRoot});
            }
        }
    }
}

void RenderWorld::upsertMaterial(core::MaterialUpsert const &upsert, SceneUpdate &update) {
    auto const found = materials.find(upsert.id);
    auto const blends = [](core::MaterialDesc const &material) { return material.alphaMode == core::AlphaMode::Blend; };
    if (found != materials.end() and blends(found->second.desc) != blends(upsert.material)) {
        markAllInstances();
    }
    auto const slot = found != materials.end() ? found->second.slot : takeSlot(freeMaterialSlots, materialSlotEnd);
    materials[upsert.id] = MaterialRecord{slot, upsert.material};
    update.materials.push_back(Indexed<MaterialEncoding>{slot, encodeMaterial(upsert.material)});
}

void RenderWorld::applyTextures(core::RenderWorldDelta const &delta, SceneUpdate &update) {
    for (auto const &upsert : delta.textures) {
        textures.place(upsert, update);
        textureIds.insert(upsert.id);
        lostTextures.erase(upsert.id);
    }
    for (auto const id : delta.removed.textures) {
        textures.remove(id);
        textureIds.erase(id);
        lostTextures.erase(id);
        update.removedTextures.push_back(id);
    }
}

void RenderWorld::loseTextures() {
    for (auto const id : textures.placed()) {
        lostTextures.insert(id);
    }
    textures.reset();
}

std::vector<core::TextureId> RenderWorld::takeLostTextures() {
    auto lost = std::vector<core::TextureId>(lostTextures.begin(), lostTextures.end());
    std::sort(lost.begin(), lost.end());
    lostTextures.clear();
    return lost;
}

std::optional<std::uint32_t> RenderWorld::meshSlot(core::MeshId const id) const {
    auto const found = meshes.find(id);
    return found != meshes.end() ? std::optional<std::uint32_t>{found->second.slot} : std::nullopt;
}

std::optional<ShadowLight> RenderWorld::shadowLight() const {
    for (auto index = std::uint32_t{0}; index < lightCount(); ++index) {
        auto const *directional = std::get_if<core::DirectionalLight>(&lights[index].desc);
        if (directional != nullptr and directional->castsShadows) {
            return ShadowLight{index, tc::normalize(directional->direction)};
        }
    }
    return std::nullopt;
}

namespace {

constexpr auto kInstancesPerChunk = std::uint32_t{1024};

} // namespace

void RenderWorld::markInstance(std::uint32_t const index) {
    auto const chunk = index / kInstancesPerChunk;
    if (chunk < instanceChunks.size()) {
        instanceChunks[chunk].dirty = true;
    }
}

void RenderWorld::markAllInstances() {
    for (auto &chunk : instanceChunks) {
        chunk.dirty = true;
    }
}

// The changed chunks are added up again as items of `parallel`, each on its own, so the
// sums are those of adding up every instance in order.
void RenderWorld::updateSummaries() {
    instanceChunks.resize((instanceCount() + kInstancesPerChunk - 1) / kInstancesPerChunk);
    auto dirty = std::vector<std::uint32_t>{};
    for (auto index = std::uint32_t{0}; index < instanceChunks.size(); ++index) {
        if (instanceChunks[index].dirty) {
            dirty.push_back(index);
        }
    }
    parallel->run(dirty.size(), [&](std::size_t const item) {
        // Added up here and stored once: neighbouring chunks share cache lines.
        auto sums = InstanceChunk{};
        sums.dirty = false;
        auto const first = dirty[item] * kInstancesPerChunk;
        for (auto at = first; at < std::min(first + kInstancesPerChunk, instanceCount()); ++at) {
            auto const &record = instances[at];
            if (materials.at(record.material).desc.alphaMode == core::AlphaMode::Blend) {
                ++sums.transparent;
            }
            auto const &mesh = meshes.at(record.mesh).gpu;
            sums.clusters += mesh.clusterCount;
            if (mesh.triangleCount != 0) {
                auto const box = core::transformed(core::Aabb{mesh.boundsMin, mesh.boundsMax}, record.objectToWorld);
                sums.bounds.min = tc::min(sums.bounds.min, box.min);
                sums.bounds.max = tc::max(sums.bounds.max, box.max);
            }
        }
        instanceChunks[dirty[item]] = sums;
    });
    transparentInstances = 0;
    clusterInstances = 0;
    sceneBounds = core::Aabb{core::vec3{INFINITY}, core::vec3{-INFINITY}};
    for (auto const &chunk : instanceChunks) {
        transparentInstances += chunk.transparent;
        clusterInstances += chunk.clusters;
        sceneBounds.min = tc::min(sceneBounds.min, chunk.bounds.min);
        sceneBounds.max = tc::max(sceneBounds.max, chunk.bounds.max);
    }
}

GpuInstance RenderWorld::encodeInstance(std::uint32_t const index) const {
    auto const &record = instances[index];
    return GpuInstance{record.objectToWorld, record.worldToObject, meshes.at(record.mesh).slot,
                       materials.at(record.material).slot, record.previousObjectToWorld, record.movedFrame};
}

void RenderWorld::writeInstance(std::uint32_t const index, SceneUpdate &update) const {
    update.instances.push_back(Indexed<GpuInstance>{index, encodeInstance(index)});
}

// A new instance was nowhere before; one already there was where the last frame had it.
void RenderWorld::upsertInstance(core::InstanceUpsert const &upsert, SceneUpdate &update) {
    auto const found = instanceIndex.find(upsert.id);
    if (not found) {
        auto const index = instanceCount();
        instances.push_back(InstanceRecord{upsert.id, upsert.mesh, upsert.material, upsert.objectToWorld,
                                           upsert.objectToWorld, tc::inverse(upsert.objectToWorld), frames});
        instanceIndex.set(upsert.id, index);
        addedInstances = true;
        markInstance(index);
        writeInstance(index, update);
        return;
    }
    auto &record = instances[*found];
    // Upserted again in the same frame, it was where the frame before had it.
    auto const before = record.movedFrame == frames ? record.previousObjectToWorld : record.objectToWorld;
    record = InstanceRecord{upsert.id,   upsert.mesh, upsert.material, upsert.objectToWorld,
                            before,      tc::inverse(upsert.objectToWorld), frames};
    markInstance(*found);
    writeInstance(*found, update);
}

// New instances go after the others, in turn; those already there move as items of
// `parallel`, each into its own place, or in turn when the delta upserts one of them twice.
void RenderWorld::upsertInstances(std::vector<core::InstanceUpsert> const &upserts, SceneUpdate &update) {
    upsertFrames.resize(instances.size(), 0);
    auto moves = std::vector<std::pair<core::InstanceUpsert const *, std::uint32_t>>{};
    auto twice = false;
    for (auto const &upsert : upserts) {
        auto const found = instanceIndex.find(upsert.id);
        if (not found) {
            upsertInstance(upsert, update);
            upsertFrames.push_back(frames);
            continue;
        }
        twice = twice or upsertFrames[*found] == frames;
        upsertFrames[*found] = frames;
        moves.emplace_back(&upsert, *found);
    }
    if (twice) {
        for (auto const &move : moves) {
            upsertInstance(*move.first, update);
        }
        return;
    }
    auto const first = update.instances.size();
    update.instances.resize(first + moves.size());
    constexpr auto kMovesPerItem = std::size_t{512};
    parallel->run((moves.size() + kMovesPerItem - 1) / kMovesPerItem, [&](std::size_t const block) {
        for (auto item = block * kMovesPerItem; item < std::min(moves.size(), (block + 1) * kMovesPerItem); ++item) {
            auto const &[upsert, index] = moves[item];
            auto &record = instances[index];
            record = InstanceRecord{upsert->id,           upsert->mesh,         upsert->material,
                                    upsert->objectToWorld, record.objectToWorld, tc::inverse(upsert->objectToWorld),
                                    frames};
            update.instances[first + item] = Indexed<GpuInstance>{index, encodeInstance(index)};
        }
    });
    for (auto const &move : moves) {
        markInstance(move.second);
    }
}

void RenderWorld::writeLight(std::uint32_t const index, SceneUpdate &update) const {
    update.lights.push_back(Indexed<GpuLight>{index, encodeLight(lights[index].desc)});
}

void RenderWorld::removeInstance(core::InstanceId const id, SceneUpdate &update) {
    auto const index = instanceIndex.at(id);
    auto const last = static_cast<std::uint32_t>(instances.size() - 1);
    instanceIndex.erase(id);
    if (index != last) {
        instances[index] = instances[last];
        instanceIndex.set(instances[index].id, index);
        writeInstance(index, update);
    }
    instances.pop_back();
    markInstance(index);
    markInstance(last);
}

void RenderWorld::removeLight(core::LightId const id, SceneUpdate &update) {
    auto const index = lightIndex.at(id);
    auto const last = static_cast<std::uint32_t>(lights.size() - 1);
    lightIndex.erase(id);
    if (index != last) {
        lights[index] = lights[last];
        lightIndex[lights[index].id] = index;
        writeLight(index, update);
    }
    lights.pop_back();
}

void RenderWorld::buildBlases(std::vector<MeshRecord *> const &records, SceneUpdate &update) {
    auto built = std::vector<Bvh>(records.size());
    parallel->run(records.size(), [&](std::size_t const index) {
        built[index] = buildBvh(triangleBounds(*records[index]->data), BvhBuildConfig{4});
    });
    for (auto index = std::size_t{0}; index < records.size(); ++index) {
        placeBlas(*records[index], built[index], update);
    }
}

void RenderWorld::placeBlas(MeshRecord &record, Bvh const &bvh, SceneUpdate &update) {
    auto const &data = *record.data;
    releaseHierarchy(record);
    auto const nodeCount = static_cast<std::uint32_t>(bvh.nodes.size());
    auto const triangleCount = static_cast<std::uint32_t>(bvh.order.size());
    auto const firstNode = blasNodeRanges.allocate(nodeCount);
    auto const firstTriangle = blasTriangleRanges.allocate(triangleCount);
    record.blasRoot = bvh.nodes.empty() ? noBvhNode() : firstNode;
    record.blasBounds = bvh.nodes.empty() ? core::Aabb{} : boundsOf(bvh.nodes[0]);
    record.blasNodeCount = bvh.nodes.empty() ? 0u : nodeCount;
    record.blasTriangleFirst = firstTriangle;
    record.blasTriangleCount = bvh.nodes.empty() ? 0u : triangleCount;
    update.blasRoots.push_back(Indexed<tc::uint>{record.slot, record.blasRoot});
    for (auto index = std::size_t{0}; index < bvh.nodes.size(); ++index) {
        auto node = bvh.nodes[index];
        node.first += node.count == 0 ? firstNode : firstTriangle;
        update.blasNodes.push_back(Indexed<GpuBvhNode>{firstNode + static_cast<std::uint32_t>(index), node});
    }
    auto const firstVertex = record.gpu.firstVertex;
    for (auto index = std::size_t{0}; index < bvh.order.size(); ++index) {
        auto const primitive = bvh.order[index];
        auto const first = std::size_t{primitive} * 3;
        auto const corners = tc::uvec3{firstVertex + data.indices[first], firstVertex + data.indices[first + 1],
                                       firstVertex + data.indices[first + 2]};
        update.blasTriangles.push_back(Indexed<GpuBvhTriangle>{firstTriangle + static_cast<std::uint32_t>(index),
                                                               GpuBvhTriangle{corners, primitive}});
    }
}

// Instances of meshes without triangles are left out.
void RenderWorld::buildTlas(SceneUpdate &update) {
    auto bounds = std::vector<core::Aabb>{};
    auto members = std::vector<std::uint32_t>{};
    for (auto index = std::uint32_t{0}; index < instanceCount(); ++index) {
        auto const &mesh = meshes.at(instances[index].mesh);
        if (mesh.blasRoot != noBvhNode()) {
            bounds.push_back(core::transformed(mesh.blasBounds, instances[index].objectToWorld));
            members.push_back(index);
        }
    }
    auto const bvh = buildBvh(bounds, BvhBuildConfig{1});
    update.tlasNodes.clear();
    tlas = bvh.nodes;
    tlasParents.assign(tlas.size(), noBvhNode());
    tlasLeaves.assign(instanceCount(), noBvhNode());
    for (auto index = std::uint32_t{0}; index < tlas.size(); ++index) {
        auto &node = tlas[index];
        if (node.count != 0) {
            node.first = members[bvh.order[node.first]];
            tlasLeaves[node.first] = index;
        } else {
            tlasParents[node.first] = index;
            tlasParents[node.first + 1] = index;
        }
        update.tlasNodes.push_back(Indexed<GpuBvhNode>{index, node});
    }
    tlasNodeCount = static_cast<std::uint32_t>(tlas.size());
}

bool RenderWorld::refitTlas(core::RenderWorldDelta const &delta, SceneUpdate &update) {
    for (auto const &upsert : delta.instances) {
        auto const index = instanceIndex.at(upsert.id);
        if ((tlasLeaves[index] == noBvhNode()) != (meshes.at(instances[index].mesh).blasRoot == noBvhNode())) {
            return false;
        }
    }
    auto written = std::vector<std::uint32_t>{};
    auto const rewrite = [&](std::uint32_t const node, core::Aabb const &box) {
        auto &held = tlas[node];
        if (held.boundsMin == box.min and held.boundsMax == box.max) {
            return false;
        }
        held.boundsMin = box.min;
        held.boundsMax = box.max;
        written.push_back(node);
        return true;
    };
    for (auto const &upsert : delta.instances) {
        auto const index = instanceIndex.at(upsert.id);
        auto node = tlasLeaves[index];
        if (node == noBvhNode()) {
            continue;
        }
        auto const &record = instances[index];
        auto changed = rewrite(node, core::transformed(meshes.at(record.mesh).blasBounds, record.objectToWorld));
        while (changed and tlasParents[node] != noBvhNode()) {
            node = tlasParents[node];
            auto const first = tlas[node].first;
            auto const children = core::Aabb{tc::min(tlas[first].boundsMin, tlas[first + 1].boundsMin),
                                             tc::max(tlas[first].boundsMax, tlas[first + 1].boundsMax)};
            changed = rewrite(node, children);
        }
    }
    std::sort(written.begin(), written.end());
    written.erase(std::unique(written.begin(), written.end()), written.end());
    for (auto const node : written) {
        update.tlasNodes.push_back(Indexed<GpuBvhNode>{node, tlas[node]});
    }
    return true;
}

void RenderWorld::updateHierarchies(core::RenderWorldDelta const &delta, SceneUpdate &update) {
    auto records = std::vector<MeshRecord *>{};
    for (auto const &upsert : delta.meshes) {
        auto const found = meshes.find(upsert.id);
        if (found != meshes.end()) {
            records.push_back(&found->second);
        }
    }
    buildBlases(records, update);
    if (not delta.meshes.empty() or addedInstances or not delta.removed.instances.empty()) {
        buildTlas(update);
    } else if (not delta.instances.empty() and not refitTlas(delta, update)) {
        buildTlas(update);
    }
}

SceneUpdate RenderWorld::fullUpdate() {
    auto update = SceneUpdate{};
    for (auto const &[id, record] : meshes) {
        auto geometry = encodeGeometry(*record.data, record.gpu.firstVertex, record.gpu.firstIndex);
        geometry.firstCluster = record.gpu.firstCluster;
        geometry.clusters = record.clusters;
        update.geometry.push_back(std::move(geometry));
        update.meshes.push_back(Indexed<GpuMesh>{record.slot, record.gpu});
    }
    for (auto const &[id, record] : materials) {
        update.materials.push_back(Indexed<MaterialEncoding>{record.slot, encodeMaterial(record.desc)});
    }
    for (auto index = std::uint32_t{0}; index < instanceCount(); ++index) {
        writeInstance(index, update);
    }
    for (auto index = std::uint32_t{0}; index < lightCount(); ++index) {
        writeLight(index, update);
    }
    if (hierarchies) {
        hierarchies = false;
        blasNodeRanges.clear();
        blasTriangleRanges.clear();
        for (auto &[id, record] : meshes) {
            record.blasRoot = noBvhNode();
            record.blasNodeCount = 0;
            record.blasTriangleCount = 0;
        }
        keepHierarchies(update);
    }
    update.capacity = capacity();
    update.instanceCount = instanceCount();
    update.lightCount = lightCount();
    update.tlasNodeCount = tlasNodeCount;
    return update;
}

void RenderWorld::keepHierarchies(SceneUpdate &update) {
    if (hierarchies) {
        return;
    }
    hierarchies = true;
    auto records = std::vector<MeshRecord *>{};
    for (auto &[id, record] : meshes) {
        records.push_back(&record);
    }
    std::sort(records.begin(), records.end(), [](MeshRecord const *a, MeshRecord const *b) { return a->slot < b->slot; });
    buildBlases(records, update);
    buildTlas(update);
    update.capacity = capacity();
    update.tlasNodeCount = tlasNodeCount;
}

SceneCapacity RenderWorld::capacity() const noexcept {
    auto const kept = [this](std::uint32_t const count) { return hierarchies ? roundUpToPowerOfTwo(count) : 0u; };
    return SceneCapacity{roundUpToPowerOfTwo(vertexRanges.end()),  roundUpToPowerOfTwo(indexRanges.end()),
                         roundUpToPowerOfTwo(clusterRanges.end()), roundUpToPowerOfTwo(meshSlotEnd),
                         roundUpToPowerOfTwo(instanceCount()),     roundUpToPowerOfTwo(materialSlotEnd),
                         roundUpToPowerOfTwo(lightCount()),        kept(blasNodeRanges.end()),
                         kept(blasTriangleRanges.end()),           kept(tlasNodeCount)};
}

SceneUpdate RenderWorld::apply(core::RenderWorldDelta const &delta) {
    validate(delta);
    auto update = SceneUpdate{};
    ++frames;
    addedInstances = false;
    if (fragmented()) {
        compactGeometry(update);
    }
    upsertMeshes(delta.meshes, update);
    applyTextures(delta, update);
    textures.deliverConversions(update);
    for (auto const &upsert : delta.materials) {
        upsertMaterial(upsert, update);
    }
    upsertInstances(delta.instances, update);
    for (auto const &upsert : delta.lights) {
        auto const found = lightIndex.find(upsert.id);
        auto const index = found != lightIndex.end() ? found->second : lightCount();
        if (index == lightCount()) {
            lights.push_back(LightRecord{upsert.id, upsert.light});
            lightIndex[upsert.id] = index;
        } else {
            lights[index].desc = upsert.light;
        }
        writeLight(index, update);
    }
    if (delta.environment) {
        sky = *delta.environment;
    }

    for (auto const id : delta.removed.instances) {
        removeInstance(id, update);
    }
    for (auto const id : delta.removed.lights) {
        removeLight(id, update);
    }
    for (auto const id : delta.removed.meshes) {
        auto &record = meshes.at(id);
        releaseGeometry(record);
        releaseHierarchy(record);
        freeMeshSlots.push_back(record.slot);
        update.removedMeshes.push_back(record.slot);
        meshes.erase(id);
    }
    for (auto const id : delta.removed.materials) {
        freeMaterialSlots.push_back(materials.at(id).slot);
        materials.erase(id);
    }
    if (hierarchies) {
        updateHierarchies(delta, update);
    }
    auto const &removed = delta.removed;
    if (not delta.meshes.empty() or not delta.materials.empty() or not delta.instances.empty()
        or not removed.instances.empty()) {
        updateSummaries();
    }
    if (changesScene(delta)) {
        ++changes;
    }

    // Writes to elements that later moved or went away are harmless: each index keeps its
    // last write, and the tables hold every index written.
    update.capacity = capacity();
    auto const reach = [](auto const &writes) {
        auto last = std::uint32_t{0};
        for (auto const &write : writes) {
            last = std::max(last, write.index + 1);
        }
        return roundUpToPowerOfTwo(last);
    };
    update.capacity.instances = std::max(update.capacity.instances, reach(update.instances));
    update.capacity.lights = std::max(update.capacity.lights, reach(update.lights));
    update.instanceCount = instanceCount();
    update.lightCount = lightCount();
    update.tlasNodeCount = tlasNodeCount;
    return update;
}

} // namespace lr::render
