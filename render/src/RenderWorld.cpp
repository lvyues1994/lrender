#include "RenderWorld.h"

#include "Bvh.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

GpuMaterial encodeMaterial(core::MaterialDesc const &desc) {
    return GpuMaterial{desc.baseColor,   desc.emissive,    desc.metallic, desc.roughness,
                       desc.alphaCutoff, desc.doubleSided ? materialDoubleSided() : 0u};
}

bool changesScene(core::RenderWorldDelta const &delta) noexcept {
    auto const &removed = delta.removed;
    return not delta.meshes.empty() or not delta.materials.empty() or not delta.instances.empty()
        or not delta.lights.empty() or delta.environment.has_value() or not removed.meshes.empty()
        or not removed.materials.empty() or not removed.instances.empty() or not removed.lights.empty();
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
    auto meshIds = std::unordered_set<core::MeshId>{};
    for (auto const &[id, record] : meshes) {
        meshIds.insert(id);
    }
    for (auto const &upsert : delta.meshes) {
        validateMesh(upsert);
        meshIds.insert(upsert.id);
    }
    auto materialIds = std::unordered_set<core::MaterialId>{};
    for (auto const &[id, record] : materials) {
        materialIds.insert(id);
    }
    for (auto const &upsert : delta.materials) {
        materialIds.insert(upsert.id);
    }
    for (auto const &upsert : delta.instances) {
        if (meshIds.count(upsert.mesh) == 0 or materialIds.count(upsert.material) == 0) {
            throw std::invalid_argument{"lr::render: instance " + std::to_string(upsert.id.value)
                                        + " refers to a mesh or material the world does not have"};
        }
    }

    auto instanceIds = std::unordered_set<core::InstanceId>{};
    for (auto const &record : instances) {
        instanceIds.insert(record.id);
    }
    for (auto const &upsert : delta.instances) {
        instanceIds.insert(upsert.id);
    }
    auto lightIds = std::unordered_set<core::LightId>{};
    for (auto const &record : lights) {
        lightIds.insert(record.id);
    }
    for (auto const &upsert : delta.lights) {
        lightIds.insert(upsert.id);
    }
    auto const &removed = delta.removed;
    auto const missing = [](auto const &ids, auto const &known) {
        return std::any_of(ids.begin(), ids.end(), [&known](auto const id) { return known.count(id) == 0; });
    };
    if (missing(removed.meshes, meshIds) or missing(removed.materials, materialIds)
        or missing(removed.instances, instanceIds) or missing(removed.lights, lightIds)) {
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

void RenderWorld::upsertMesh(core::MeshUpsert const &upsert, SceneUpdate &update) {
    auto const &data = *upsert.mesh;
    auto const found = meshes.find(upsert.id);
    auto const slot = found != meshes.end() ? found->second.slot : takeSlot(freeMeshSlots, meshSlotEnd);

    auto geometry = GeometryUpload{};
    geometry.firstVertex = vertexEnd;
    geometry.firstIndex = indexEnd;
    geometry.vertices.reserve(data.positions.size());
    for (auto index = std::size_t{0}; index < data.positions.size(); ++index) {
        auto const uv = data.uvs.empty() ? core::vec2{0.0f} : data.uvs[index];
        geometry.vertices.push_back(GpuVertex{data.positions[index], tc::normalize(data.normals[index]), uv});
    }
    geometry.indices.assign(data.indices.begin(), data.indices.end());
    vertexEnd += static_cast<std::uint32_t>(geometry.vertices.size());
    indexEnd += static_cast<std::uint32_t>(geometry.indices.size());

    auto const bounds = core::boundsOf(data);
    auto const gpu = GpuMesh{geometry.firstVertex, geometry.firstIndex,
                             static_cast<std::uint32_t>(data.indices.size() / 3), bounds.min, bounds.max};
    meshes[upsert.id] = MeshRecord{slot, gpu, upsert.mesh, noBvhNode(), core::Aabb{}};
    update.geometry.push_back(std::move(geometry));
    update.meshes.push_back(Indexed<GpuMesh>{slot, gpu});
}

void RenderWorld::upsertMaterial(core::MaterialUpsert const &upsert, SceneUpdate &update) {
    auto const found = materials.find(upsert.id);
    auto const slot = found != materials.end() ? found->second.slot : takeSlot(freeMaterialSlots, materialSlotEnd);
    materials[upsert.id] = MaterialRecord{slot, upsert.material};
    update.materials.push_back(Indexed<GpuMaterial>{slot, encodeMaterial(upsert.material)});
}

void RenderWorld::writeInstance(std::uint32_t const index, SceneUpdate &update) const {
    auto const &record = instances[index];
    auto const gpu = GpuInstance{record.objectToWorld, tc::inverse(record.objectToWorld),
                                 meshes.at(record.mesh).slot, materials.at(record.material).slot};
    update.instances.push_back(Indexed<GpuInstance>{index, gpu});
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
        instanceIndex[instances[index].id] = index;
        writeInstance(index, update);
    }
    instances.pop_back();
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

void RenderWorld::buildBlas(MeshRecord &record, SceneUpdate &update) {
    auto const &data = *record.data;
    auto const bvh = buildBvh(triangleBounds(data), BvhBuildConfig{4});
    record.blasRoot = bvh.nodes.empty() ? noBvhNode() : blasNodeEnd;
    record.blasBounds = bvh.nodes.empty() ? core::Aabb{} : boundsOf(bvh.nodes[0]);
    update.blasRoots.push_back(Indexed<tc::uint>{record.slot, record.blasRoot});
    for (auto index = std::size_t{0}; index < bvh.nodes.size(); ++index) {
        auto node = bvh.nodes[index];
        node.first += node.count == 0 ? blasNodeEnd : blasTriangleEnd;
        update.blasNodes.push_back(Indexed<GpuBvhNode>{blasNodeEnd + static_cast<std::uint32_t>(index), node});
    }
    auto const firstVertex = record.gpu.firstVertex;
    for (auto index = std::size_t{0}; index < bvh.order.size(); ++index) {
        auto const first = std::size_t{bvh.order[index]} * 3;
        auto const corners = tc::uvec3{firstVertex + data.indices[first], firstVertex + data.indices[first + 1],
                                       firstVertex + data.indices[first + 2]};
        update.blasTriangles.push_back(
            Indexed<GpuBvhTriangle>{blasTriangleEnd + static_cast<std::uint32_t>(index), GpuBvhTriangle{corners}});
    }
    blasNodeEnd += static_cast<std::uint32_t>(bvh.nodes.size());
    blasTriangleEnd += static_cast<std::uint32_t>(bvh.order.size());
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
    for (auto index = std::size_t{0}; index < bvh.nodes.size(); ++index) {
        auto node = bvh.nodes[index];
        if (node.count != 0) {
            node.first = members[bvh.order[node.first]];
        }
        update.tlasNodes.push_back(Indexed<GpuBvhNode>{static_cast<std::uint32_t>(index), node});
    }
    tlasNodeCount = static_cast<std::uint32_t>(bvh.nodes.size());
}

void RenderWorld::updateHierarchies(core::RenderWorldDelta const &delta, SceneUpdate &update) {
    for (auto const &upsert : delta.meshes) {
        auto const found = meshes.find(upsert.id);
        if (found != meshes.end()) {
            buildBlas(found->second, update);
        }
    }
    if (not delta.meshes.empty() or not delta.instances.empty() or not delta.removed.instances.empty()) {
        buildTlas(update);
    }
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
    for (auto *const record : records) {
        buildBlas(*record, update);
    }
    buildTlas(update);
    update.capacity = capacity();
    update.tlasNodeCount = tlasNodeCount;
}

SceneCapacity RenderWorld::capacity() const noexcept {
    auto const kept = [this](std::uint32_t const count) { return hierarchies ? roundUpToPowerOfTwo(count) : 0u; };
    return SceneCapacity{roundUpToPowerOfTwo(vertexEnd),       roundUpToPowerOfTwo(indexEnd),
                         roundUpToPowerOfTwo(meshSlotEnd),     roundUpToPowerOfTwo(instanceCount()),
                         roundUpToPowerOfTwo(materialSlotEnd), roundUpToPowerOfTwo(lightCount()),
                         kept(blasNodeEnd),                    kept(blasTriangleEnd),
                         kept(tlasNodeCount)};
}

SceneUpdate RenderWorld::apply(core::RenderWorldDelta const &delta) {
    validate(delta);
    auto update = SceneUpdate{};
    for (auto const &upsert : delta.meshes) {
        upsertMesh(upsert, update);
    }
    for (auto const &upsert : delta.materials) {
        upsertMaterial(upsert, update);
    }
    for (auto const &upsert : delta.instances) {
        auto const found = instanceIndex.find(upsert.id);
        auto const index = found != instanceIndex.end() ? found->second : instanceCount();
        auto const record = InstanceRecord{upsert.id, upsert.mesh, upsert.material, upsert.objectToWorld};
        if (index == instanceCount()) {
            instances.push_back(record);
            instanceIndex[upsert.id] = index;
        } else {
            instances[index] = record;
        }
        writeInstance(index, update);
    }
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
        freeMeshSlots.push_back(meshes.at(id).slot);
        meshes.erase(id);
    }
    for (auto const id : delta.removed.materials) {
        freeMaterialSlots.push_back(materials.at(id).slot);
        materials.erase(id);
    }
    if (hierarchies) {
        updateHierarchies(delta, update);
    }
    if (changesScene(delta)) {
        ++changes;
    }

    // Writes to elements that later moved or went away are harmless: each index keeps its
    // last write, and the tables only ever grow.
    update.capacity = capacity();
    update.instanceCount = instanceCount();
    update.lightCount = lightCount();
    update.tlasNodeCount = tlasNodeCount;
    return update;
}

} // namespace lr::render
