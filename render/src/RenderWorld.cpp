#include "RenderWorld.h"

#include "Bvh.h"

#include <algorithm>
#include <cmath>
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
    auto knownTextures = textureIds;
    for (auto const &upsert : delta.textures) {
        validateImage(upsert);
        knownTextures.insert(upsert.id);
    }
    auto materialIds = std::unordered_set<core::MaterialId>{};
    for (auto const &[id, record] : materials) {
        materialIds.insert(id);
    }
    for (auto const &upsert : delta.materials) {
        if (upsert.material.model.value >= materialModels) {
            throw std::invalid_argument{"lr::render: material " + std::to_string(upsert.id.value)
                                        + " uses material model " + std::to_string(upsert.material.model.value)
                                        + ", which the renderer does not have"};
        }
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
    if (missing(removed.meshes, meshIds) or missing(removed.textures, knownTextures)
        or missing(removed.materials, materialIds) or missing(removed.instances, instanceIds)
        or missing(removed.lights, lightIds)) {
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

    auto geometry = encodeGeometry(data, vertexEnd, indexEnd);
    vertexEnd += static_cast<std::uint32_t>(geometry.vertices.size());
    indexEnd += static_cast<std::uint32_t>(geometry.indices.size());

    auto const bounds = core::boundsOf(data);
    auto const gpu =
        GpuMesh{geometry.firstVertex, geometry.firstIndex, static_cast<std::uint32_t>(data.indices.size() / 3),
                bounds.min,           bounds.max,          static_cast<std::uint32_t>(geometry.vertices.size())};
    meshes[upsert.id] = MeshRecord{slot, gpu, upsert.mesh, noBvhNode(), core::Aabb{}};
    update.geometry.push_back(std::move(geometry));
    update.meshes.push_back(Indexed<GpuMesh>{slot, gpu});
}

void RenderWorld::upsertMaterial(core::MaterialUpsert const &upsert, SceneUpdate &update) {
    auto const found = materials.find(upsert.id);
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

std::optional<ShadowLight> RenderWorld::shadowLight() const {
    for (auto index = std::uint32_t{0}; index < lightCount(); ++index) {
        auto const *directional = std::get_if<core::DirectionalLight>(&lights[index].desc);
        if (directional != nullptr and directional->castsShadows) {
            return ShadowLight{index, tc::normalize(directional->direction)};
        }
    }
    return std::nullopt;
}

void RenderWorld::updateSummaries() {
    transparentInstances = 0;
    sceneBounds = core::Aabb{core::vec3{INFINITY}, core::vec3{-INFINITY}};
    for (auto const &record : instances) {
        if (materials.at(record.material).desc.alphaMode == core::AlphaMode::Blend) {
            ++transparentInstances;
        }
        auto const &mesh = meshes.at(record.mesh).gpu;
        if (mesh.triangleCount != 0) {
            auto const box = core::transformed(core::Aabb{mesh.boundsMin, mesh.boundsMax}, record.objectToWorld);
            sceneBounds.min = tc::min(sceneBounds.min, box.min);
            sceneBounds.max = tc::max(sceneBounds.max, box.max);
        }
    }
}

void RenderWorld::writeInstance(std::uint32_t const index, SceneUpdate &update) const {
    auto const &record = instances[index];
    auto const gpu = GpuInstance{record.objectToWorld, tc::inverse(record.objectToWorld), meshes.at(record.mesh).slot,
                                 materials.at(record.material).slot, record.previousObjectToWorld};
    update.instances.push_back(Indexed<GpuInstance>{index, gpu});
}

// A new instance was nowhere before; one already there was where the last frame had it.
void RenderWorld::upsertInstance(core::InstanceUpsert const &upsert, SceneUpdate &update) {
    auto const found = instanceIndex.find(upsert.id);
    if (found == instanceIndex.end()) {
        auto const index = instanceCount();
        instances.push_back(
            InstanceRecord{upsert.id, upsert.mesh, upsert.material, upsert.objectToWorld, upsert.objectToWorld});
        instanceIndex[upsert.id] = index;
        writeInstance(index, update);
        return;
    }
    auto &record = instances[found->second];
    record = InstanceRecord{upsert.id, upsert.mesh, upsert.material, upsert.objectToWorld, record.objectToWorld};
    movedInstances.push_back(upsert.id);
    writeInstance(found->second, update);
}

// The instances the last frame moved stay where it put them.
void RenderWorld::settleMovedInstances(SceneUpdate &update) {
    for (auto const id : movedInstances) {
        auto const found = instanceIndex.find(id);
        if (found != instanceIndex.end()) {
            auto &record = instances[found->second];
            record.previousObjectToWorld = record.objectToWorld;
            writeInstance(found->second, update);
        }
    }
    movedInstances.clear();
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
        auto const primitive = bvh.order[index];
        auto const first = std::size_t{primitive} * 3;
        auto const corners = tc::uvec3{firstVertex + data.indices[first], firstVertex + data.indices[first + 1],
                                       firstVertex + data.indices[first + 2]};
        update.blasTriangles.push_back(Indexed<GpuBvhTriangle>{blasTriangleEnd + static_cast<std::uint32_t>(index),
                                                               GpuBvhTriangle{corners, primitive}});
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

SceneUpdate RenderWorld::fullUpdate() {
    auto update = SceneUpdate{};
    for (auto const &[id, record] : meshes) {
        update.geometry.push_back(encodeGeometry(*record.data, record.gpu.firstVertex, record.gpu.firstIndex));
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
        blasNodeEnd = 0;
        blasTriangleEnd = 0;
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
    applyTextures(delta, update);
    for (auto const &upsert : delta.materials) {
        upsertMaterial(upsert, update);
    }
    settleMovedInstances(update);
    for (auto const &upsert : delta.instances) {
        upsertInstance(upsert, update);
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
        update.removedMeshes.push_back(meshes.at(id).slot);
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
    for (auto const &write : update.instances) {
        update.capacity.instances = std::max(update.capacity.instances, roundUpToPowerOfTwo(write.index + 1));
    }
    for (auto const &write : update.lights) {
        update.capacity.lights = std::max(update.capacity.lights, roundUpToPowerOfTwo(write.index + 1));
    }
    update.instanceCount = instanceCount();
    update.lightCount = lightCount();
    update.tlasNodeCount = tlasNodeCount;
    return update;
}

} // namespace lr::render
