#pragma once

#include "Bvh.h"
#include "RenderWork.h"
#include "SceneUpdate.h"
#include "TextureLayout.h"

#include <lr/core/RenderWorldDelta.h>
#include <lr/render/FrameGraph.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lr::render {

// The render thread's copy of the scene. Each frame applies a RenderWorldDelta and turns
// the changes into the SceneUpdate the GPU thread applies to its tables. Instances and
// lights stay dense (removal moves the last one into the gap) because kernels loop over
// them; meshes and materials keep their slots. Geometry, clusters and hierarchies of
// replaced or removed meshes are not reclaimed yet. Texture images pass through to the GPU
// thread (TextureLayout places them); the world keeps only their ids. The meshes of a delta
// are encoded and cut into clusters, and their hierarchies built, as items of `parallel`;
// the textures that need converting are converted on `conversions` and upload with the
// next frame.
struct RenderWorld {
    // Encodes and converts on the calling thread.
    explicit RenderWorld(TextureLimits const &limits = TextureLimits{});
    // `parallel` and `conversions` must outlive the world.
    RenderWorld(TextureLimits const &limits, ParallelFor &parallel, ImageConversions &conversions,
                ConversionDelivery delivery);

    // Material models 0 to count - 1 exist; materials of other models are rejected.
    void allowMaterialModels(std::uint32_t const count) noexcept { materialModels = count; }

    // One call per frame: instances keep the transform the previous call left them with,
    // for motion vectors. Throws std::invalid_argument, leaving the world unchanged, when
    // the delta refers to ids it does not know, materials of material models it does not
    // allow, or carries malformed meshes or images.
    SceneUpdate apply(core::RenderWorldDelta const &delta);

    // Everything the world holds, for a GPU copy that starts out empty (after the device
    // was lost): geometry from the meshes' handles, every slot, and the hierarchies, built
    // anew, when they are kept. Textures are not in it (see loseTextures()).
    SceneUpdate fullUpdate();

    // From now on keeps a hierarchy over the triangles of each mesh and one over the
    // instances: builds them into `update`, and later apply() calls rebuild what their
    // deltas change.
    void keepHierarchies(SceneUpdate &update);
    bool keepsHierarchies() const noexcept { return hierarchies; }

    // For a new device, which holds no textures: the textures placed so far are lost until
    // they are upserted again, and takeLostTextures() lists them.
    void loseTextures();
    std::vector<core::TextureId> takeLostTextures();
    TextureLayout const &textureLayout() const noexcept { return textures; }

    std::uint32_t instanceCount() const noexcept { return static_cast<std::uint32_t>(instances.size()); }
    std::uint32_t lightCount() const noexcept { return static_cast<std::uint32_t>(lights.size()); }
    std::uint32_t transparentInstanceCount() const noexcept { return transparentInstances; }
    // The clusters of every instance's mesh, added up.
    std::uint32_t clusterInstanceCount() const noexcept { return clusterInstances; }
    // Of every instance; empty (min > max) without instances.
    core::Aabb bounds() const noexcept { return sceneBounds; }
    // The first directional light that casts shadows.
    std::optional<ShadowLight> shadowLight() const;
    core::EnvironmentDesc const &environment() const noexcept { return sky; }
    // Counts the deltas that changed meshes, textures, materials, instances, lights or the sky.
    std::uint64_t version() const noexcept { return changes; }

private:
    struct MeshRecord {
        std::uint32_t slot;
        GpuMesh gpu;
        core::MeshHandle data;
        // While hierarchies are kept: the root of the mesh's, and the bounds of its triangles.
        std::uint32_t blasRoot;
        core::Aabb blasBounds;
        std::vector<GpuCluster> clusters; // for a full update
    };
    struct MaterialRecord {
        std::uint32_t slot;
        core::MaterialDesc desc;
    };
    struct InstanceRecord {
        core::InstanceId id;
        core::MeshId mesh;
        core::MaterialId material;
        core::mat4 objectToWorld;
        core::mat4 previousObjectToWorld; // as of the previous apply()
    };
    struct LightRecord {
        core::LightId id;
        core::LightDesc desc;
    };

    void validate(core::RenderWorldDelta const &delta) const;
    std::uint32_t takeSlot(std::vector<std::uint32_t> &freeSlots, std::uint32_t &end);
    void upsertMeshes(std::vector<core::MeshUpsert> const &upserts, SceneUpdate &update);
    void upsertMaterial(core::MaterialUpsert const &upsert, SceneUpdate &update);
    void writeInstance(std::uint32_t index, SceneUpdate &update) const;
    void writeLight(std::uint32_t index, SceneUpdate &update) const;
    void removeInstance(core::InstanceId id, SceneUpdate &update);
    void upsertInstance(core::InstanceUpsert const &upsert, SceneUpdate &update);
    void settleMovedInstances(SceneUpdate &update);
    void removeLight(core::LightId id, SceneUpdate &update);
    void applyTextures(core::RenderWorldDelta const &delta, SceneUpdate &update);
    void updateHierarchies(core::RenderWorldDelta const &delta, SceneUpdate &update);
    void updateSummaries();
    // The hierarchies of `records`' meshes, built as items of `parallel`, in that order.
    void buildBlases(std::vector<MeshRecord *> const &records, SceneUpdate &update);
    void placeBlas(MeshRecord &record, Bvh const &bvh, SceneUpdate &update);
    void buildTlas(SceneUpdate &update);
    SceneCapacity capacity() const noexcept;

    std::unique_ptr<ParallelFor> ownParallel;
    std::unique_ptr<ImageConversions> ownConversions;
    ParallelFor *parallel;

    std::unordered_map<core::MeshId, MeshRecord> meshes;
    std::vector<std::uint32_t> freeMeshSlots;
    std::uint32_t meshSlotEnd = 0;
    std::uint32_t vertexEnd = 0;
    std::uint32_t indexEnd = 0;
    std::uint32_t clusterEnd = 0;

    TextureLayout textures;
    std::unordered_set<core::TextureId> textureIds;
    std::unordered_set<core::TextureId> lostTextures;

    std::unordered_map<core::MaterialId, MaterialRecord> materials;
    std::vector<std::uint32_t> freeMaterialSlots;
    std::uint32_t materialSlotEnd = 0;

    std::vector<InstanceRecord> instances;
    std::unordered_map<core::InstanceId, std::uint32_t> instanceIndex;
    // Upserted by the last apply(): their previous transform is out of date at the next.
    std::vector<core::InstanceId> movedInstances;
    std::uint32_t transparentInstances = 0;
    std::uint32_t clusterInstances = 0;
    std::uint32_t materialModels = 1;
    core::Aabb sceneBounds{core::vec3{INFINITY}, core::vec3{-INFINITY}};

    std::vector<LightRecord> lights;
    std::unordered_map<core::LightId, std::uint32_t> lightIndex;

    core::EnvironmentDesc sky;
    std::uint64_t changes = 0;

    bool hierarchies = false;
    std::uint32_t blasNodeEnd = 0;
    std::uint32_t blasTriangleEnd = 0;
    std::uint32_t tlasNodeCount = 0;
};

} // namespace lr::render
