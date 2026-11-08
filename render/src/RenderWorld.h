#pragma once

#include "Bvh.h"
#include "IdIndex.h"
#include "RangeAllocator.h"
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
// them; meshes and materials keep their slots. A mesh's vertices, indices, clusters and
// hierarchy take ranges of their tables, which a replaced or removed mesh gives back; once
// more of a table lies free than taken, the next apply() packs the meshes together
// (SceneUpdate::moves). Texture images pass through to the GPU
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

    // The slot `id` has in the mesh table, while the world holds it.
    std::optional<std::uint32_t> meshSlot(core::MeshId id) const;
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
    // Counts apply() calls: the frames, as GpuInstance::movedFrame has them.
    std::uint32_t frame() const noexcept { return frames; }

private:
    struct MeshRecord {
        std::uint32_t slot;
        GpuMesh gpu;
        core::MeshHandle data;
        // While hierarchies are kept: the root of the mesh's (its first node), and the bounds
        // of its triangles.
        std::uint32_t blasRoot;
        core::Aabb blasBounds;
        std::vector<GpuCluster> clusters; // for a full update
        std::uint32_t blasNodeCount = 0;
        std::uint32_t blasTriangleFirst = 0;
        std::uint32_t blasTriangleCount = 0;
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
        core::mat4 previousObjectToWorld; // as of the apply() before movedFrame's
        core::mat4 worldToObject;
        std::uint32_t movedFrame;
    };
    // What updateSummaries() adds up, of kInstancesPerChunk instances in order; only the
    // chunks an apply() changes are added up again.
    struct InstanceChunk {
        std::uint32_t transparent = 0;
        std::uint32_t clusters = 0;
        core::Aabb bounds{core::vec3{INFINITY}, core::vec3{-INFINITY}};
        bool dirty = true;
    };
    struct LightRecord {
        core::LightId id;
        core::LightDesc desc;
    };

    void validate(core::RenderWorldDelta const &delta) const;
    std::uint32_t takeSlot(std::vector<std::uint32_t> &freeSlots, std::uint32_t &end);
    void upsertMeshes(std::vector<core::MeshUpsert> const &upserts, SceneUpdate &update);
    void releaseGeometry(MeshRecord const &record);
    void releaseHierarchy(MeshRecord &record);
    // Whether some table has more elements free before its end than taken.
    bool fragmented() const noexcept;
    void compactGeometry(SceneUpdate &update);
    void upsertMaterial(core::MaterialUpsert const &upsert, SceneUpdate &update);
    GpuInstance encodeInstance(std::uint32_t index) const;
    void writeInstance(std::uint32_t index, SceneUpdate &update) const;
    void upsertInstances(std::vector<core::InstanceUpsert> const &upserts, SceneUpdate &update);
    void writeLight(std::uint32_t index, SceneUpdate &update) const;
    void removeInstance(core::InstanceId id, SceneUpdate &update);
    void upsertInstance(core::InstanceUpsert const &upsert, SceneUpdate &update);
    void removeLight(core::LightId id, SceneUpdate &update);
    void applyTextures(core::RenderWorldDelta const &delta, SceneUpdate &update);
    void updateHierarchies(core::RenderWorldDelta const &delta, SceneUpdate &update);
    void markInstance(std::uint32_t index);
    void markAllInstances();
    void updateSummaries();
    // The hierarchies of `records`' meshes, built as items of `parallel`, in that order.
    void buildBlases(std::vector<MeshRecord *> const &records, SceneUpdate &update);
    void placeBlas(MeshRecord &record, Bvh const &bvh, SceneUpdate &update);
    void buildTlas(SceneUpdate &update);
    // Changes to the instance hierarchy as apply() adds and removes instances, between
    // builds: a new leaf goes where its box makes the tree grow least, beside the leaf it
    // finds there; a removed leaf's sibling takes their parent's place, and the pair they
    // held is kept for a later leaf. Their boxes are made right by refitTlas().
    void insertTlasLeaf(std::uint32_t instance, core::Aabb const &box);
    void removeTlasLeaf(std::uint32_t instance);
    // The boxes again of the leaves of `delta`'s instances and of the nodes the changes
    // above touched, and of every node over them.
    void refitTlas(core::RenderWorldDelta const &delta, SceneUpdate &update);
    SceneCapacity capacity() const noexcept;

    std::unique_ptr<ParallelFor> ownParallel;
    std::unique_ptr<ImageConversions> ownConversions;
    ParallelFor *parallel;

    std::unordered_map<core::MeshId, MeshRecord> meshes;
    std::vector<std::uint32_t> freeMeshSlots;
    std::uint32_t meshSlotEnd = 0;
    RangeAllocator vertexRanges;
    RangeAllocator indexRanges;
    RangeAllocator clusterRanges;

    TextureLayout textures;
    std::unordered_set<core::TextureId> textureIds;
    std::unordered_set<core::TextureId> lostTextures;

    std::unordered_map<core::MaterialId, MaterialRecord> materials;
    std::vector<std::uint32_t> freeMaterialSlots;
    std::uint32_t materialSlotEnd = 0;

    std::vector<InstanceRecord> instances;
    IdIndex<core::InstanceId> instanceIndex;
    std::vector<InstanceChunk> instanceChunks;
    // Per instance, the frame that last upserted it, while upsertInstances() runs.
    std::vector<std::uint32_t> upsertFrames;
    std::uint32_t transparentInstances = 0;
    std::uint32_t clusterInstances = 0;
    std::uint32_t materialModels = 1;
    core::Aabb sceneBounds{core::vec3{INFINITY}, core::vec3{-INFINITY}};

    std::vector<LightRecord> lights;
    std::unordered_map<core::LightId, std::uint32_t> lightIndex;

    core::EnvironmentDesc sky;
    std::uint64_t changes = 0;
    std::uint32_t frames = 0;

    bool hierarchies = false;
    RangeAllocator blasNodeRanges;
    RangeAllocator blasTriangleRanges;
    std::uint32_t tlasNodeCount = 0;
    // The instance hierarchy as it was last built or changed, each node's parent, and each
    // instance's leaf (noBvhNode() for those it leaves out).
    std::vector<GpuBvhNode> tlas;
    std::vector<std::uint32_t> tlasParents;
    std::vector<std::uint32_t> tlasLeaves;
    std::vector<std::uint32_t> tlasFreePairs; // the first of two sibling slots no node uses
    std::vector<std::uint32_t> tlasChanged;   // by this apply(), for refitTlas()
    std::uint32_t tlasEdits = 0;              // leaves added and removed since the last build
    bool tlasStale = false;                   // to be built again in updateHierarchies()
};

} // namespace lr::render
