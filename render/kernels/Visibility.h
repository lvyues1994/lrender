#pragma once

// The visibility buffer: cull instances, compact the visible ones in instance order, then
// rasterize their triangles, keeping per pixel the nearest depth and, among the triangles
// at that depth, the lowest visible-triangle number. Devices with 64-bit atomics do that
// in one pass, an atomicMin of (depth << 32 | triangle) per pixel, and unpack the result;
// the others rasterize twice with RasterVisibility, first the depth, then the triangle
// where its depth equals the nearest. Both rasterizers run the same code
// (LR_TRIANGLE_RASTERIZER), so they find bit-identical depths and triangles, and split the
// work the same way: an invocation per small triangle, a workgroup per tile of a large one.
// RasterVisibility also draws shadow maps (depth only) and collects the fragments of
// transparent instances into per-pixel lists.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/Material.h>
#include <lr/render/kernels/VisibleInstances.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

struct TC_KERNEL ClearUint {
    static constexpr char fileLocation[] = "lr_clear_uint";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> target;
    Uniform<uint, 0> fill{};
    Uniform<uint, 1> first{};
    Uniform<uint, 2> count{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < count.value) {
            target[first.value + i] = fill.value;
        }
    }
};

// Whether the box, seen through `toClip`, lies wholly outside one plane of the frustum.
inline bool outsideFrustum(mat4 const &toClip, vec3 const &low, vec3 const &high) {
    uint left = 0u;
    uint right = 0u;
    uint bottom = 0u;
    uint top = 0u;
    uint nearSide = 0u;
    uint farSide = 0u;
    for (uint corner = 0u; corner < 8u; ++corner) {
        vec3 point = vec3{(corner & 1u) != 0u ? high.x : low.x, (corner & 2u) != 0u ? high.y : low.y,
                          (corner & 4u) != 0u ? high.z : low.z};
        vec4 clip = toClip * vec4(point, 1.0f);
        if (clip.x < -clip.w) {
            left += 1u;
        }
        if (clip.x > clip.w) {
            right += 1u;
        }
        if (clip.y < -clip.w) {
            bottom += 1u;
        }
        if (clip.y > clip.w) {
            top += 1u;
        }
        if (clip.z < 0.0f) {
            nearSide += 1u;
        }
        if (clip.z > clip.w) {
            farSide += 1u;
        }
    }
    return left == 8u or right == 8u or bottom == 8u or top == 8u or nearSide == 8u or farSide == 8u;
}

// Whether the box, seen through `toClip`, lies wholly beyond a side of the view or before
// its near plane: where the rasterizer covers no pixel of what is in it. It does not clip
// at the far plane, so neither does this.
inline bool outsideRaster(mat4 const &toClip, vec3 const &low, vec3 const &high) {
    uint left = 0u;
    uint right = 0u;
    uint bottom = 0u;
    uint top = 0u;
    uint nearSide = 0u;
    for (uint corner = 0u; corner < 8u; ++corner) {
        vec3 point = vec3{(corner & 1u) != 0u ? high.x : low.x, (corner & 2u) != 0u ? high.y : low.y,
                          (corner & 4u) != 0u ? high.z : low.z};
        vec4 clip = toClip * vec4(point, 1.0f);
        if (clip.x < -clip.w) {
            left += 1u;
        }
        if (clip.x > clip.w) {
            right += 1u;
        }
        if (clip.y < -clip.w) {
            bottom += 1u;
        }
        if (clip.y > clip.w) {
            top += 1u;
        }
        if (clip.z < 0.0f) {
            nearSide += 1u;
        }
    }
    return left == 8u or right == 8u or bottom == 8u or top == 8u or nearSide == 8u;
}

// counts[i] = (clusters, triangles, 0) of instance i if it may be visible and its material
// blends (`blended` 1) or does not (`blended` 0), else (0, 0, 0), with its mesh's clusters
// in w either way: they number the scene's clusters for occlusion culling's history.
struct TC_KERNEL CullInstances {
    static constexpr char fileLocation[] = "lr_cull_instances";
    uvec3 local_size{64, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    MaterialTable<2> materials;
    BufferBinding<uvec4, 3> counts;
    Uniform<mat4, 0> viewProjection{};
    Uniform<uint, 1> instanceCount{};
    Uniform<uint, 2> blended{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= instanceCount.value) {
            return;
        }
        uint meshIndex = instances[i].mesh;
        GpuMesh mesh = meshes[meshIndex];
        uint blends = (materials[instances[i].material].flags & materialAlphaBlend()) != 0u ? 1u : 0u;
        mat4 toClip = viewProjection.value * instances[i].objectToWorld.value;
        if (blends != blended.value or outsideFrustum(toClip, mesh.boundsMin, mesh.boundsMax)) {
            counts[i] = uvec4{0u, 0u, 0u, mesh.clusterCount};
        } else {
            counts[i] = uvec4{mesh.clusterCount, mesh.triangleCount, 0u, mesh.clusterCount};
        }
    }
};

// findInstance(candidate): the slot in `list`, a list of instances, of the visible
// instance whose clusters include candidate cluster `candidate`.
#define LR_FIND_INSTANCE(list)                                                                                        \
    uint findInstance(uint candidate) {                                                                               \
        uint tail = list[0].w;                                                                                        \
        uint low = 0u;                                                                                                \
        uint high = list[0].x;                                                                                        \
        while (high - low > 1u) {                                                                                     \
            uint middle = (low + high) / 2u;                                                                          \
            if (list[tail + middle].x <= candidate) {                                                                 \
                low = middle;                                                                                         \
            } else {                                                                                                  \
                high = middle;                                                                                        \
            }                                                                                                         \
        }                                                                                                             \
        return low;                                                                                                   \
    }

inline uint cullClusterBoxes() { return 1u; }
inline uint cullClusterCones() { return 2u; }
// Cones that only just cull keep their clusters: their triangles are nearly edge-on, and
// whether the rasterizer finds such a triangle's area negative depends on its rounding.
inline float coneMargin() { return 1.0e-3f; }

// Whether a viewer at viewer.xyz (w 1), or looking along viewer.xyz (w 0), all in the
// cluster's space, sees only the backs of its triangles: meshoptimizer's test with a margin.
inline bool facesAway(GpuCluster const &cluster, vec4 const &viewer) {
    float cutoff = cluster.coneCutoff + coneMargin();
    if (viewer.w == 0.0f) {
        return dot(cluster.coneAxis, normalize(viewer.xyz())) >= cutoff;
    }
    vec3 toCenter = cluster.center - viewer.xyz();
    return dot(toCenter, cluster.coneAxis) >= cutoff * length(toCenter) + cluster.radius;
}

// Occlusion culling's phases for CullClusters (see CullOccludedClusters).
inline uint occlusionOff() { return 0u; }
inline uint occlusionWithHistory() { return 1u; }
inline uint occlusionWithoutHistory() { return 2u; }

// The clusters of the visible instances a compaction listed (see compactBlockInstances()),
// whose tail holds the first cluster of each among them and visible[0].z their number:
// an invocation each (the compaction's workgroup counts). counts[c] are the triangles of
// candidate c if they may show, else 0. With cullClusterBoxes() in `tests` a cluster
// whose box lies where the rasterizer covers nothing goes, with cullClusterCones() one
// of a single-sided material that faces away from `viewer`: in world space, a position
// (w 1) or the direction the view looks along (w 0). With occlusion culling, only the
// clusters that stay and were visible the frame before (previous[their number among the
// scene's], if occlusionWithHistory()) are counted, and states[c] records whether
// candidate c stayed (bit 0) and was visible (bit 1).
struct TC_KERNEL CullClusters {
    static constexpr char fileLocation[] = "lr_cull_clusters";
    uvec3 local_size{64, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    ClusterTable<2> clusters;
    MaterialTable<3> materials;
    BufferBinding<uvec4, 4> visible;
    BufferBinding<uint, 5> counts;
    BufferBinding<uint, 6> previous;
    BufferBinding<uint, 7> states;
    Uniform<mat4, 0> viewProjection{};
    Uniform<vec4, 1> viewer{};
    Uniform<uint, 2> tests{};
    Uniform<uint, 3> occlusion{};

    LR_FIND_INSTANCE(visible)

    void main() {
        uint candidate = gl_GlobalInvocationID.x;
        if (candidate >= visible[0].z) {
            return;
        }
        uint slot = findInstance(candidate);
        uint instance = visible[1u + slot].x;
        uvec4 tail = visible[visible[0].w + slot];
        GpuMesh mesh = meshes[instances[instance].mesh];
        GpuCluster cluster = clusters[mesh.firstCluster + candidate - tail.x];
        bool keep = true;
        if ((tests.value & cullClusterBoxes()) != 0u) {
            mat4 toClip = viewProjection.value * instances[instance].objectToWorld.value;
            keep = not outsideRaster(toClip, cluster.boundsMin, cluster.boundsMax);
        }
        if (keep and (tests.value & cullClusterCones()) != 0u
            and (materials[instances[instance].material].flags & materialDoubleSided()) == 0u) {
            keep = not facesAway(cluster, instances[instance].worldToObject.value * viewer.value);
        }
        bool seen = occlusion.value == occlusionWithHistory() and previous[tail.y + candidate - tail.x] != 0u;
        bool counted = keep and (occlusion.value == occlusionOff() or seen);
        counts[candidate] = counted ? cluster.triangleCount : 0u;
        if (occlusion.value != occlusionOff()) {
            states[candidate] = (keep ? 1u : 0u) | (seen ? 2u : 0u);
        }
    }
};

// Compaction lists the items that have a count, in order, in three passes: the workgroups
// of SumVisibleBlocks add up their sums in a block of compactBlockInstances() items each,
// ScanVisibleBlocks (one workgroup) turns those into each block's first sums, and the
// workgroups of WriteVisible write their block's entries from there. The sums of an item
// are (1 if it is listed, then up to three counts). The items are instances, with sums
// (listed, triangles, clusters) from CullInstances, then the clusters of the visible ones,
// with sums (listed, triangles) from CullClusters (SumClusterBlocks, WriteVisibleClusters).
// A list: visible[0] = (items listed, the sums in all, where its tail starts); visible[1 +
// k] = (instance, its first triangle, first index, first vertex) for the k-th, the first
// index that of the cluster's triangles for clusters; the tail, visible[visible[0].w + k],
// holds more for each: an instance's first cluster among the candidates in x and among
// all the scene's clusters (instances in order) in y; in x the number a cluster's first
// triangle has among those of the visible instances. The first
// triangles count from 0 in each list; the raster passes write the clusters' numbers, so
// what they write does not depend on which clusters a list holds, and shading resolves
// them through the list of instances. ScanVisibleBlocks also writes the workgroup counts
// of the next pass, an invocation per count of one of the sums: per candidate cluster,
// per triangle of a listed cluster. The block passes have a variant that scans within
// subgroups.
inline uint compactBlockInstances() { return 1024u; } // four per invocation
inline uint compactBlocks(uint instances) { return (instances + 1023u) / 1024u; }

// groupExclusiveScan(value, local): the sum of `value` over the invocations of the
// 256-invocation workgroup before this one; scanTotal.value then holds the sum over all.
// Every invocation calls it.
#define LR_GROUP_SCAN                                                                                                 \
    Shared<std::array<uvec4, 256>> scanned;                                                                           \
    Shared<uvec4> scanTotal;                                                                                          \
                                                                                                                      \
    uvec4 groupExclusiveScan(uvec4 value, uint local) {                                                               \
        scanned[local] = value;                                                                                       \
        barrier();                                                                                                    \
        for (uint offset = 1u; offset < 256u; offset <<= 1u) {                                                        \
            uvec4 before = local >= offset ? scanned[local - offset] : uvec4(0u);                                     \
            barrier();                                                                                                \
            scanned[local] = scanned[local] + before;                                                                 \
            barrier();                                                                                                \
        }                                                                                                             \
        uvec4 inclusive = scanned[local];                                                                             \
        if (local == 255u) {                                                                                          \
            scanTotal.value = inclusive;                                                                              \
        }                                                                                                             \
        barrier();                                                                                                    \
        return inclusive - value;                                                                                     \
    }

// LR_GROUP_SCAN within subgroups; shared memory only holds the subgroups' sums.
#define LR_GROUP_SCAN_SUBGROUPS                                                                                       \
    Shared<std::array<uvec4, 256>> subgroupSums;                                                                      \
    Shared<uvec4> scanTotal;                                                                                          \
                                                                                                                      \
    uvec4 groupExclusiveScan(uvec4 value, uint local) {                                                               \
        uvec4 before = uvec4{subgroupExclusiveAdd(value.x), subgroupExclusiveAdd(value.y),                            \
                             subgroupExclusiveAdd(value.z), subgroupExclusiveAdd(value.w)};                           \
        uvec4 sum = uvec4{subgroupAdd(value.x), subgroupAdd(value.y), subgroupAdd(value.z), subgroupAdd(value.w)};   \
        if (subgroupElect()) {                                                                                        \
            subgroupSums[gl_SubgroupID] = sum;                                                                        \
        }                                                                                                             \
        barrier();                                                                                                    \
        for (uint s = 0u; s < gl_SubgroupID; ++s) {                                                                   \
            before = before + subgroupSums[s];                                                                        \
        }                                                                                                             \
        if (local == 0u) {                                                                                            \
            uvec4 all = uvec4(0u);                                                                                    \
            for (uint s = 0u; s < gl_NumSubgroups; ++s) {                                                             \
                all = all + subgroupSums[s];                                                                          \
            }                                                                                                         \
            scanTotal.value = all;                                                                                    \
        }                                                                                                             \
        barrier();                                                                                                    \
        return before;                                                                                                \
    }

// blocks[b] = the sums of block b, of the first `items`: valueOf(item) for each.
#define LR_SUM_BLOCKS(items)                                                                                          \
    void main() {                                                                                                     \
        uint local = gl_LocalInvocationIndex;                                                                         \
        uint first = gl_WorkGroupID.x * compactBlockInstances() + local * 4u;                                         \
        uvec4 sum = uvec4(0u);                                                                                        \
        for (uint i = 0u; i < 4u; ++i) {                                                                              \
            if (first + i < (items)) {                                                                                \
                sum = sum + valueOf(first + i);                                                                       \
            }                                                                                                         \
        }                                                                                                             \
        groupExclusiveScan(sum, local);                                                                               \
        if (local == 0u) {                                                                                            \
            blocks[gl_WorkGroupID.x] = scanTotal.value;                                                               \
        }                                                                                                             \
    }

#define LR_INSTANCE_SUMS                                                                                              \
    uvec4 valueOf(uint instance) {                                                                                    \
        uvec4 count = counts[instance];                                                                               \
        return uvec4{count.x > 0u ? 1u : 0u, count.y, count.x, count.w};                                              \
    }

#define LR_CLUSTER_SUMS                                                                                               \
    uvec4 valueOf(uint candidate) {                                                                                   \
        uint count = counts[candidate];                                                                               \
        return uvec4{count > 0u ? 1u : 0u, count, 0u, 0u};                                                            \
    }

#define LR_SUM_VISIBLE_BLOCKS                                                                                         \
    BufferBinding<uvec4, 0> counts;                                                                                   \
    BufferBinding<uvec4, 1> blocks;                                                                                   \
    Uniform<uint, 0> instanceCount{};                                                                                 \
    LR_INSTANCE_SUMS                                                                                                  \
    LR_SUM_BLOCKS(instanceCount.value)

// Over the candidate clusters of CullClusters, as many as candidates[0].z.
#define LR_SUM_CLUSTER_BLOCKS                                                                                         \
    BufferBinding<uint, 0> counts;                                                                                    \
    BufferBinding<uvec4, 1> blocks;                                                                                   \
    BufferBinding<uvec4, 2> candidates;                                                                               \
    LR_CLUSTER_SUMS                                                                                                   \
    LR_SUM_BLOCKS(candidates[0].z)

// writeEntry(item, its first sums) for the listed items of block b, from blocks[b] on, of
// the first `items`.
#define LR_WRITE_ENTRIES(items)                                                                                       \
    void main() {                                                                                                     \
        uint local = gl_LocalInvocationIndex;                                                                         \
        uint first = gl_WorkGroupID.x * compactBlockInstances() + local * 4u;                                         \
        std::array<uvec4, 4> mine;                                                                                    \
        uvec4 sum = uvec4(0u);                                                                                        \
        for (uint i = 0u; i < 4u; ++i) {                                                                              \
            mine[i] = first + i < (items) ? valueOf(first + i) : uvec4(0u);                                           \
            sum = sum + mine[i];                                                                                      \
        }                                                                                                             \
        uvec4 next = blocks[gl_WorkGroupID.x] + groupExclusiveScan(sum, local);                                       \
        for (uint i = 0u; i < 4u; ++i) {                                                                              \
            if (mine[i].x > 0u) {                                                                                     \
                writeEntry(first + i, next);                                                                          \
                next = next + mine[i];                                                                                \
            }                                                                                                         \
        }                                                                                                             \
    }

#define LR_WRITE_VISIBLE                                                                                              \
    InstanceTable<0> instances;                                                                                       \
    MeshTable<1> meshes;                                                                                              \
    BufferBinding<uvec4, 2> counts;                                                                                   \
    BufferBinding<uvec4, 3> blocks;                                                                                   \
    BufferBinding<uvec4, 4> visible;                                                                                  \
    Uniform<uint, 0> instanceCount{};                                                                                 \
    LR_INSTANCE_SUMS                                                                                                  \
                                                                                                                      \
    void writeEntry(uint instance, uvec4 at) {                                                                        \
        GpuMesh mesh = meshes[instances[instance].mesh];                                                              \
        visible[1u + at.x] = uvec4{instance, at.y, mesh.firstIndex, mesh.firstVertex};                                \
        visible[visible[0].w + at.x] = uvec4{at.z, at.w, 0u, 0u};                                                     \
    }                                                                                                                 \
    LR_WRITE_ENTRIES(instanceCount.value)

// The visible clusters of CullClusters: (instance, first triangle in this list, first index
// of the cluster's triangles, first vertex of its mesh), its first triangle's number among
// those of the visible instances in the tail.
#define LR_WRITE_VISIBLE_CLUSTERS                                                                                     \
    BufferBinding<uint, 0> counts;                                                                                    \
    BufferBinding<uvec4, 1> blocks;                                                                                   \
    BufferBinding<uvec4, 2> visible;                                                                                  \
    BufferBinding<uvec4, 3> candidates;                                                                               \
    InstanceTable<4> instances;                                                                                       \
    MeshTable<5> meshes;                                                                                              \
    ClusterTable<6> clusters;                                                                                         \
    LR_CLUSTER_SUMS                                                                                                   \
    LR_FIND_INSTANCE(candidates)                                                                                      \
                                                                                                                      \
    void writeEntry(uint candidate, uvec4 at) {                                                                       \
        uint slot = findInstance(candidate);                                                                          \
        uvec4 entry = candidates[1u + slot];                                                                          \
        GpuMesh mesh = meshes[instances[entry.x].mesh];                                                               \
        GpuCluster cluster = clusters[mesh.firstCluster + candidate - candidates[candidates[0].w + slot].x];          \
        visible[1u + at.x] = uvec4{entry.x, at.y, mesh.firstIndex + 3u * cluster.firstTriangle, mesh.firstVertex};    \
        visible[visible[0].w + at.x] = uvec4{entry.y + cluster.firstTriangle, 0u, 0u, 0u};                            \
    }                                                                                                                 \
    LR_WRITE_ENTRIES(candidates[0].z)

struct TC_KERNEL SumVisibleBlocks {
    static constexpr char fileLocation[] = "lr_sum_visible_blocks";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN
    LR_SUM_VISIBLE_BLOCKS
};

struct TC_KERNEL SumVisibleBlocksSubgroups {
    static constexpr char fileLocation[] = "lr_sum_visible_blocks_subgroups";
    static constexpr bool usesBarrier = true;
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN_SUBGROUPS
    LR_SUM_VISIBLE_BLOCKS
};

// One workgroup: blocks[b] becomes the sums of the blocks before b, 256 blocks at a time;
// visible[0] gets the totals and where the tail starts, after `capacity` entries, and the
// next pass's workgroup counts follow from the total of sum `counted` (1 or 2).
struct TC_KERNEL ScanVisibleBlocks {
    static constexpr char fileLocation[] = "lr_scan_visible_blocks";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    BufferBinding<uvec4, 0> blocks;
    BufferBinding<uvec4, 1> visible;
    BufferBinding<DispatchIndirectCommand, 2> rasterArguments;
    Uniform<uint, 0> blockCount{};
    Uniform<uint, 1> counted{};
    Uniform<uint, 2> capacity{};
    Shared<uvec4> base;
    LR_GROUP_SCAN

    void main() {
        uint local = gl_LocalInvocationIndex;
        if (local == 0u) {
            base.value = uvec4(0u);
        }
        barrier();
        for (uint chunk = 0u; chunk < blockCount.value; chunk += 256u) {
            uint block = chunk + local;
            uvec4 sum = block < blockCount.value ? blocks[block] : uvec4(0u);
            uvec4 before = groupExclusiveScan(sum, local);
            if (block < blockCount.value) {
                blocks[block] = base.value + before;
            }
            barrier();
            if (local == 0u) {
                base.value = base.value + scanTotal.value;
            }
            barrier();
        }
        if (local == 0u) {
            visible[0] = uvec4{base.value.x, base.value.y, base.value.z, 1u + capacity.value};
            rasterArguments[0].x = ((counted.value == 2u ? base.value.z : base.value.y) + 63u) / 64u;
            rasterArguments[0].y = 1u;
            rasterArguments[0].z = 1u;
        }
    }
};

// The depth pyramid of occlusion culling: level 0 has half the view's pixels across and
// down (rounded up), each the farthest floatBitsToUint(depth) of the up to 2x2 pixels under
// it, and each level after it half the one before it the same way, down to 1x1; level L's
// texel (x, y) holds the pixels [x, x + 1) * 2^(L + 1) by [y, y + 1) * 2^(L + 1). Pixels
// nothing covers hold the cleared 0xFFFFFFFF, which occludes nothing. The levels follow
// each other in one buffer.
inline ivec2 pyramidLevelSize(ivec2 size) { return (size + ivec2(1)) / 2; }

// hiz[target texel] = the farthest of the 2x2 under it at sourceOffset (a level, or pixels).
#define LR_DOWNSAMPLE_DEPTH(read)                                                                                     \
    Uniform<ivec2, 0> sourceSize{};                                                                                   \
    Uniform<ivec2, 1> targetSize{};                                                                                   \
    Uniform<uint, 2> sourceOffset{};                                                                                  \
    Uniform<uint, 3> targetOffset{};                                                                                  \
                                                                                                                      \
    void main() {                                                                                                     \
        ivec2 texel = ivec2(gl_GlobalInvocationID.xy());                                                              \
        if (texel.x >= targetSize.value.x or texel.y >= targetSize.value.y) {                                         \
            return;                                                                                                   \
        }                                                                                                             \
        uint farthest = 0u;                                                                                           \
        for (int y = 0; y < 2; ++y) {                                                                                 \
            for (int x = 0; x < 2; ++x) {                                                                             \
                ivec2 from = min(texel * 2 + ivec2{x, y}, sourceSize.value - ivec2(1));                               \
                farthest = max(farthest, read(sourceOffset.value + uint(from.y * sourceSize.value.x + from.x)));      \
            }                                                                                                         \
        }                                                                                                             \
        hiz[targetOffset.value + uint(texel.y * targetSize.value.x + texel.x)] = farthest;                            \
    }

struct TC_KERNEL DownsampleDepth {
    static constexpr char fileLocation[] = "lr_downsample_depth";
    uvec3 local_size{8, 8, 1};

    BufferBinding<uint, 0> depth;
    BufferBinding<uint, 1> hiz;

    uint depthAt(uint pixel) { return depth[pixel]; }
    LR_DOWNSAMPLE_DEPTH(depthAt)
};

// The first level from a packed visibility buffer (see RasterPackedVisibility).
struct TC_KERNEL DownsamplePackedDepth {
    static constexpr char fileLocation[] = "lr_downsample_packed_depth";
    uvec3 local_size{8, 8, 1};

    BufferBinding<u64, 0> packed;
    BufferBinding<uint, 1> hiz;

    uint depthAt(uint pixel) { return uint(packed[pixel] >> 32u); }
    LR_DOWNSAMPLE_DEPTH(depthAt)
};

struct TC_KERNEL DownsamplePyramid {
    static constexpr char fileLocation[] = "lr_downsample_pyramid";
    uvec3 local_size{8, 8, 1};

    BufferBinding<uint, 0> hiz;

    uint depthAt(uint texel) { return hiz[texel]; }
    LR_DOWNSAMPLE_DEPTH(depthAt)
};

// Depth that the rasterizer's interpolation may round below a box's nearest corner.
inline float occlusionMargin() { return 1.0e-5f; }

// The second phase of occlusion culling, after the clusters CullClusters counted are drawn
// and the pyramid is built from their depth: every candidate that stayed (states[c] bit 0)
// is tested against the pyramid, current[its number among the scene's] records whether it
// may show, and counts[c] are its triangles if it may show and was not drawn yet (bit 1),
// else 0. `current` must be cleared first: clusters that were not candidates are not
// visible either. A cluster goes only when its nearest point is behind the farthest depth
// drawn over every pixel its box may cover: what the box holds can cover no pixel then,
// nor tie with what is there.
struct TC_KERNEL CullOccludedClusters {
    static constexpr char fileLocation[] = "lr_cull_occluded_clusters";
    uvec3 local_size{64, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    ClusterTable<2> clusters;
    BufferBinding<uvec4, 3> visible;
    BufferBinding<uint, 4> counts;
    BufferBinding<uint, 5> states;
    BufferBinding<uint, 6> current;
    BufferBinding<uint, 7> hiz;
    Uniform<mat4, 0> viewProjection{};
    Uniform<ivec2, 1> size{};
    Uniform<uint, 2> levels{};

    LR_FIND_INSTANCE(visible)

    bool occluded(mat4 toClip, vec3 low, vec3 high) {
        vec2 lowPixel = vec2(3.0e38f);
        vec2 highPixel = vec2(-3.0e38f);
        float nearest = 1.0f;
        for (uint corner = 0u; corner < 8u; ++corner) {
            vec3 point = vec3{(corner & 1u) != 0u ? high.x : low.x, (corner & 2u) != 0u ? high.y : low.y,
                              (corner & 4u) != 0u ? high.z : low.z};
            vec4 clip = toClip * vec4(point, 1.0f);
            if (clip.w <= 0.0f or clip.z < 0.0f) {
                return false; // crosses the near plane: its depth over the screen is unbounded
            }
            float inverseW = 1.0f / clip.w;
            vec2 pixel = vec2{(clip.x * inverseW * 0.5f + 0.5f) * float(size.value.x),
                              (0.5f - clip.y * inverseW * 0.5f) * float(size.value.y)};
            lowPixel = min(lowPixel, pixel);
            highPixel = max(highPixel, pixel);
            nearest = min(nearest, clip.z * inverseW);
        }
        vec2 bound = vec2(size.value);
        ivec2 first = max(ivec2(floor(clamp(lowPixel, vec2(-1.0f), bound))) - ivec2(1), ivec2(0));
        ivec2 last = min(ivec2(floor(clamp(highPixel, vec2(-1.0f), bound))) + ivec2(1), size.value - ivec2(1));
        if (last.x < first.x or last.y < first.y) {
            return false;
        }
        uint span = uint(max(last.x - first.x, last.y - first.y)) + 1u;
        uint level = 0u;
        while (level + 1u < levels.value and (2u << level) < span) {
            level += 1u;
        }
        ivec2 levelSize = pyramidLevelSize(size.value);
        uint offset = 0u;
        for (uint l = 0u; l < level; ++l) {
            offset += uint(levelSize.x * levelSize.y);
            levelSize = pyramidLevelSize(levelSize);
        }
        int texel = int(2u << level);
        ivec2 from = min(first / texel, levelSize - ivec2(1));
        ivec2 to = min(last / texel, levelSize - ivec2(1));
        uint farthest = 0u;
        for (int y = from.y; y <= to.y; ++y) {
            for (int x = from.x; x <= to.x; ++x) {
                farthest = max(farthest, hiz[offset + uint(y * levelSize.x + x)]);
            }
        }
        return floatBitsToUint(max(nearest - occlusionMargin(), 0.0f)) > farthest;
    }

    void main() {
        uint candidate = gl_GlobalInvocationID.x;
        if (candidate >= visible[0].z) {
            return;
        }
        uint state = states[candidate];
        if ((state & 1u) == 0u) {
            counts[candidate] = 0u;
            return;
        }
        uint slot = findInstance(candidate);
        uint instance = visible[1u + slot].x;
        uvec4 tail = visible[visible[0].w + slot];
        GpuMesh mesh = meshes[instances[instance].mesh];
        GpuCluster cluster = clusters[mesh.firstCluster + candidate - tail.x];
        mat4 toClip = viewProjection.value * instances[instance].objectToWorld.value;
        bool shows = not occluded(toClip, cluster.boundsMin, cluster.boundsMax);
        current[tail.y + candidate - tail.x] = shows ? 1u : 0u;
        counts[candidate] = shows and (state & 2u) == 0u ? cluster.triangleCount : 0u;
    }
};

struct TC_KERNEL WriteVisible {
    static constexpr char fileLocation[] = "lr_write_visible";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN
    LR_WRITE_VISIBLE
};

struct TC_KERNEL WriteVisibleSubgroups {
    static constexpr char fileLocation[] = "lr_write_visible_subgroups";
    static constexpr bool usesBarrier = true;
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN_SUBGROUPS
    LR_WRITE_VISIBLE
};

// The same three passes over the candidate clusters (ScanVisibleBlocks as it is), where
// the host knows only how many there may be: workgroups past the candidates do nothing.
struct TC_KERNEL SumClusterBlocks {
    static constexpr char fileLocation[] = "lr_sum_cluster_blocks";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN
    LR_SUM_CLUSTER_BLOCKS
};

struct TC_KERNEL SumClusterBlocksSubgroups {
    static constexpr char fileLocation[] = "lr_sum_cluster_blocks_subgroups";
    static constexpr bool usesBarrier = true;
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN_SUBGROUPS
    LR_SUM_CLUSTER_BLOCKS
};

struct TC_KERNEL WriteVisibleClusters {
    static constexpr char fileLocation[] = "lr_write_visible_clusters";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN
    LR_WRITE_VISIBLE_CLUSTERS
};

struct TC_KERNEL WriteVisibleClustersSubgroups {
    static constexpr char fileLocation[] = "lr_write_visible_clusters_subgroups";
    static constexpr bool usesBarrier = true;
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{256, 1, 1};

    LR_GROUP_SCAN_SUBGROUPS
    LR_WRITE_VISIBLE_CLUSTERS
};

inline uint rasterDepth() { return 0u; }
inline uint rasterTriangles() { return 1u; }
inline uint rasterTransparent() { return 2u; }

// A raster pass runs in two stages, each a kernel. The triangle stage gives each visible
// triangle an
// invocation, which rasterizes it unless its pixel box holds more than largeTriangleArea()
// pixels: those it lists, and ScanLargeTriangles gives each the tiles of
// rasterTileSize()^2 pixels its box meets. The tile stage then gives each tile a
// workgroup, rasterTileSize() invocations across and 4 down, each covering every fourth
// row. A pixel's depth and triangle come out of the same arithmetic either way.
inline uint largeTriangleArea() { return 256u; }
inline uint rasterTileSize() { return 16u; }
// Large triangles one pass lists; past that, the triangle stage rasterizes them itself.
inline constexpr uint largeTriangleCapacity() { return 16384u; }
// The list of large triangles, in words: [0] the triangles listed (which may pass the
// capacity), [1] their tiles in all, then four words per listed triangle from
// largeTriangleWord(i): the visible triangle, its first tile (x | y << 16), its tiles
// across and down (x | y << 16), and how many tiles the triangles before it have.
inline uint largeTriangleWord(uint i) { return 4u + 4u * i; }
inline constexpr uint largeTriangleWords() { return 4u + 4u * largeTriangleCapacity(); }
// Workgroups along x of the tile stage; the rest of the tiles go along y.
inline uint rasterTilesPerRow() { return 32768u; }

struct RasterVertex {
    vec4 clip;
    vec2 uv;
};

// The pixels a triangle may cover: from `low`, every step.x-th column and step.y-th row
// up to `high`.
struct RasterWindow {
    ivec2 low;
    ivec2 high;
    ivec2 step;
};

// A visible triangle and its material, clipped against the near plane into a fan of
// count - 2 triangles from polygon[0]: `triangle` in its raster pass's list, `id` what the
// pass writes for it (see compactBlockInstances()).
struct RasterTriangle {
    std::array<RasterVertex, 4> polygon;
    uint count;
    uint triangle;
    uint id;
    GpuMaterial material;
    bool doubleSided;
    bool mirrored;
};

// The first of low, low + step, ... that is at least `from`.
inline int windowStart(int from, int low, int step) {
    int ahead = max(from - low, 0);
    return low + (ahead + step - 1) / step * step;
}

// The edge function of point p against the edge from `from` to `to`, positive on its right
// on the screen (rows run down). Computed from the endpoints in a fixed order, so the
// triangles on both sides of an edge get exactly opposite values.
inline float edgeFunction(vec2 from, vec2 to, vec2 p) {
    bool swapped = from.x > to.x or (from.x == to.x and from.y > to.y);
    vec2 start = swapped ? to : from;
    vec2 end = swapped ? from : to;
    float value = (end.x - start.x) * (p.y - start.y) - (end.y - start.y) * (p.x - start.x);
    return swapped ? -value : value;
}

// The top-left rule for triangles that wind clockwise on the screen: a pixel centre on an
// edge belongs to the triangle whose edge runs up the screen or, horizontal, to the right.
inline bool ownsEdge(vec2 from, vec2 to) { return to.y < from.y or (to.y == from.y and to.x > from.x); }

inline bool covers(float weight, vec2 from, vec2 to, bool topLeft) {
    return weight > 0.0f or (weight == 0.0f and (not topLeft or ownsEdge(from, to)));
}

} // namespace lr::render::kernels

// The rasterizer of the visible triangles, with bindings LR_MATERIAL_TEXTURES(0u), then 12
// to 17 and uniforms 0 and 1, whose triangle stage (see largeTriangleArea()) is
// rasterizeTriangles(). The kernel adds coverPixel(pixel, depthBits, triangle), which keeps
// what it covers, and usesTopLeftRule(), whether a pixel centre on an edge two triangles
// share belongs to one of them only (or to both). Masked materials are cut out where their
// alpha falls below the cutoff, with the base colour texture sampled at level 0.
#define LR_TRIANGLE_RASTERIZER                                                                                         \
    LR_MATERIAL_TEXTURES(0u)                                                                                           \
    InstanceTable<12> instances;                                                                                       \
    VertexTable<13> vertices;                                                                                          \
    BufferBinding<uint, 14> indices;                                                                                   \
    MaterialTable<15> materials;                                                                                       \
    BufferBinding<uvec4, 16> visible;                                                                                  \
    BufferBinding<uint, 17> large; /* see largeTriangleWord() */                                                       \
    Uniform<mat4, 0> viewProjection{};                                                                                 \
    Uniform<ivec2, 1> size{};                                                                                          \
                                                                                                                       \
    LR_FIND_VISIBLE_SLOT                                                                                               \
                                                                                                                       \
    /* Pixels across, rows down from the top, and clip depth. */                                                       \
    vec3 toScreen(vec4 clip) {                                                                                         \
        float inverseW = 1.0f / clip.w;                                                                                \
        return vec3{(clip.x * inverseW * 0.5f + 0.5f) * float(size.value.x),                                           \
                    (0.5f - clip.y * inverseW * 0.5f) * float(size.value.y), clip.z * inverseW};                       \
    }                                                                                                                  \
                                                                                                                       \
    /* Whether a masked material keeps the point with perspective-correct weights `weights`. */                        \
    bool keepsMasked(GpuMaterial material, RasterVertex a, RasterVertex b, RasterVertex c, vec3 weights) {             \
        vec3 perspective = vec3{weights.x / a.clip.w, weights.y / b.clip.w, weights.z / c.clip.w};                     \
        vec2 uv = (a.uv * perspective.x + b.uv * perspective.y + c.uv * perspective.z)                                 \
                / (perspective.x + perspective.y + perspective.z);                                                     \
        float alpha = material.baseColor.w                                                                             \
                    * sampleTexture(material.textures.x, materialUv(material, uv), 0.0f, vec4(1.0f)).w;                \
        return alpha >= material.alphaCutoff;                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    /* Twice the signed area of the triangle on the screen. Rows run downwards, so a front                             \
       face (counter-clockwise in clip space) has a negative one, unless the instance's                                \
       transform mirrors, which reverses the winding. */                                                               \
    float screenArea(vec3 a, vec3 b, vec3 c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }         \
                                                                                                                       \
    bool culled(float area, bool doubleSided, bool mirrored) {                                                         \
        float facing = mirrored ? -area : area;                                                                        \
        return area == 0.0f or (facing > 0.0f and not doubleSided);                                                    \
    }                                                                                                                  \
                                                                                                                       \
    void rasterize(RasterVertex first, RasterVertex second, RasterVertex third, uint triangle, bool doubleSided,       \
                   bool mirrored, GpuMaterial material, RasterWindow window) {                                         \
        RasterVertex va = first;                                                                                       \
        RasterVertex vb = second;                                                                                      \
        RasterVertex vc = third;                                                                                       \
        vec3 a = toScreen(va.clip);                                                                                    \
        vec3 b = toScreen(vb.clip);                                                                                    \
        vec3 c = toScreen(vc.clip);                                                                                    \
        float area = screenArea(a, b, c);                                                                              \
        if (culled(area, doubleSided, mirrored)) {                                                                     \
            return;                                                                                                    \
        }                                                                                                              \
        if (area < 0.0f) {                                                                                             \
            vec3 swap = b;                                                                                             \
            b = c;                                                                                                     \
            c = swap;                                                                                                  \
            RasterVertex swapVertex = vb;                                                                              \
            vb = vc;                                                                                                   \
            vc = swapVertex;                                                                                           \
            area = -area;                                                                                              \
        }                                                                                                              \
        bool masked = (material.flags & materialAlphaMask()) != 0u;                                                    \
        bool topLeft = usesTopLeftRule();                                                                              \
        int x0 = windowStart(int(ceil(min(min(a.x, b.x), c.x) - 0.5f)), window.low.x, window.step.x);                  \
        int x1 = min(min(int(floor(max(max(a.x, b.x), c.x) - 0.5f)), window.high.x), size.value.x - 1);                \
        int y0 = windowStart(int(ceil(min(min(a.y, b.y), c.y) - 0.5f)), window.low.y, window.step.y);                  \
        int y1 = min(min(int(floor(max(max(a.y, b.y), c.y) - 0.5f)), window.high.y), size.value.y - 1);                \
        for (int y = y0; y <= y1; y += window.step.y) {                                                                \
            for (int x = x0; x <= x1; x += window.step.x) {                                                            \
                vec2 p = vec2{float(x) + 0.5f, float(y) + 0.5f};                                                       \
                float wa = edgeFunction(b.xy(), c.xy(), p);                                                            \
                float wb = edgeFunction(c.xy(), a.xy(), p);                                                            \
                float wc = edgeFunction(a.xy(), b.xy(), p);                                                            \
                if (covers(wa, b.xy(), c.xy(), topLeft) and covers(wb, c.xy(), a.xy(), topLeft)                        \
                    and covers(wc, a.xy(), b.xy(), topLeft)                                                            \
                    and (not masked or keepsMasked(material, va, vb, vc, vec3{wa, wb, wc}))) {                         \
                    float z = (wa * a.z + wb * b.z + wc * c.z) / area;                                                 \
                    coverPixel(uint(y * size.value.x + x), floatBitsToUint(max(z, 0.0f)), triangle);                   \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /* Lists the set-up triangle as large when it faces the viewer and its pixel box holds                             \
       more than largeTriangleArea() pixels, and the list has room; false otherwise. */                                \
    bool listsLarge(RasterTriangle set) {                                                                              \
        if (set.count < 3u) {                                                                                          \
            return false;                                                                                              \
        }                                                                                                              \
        vec3 a = toScreen(set.polygon[0].clip);                                                                        \
        if (culled(screenArea(a, toScreen(set.polygon[1].clip), toScreen(set.polygon[2].clip)), set.doubleSided,       \
                   set.mirrored)) {                                                                                    \
            return false;                                                                                              \
        }                                                                                                              \
        vec2 low = a.xy();                                                                                             \
        vec2 high = a.xy();                                                                                            \
        for (uint k = 1u; k < set.count; ++k) {                                                                        \
            vec2 corner = toScreen(set.polygon[k].clip).xy();                                                          \
            low = min(low, corner);                                                                                    \
            high = max(high, corner);                                                                                  \
        }                                                                                                              \
        int x0 = max(int(ceil(low.x - 0.5f)), 0);                                                                      \
        int x1 = min(int(floor(high.x - 0.5f)), size.value.x - 1);                                                     \
        int y0 = max(int(ceil(low.y - 0.5f)), 0);                                                                      \
        int y1 = min(int(floor(high.y - 0.5f)), size.value.y - 1);                                                     \
        if (x1 < x0 or y1 < y0 or uint(x1 - x0 + 1) * uint(y1 - y0 + 1) <= largeTriangleArea()) {                      \
            return false;                                                                                              \
        }                                                                                                              \
        uint slot = atomicAdd(large[0], 1u);                                                                           \
        if (slot >= largeTriangleCapacity()) {                                                                         \
            return false;                                                                                              \
        }                                                                                                              \
        uint side = rasterTileSize();                                                                                  \
        uvec2 firstTile = uvec2{uint(x0) / side, uint(y0) / side};                                                     \
        uvec2 tiles = uvec2{uint(x1) / side, uint(y1) / side} - firstTile + uvec2(1u);                                 \
        uint word = largeTriangleWord(slot);                                                                           \
        large[word] = set.triangle;                                                                                    \
        large[word + 1u] = firstTile.x | (firstTile.y << 16u);                                                         \
        large[word + 2u] = tiles.x | (tiles.y << 16u);                                                                 \
        return true;                                                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /* The visible triangle with its material, clipped against the near plane, clip z = 0,                             \
       behind which everything has z < 0. */                                                                           \
    RasterTriangle setUp(uint triangle) {                                                                              \
        uint slot = findSlot(triangle);                                                                                \
        uvec4 entry = visible[1u + slot];                                                                              \
        uint instance = entry.x;                                                                                       \
        RasterTriangle set;                                                                                            \
        set.triangle = triangle;                                                                                       \
        set.id = visible[visible[0].w + slot].x + triangle - entry.y;                                                  \
        set.material = materials[instances[instance].material];                                                        \
        set.doubleSided = (set.material.flags & materialDoubleSided()) != 0u;                                          \
        mat4 objectToWorld = instances[instance].objectToWorld.value;                                                  \
        set.mirrored = determinant(mat3(objectToWorld)) < 0.0f;                                                        \
        mat4 toClip = viewProjection.value * objectToWorld;                                                            \
        uint first = entry.z + 3u * (triangle - entry.y);                                                              \
        std::array<RasterVertex, 3> corners;                                                                           \
        for (uint k = 0u; k < 3u; ++k) {                                                                               \
            uint vertex = entry.w + indices[first + k];                                                                \
            corners[k] = RasterVertex{toClip * vec4(vertices[vertex].position, 1.0f), vertices[vertex].uv.value};      \
        }                                                                                                              \
        set.count = 0u;                                                                                                \
        for (uint k = 0u; k < 3u; ++k) {                                                                               \
            RasterVertex from = corners[k];                                                                            \
            RasterVertex to = corners[(k + 1u) % 3u];                                                                  \
            if (from.clip.z >= 0.0f) {                                                                                 \
                set.polygon[set.count] = from;                                                                         \
                set.count += 1u;                                                                                       \
            }                                                                                                          \
            if ((from.clip.z >= 0.0f) != (to.clip.z >= 0.0f)) {                                                        \
                float t = from.clip.z / (from.clip.z - to.clip.z);                                                     \
                set.polygon[set.count] = RasterVertex{mix(from.clip, to.clip, t), mix(from.uv, to.uv, t)};             \
                set.count += 1u;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return set;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    /* The set-up triangle as a fan from polygon[0], within `window`. */                                               \
    void rasterizeFan(RasterTriangle set, RasterWindow window) {                                                       \
        for (uint k = 2u; k < set.count; ++k) {                                                                        \
            rasterize(set.polygon[0], set.polygon[k - 1u], set.polygon[k], set.id, set.doubleSided,                    \
                      set.mirrored, set.material, window);                                                             \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /* The triangle stage. */                                                                                          \
    void rasterizeTriangles() {                                                                                        \
        uint triangle = gl_GlobalInvocationID.x;                                                                       \
        if (triangle >= visible[0].y) {                                                                                \
            return;                                                                                                    \
        }                                                                                                              \
        RasterTriangle set = setUp(triangle);                                                                          \
        if (not listsLarge(set)) {                                                                                     \
            rasterizeFan(set, RasterWindow{ivec2(0), size.value - ivec2(1), ivec2(1)});                                \
        }                                                                                                              \
    }

// The tile stage of LR_TRIANGLE_RASTERIZER, rasterizeTile(), for kernels that also expand
// it and declare usesBarrier: the workgroup's tile of the large triangle it falls to, which
// its first invocation sets up for all of them.
#define LR_TILE_RASTERIZER                                                                                             \
    Shared<RasterTriangle> tileTriangle;                                                                               \
                                                                                                                       \
    void rasterizeTile() {                                                                                             \
        uint tile = gl_WorkGroupID.y * rasterTilesPerRow() + gl_WorkGroupID.x;                                         \
        if (tile >= large[1]) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
        uint low = 0u;                                                                                                 \
        uint high = min(large[0], largeTriangleCapacity());                                                            \
        while (high - low > 1u) {                                                                                      \
            uint middle = (low + high) / 2u;                                                                           \
            if (large[largeTriangleWord(middle) + 3u] <= tile) {                                                       \
                low = middle;                                                                                          \
            } else {                                                                                                   \
                high = middle;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        uint word = largeTriangleWord(low);                                                                            \
        uint lane = gl_LocalInvocationIndex;                                                                           \
        if (lane == 0u) {                                                                                              \
            tileTriangle.value = setUp(large[word]);                                                                   \
        }                                                                                                              \
        barrier();                                                                                                     \
        uint firstTile = large[word + 1u];                                                                             \
        uint across = large[word + 2u] & 0xFFFFu;                                                                      \
        uint place = tile - large[word + 3u];                                                                          \
        int side = int(rasterTileSize());                                                                              \
        ivec2 corner = ivec2{int((firstTile & 0xFFFFu) + place % across), int((firstTile >> 16u) + place / across)}    \
                     * side;                                                                                           \
        RasterWindow window = RasterWindow{corner + ivec2{int(lane % rasterTileSize()), int(lane / rasterTileSize())}, \
                                           corner + ivec2(side - 1), ivec2{side, 4}};                                  \
        rasterizeFan(tileTriangle.value, window);                                                                      \
    }

// What RasterVisibility and RasterVisibilityTiles add to the rasterizer. Modes:
// rasterDepth() keeps the nearest depth per pixel in `depth`; rasterTriangles() the lowest
// visible triangle at that depth in `triangles`; rasterTransparent() pushes a fragment for
// every pixel in front of `depth` onto the pixel's list in `triangles` (see
// fragmentWords()), dropping it when fragmentCapacity are taken. The first two cover pixel
// centres on an edge in every triangle that has the edge; the third only in one of them,
// so that no pixel blends a surface twice.
#define LR_VISIBILITY_TARGETS                                                                                          \
    BufferBinding<uint, 18> depth;                                                                                     \
    BufferBinding<uint, 19> triangles;                                                                                 \
    Uniform<uint, 2> mode{};                                                                                           \
    Uniform<uint, 3> fragmentCapacity{};                                                                               \
                                                                                                                       \
    bool usesTopLeftRule() { return mode.value == rasterTransparent(); }                                               \
                                                                                                                       \
    void coverPixel(uint pixel, uint depthBits, uint triangle) {                                                       \
        if (mode.value == rasterDepth()) {                                                                             \
            atomicMin(depth[pixel], depthBits);                                                                        \
        } else if (mode.value == rasterTriangles()) {                                                                  \
            if (depth[pixel] == depthBits) {                                                                           \
                atomicMin(triangles[pixel], triangle);                                                                 \
            }                                                                                                          \
        } else if (depthBits < depth[pixel]) {                                                                         \
            uint pixels = uint(size.value.x * size.value.y);                                                           \
            uint fragment = atomicAdd(triangles[fragmentCount(pixels)], 1u);                                           \
            if (fragment < fragmentCapacity.value) {                                                                   \
                uint word = fragmentWords(pixels) + 3u * fragment;                                                     \
                triangles[word] = depthBits;                                                                           \
                triangles[word + 1u] = atomicExchange(triangles[pixel], fragment);                                     \
                triangles[word + 2u] = triangle;                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }

// What RasterPackedVisibility and RasterPackedVisibilityTiles add: per pixel the least
// floatBitsToUint(depth) << 32 | triangle, which is the nearest depth and the lowest
// triangle at it.
#define LR_PACKED_VISIBILITY_TARGETS                                                                                   \
    BufferBinding<u64, 18> packed;                                                                                     \
                                                                                                                       \
    bool usesTopLeftRule() { return false; }                                                                           \
                                                                                                                       \
    void coverPixel(uint pixel, uint depthBits, uint triangle) {                                                       \
        atomicMin(packed[pixel], (u64(depthBits) << 32u) | u64(triangle));                                             \
    }

namespace lr::render::kernels {

struct TC_KERNEL RasterVisibility {
    static constexpr char fileLocation[] = "lr_raster_visibility";
    uvec3 local_size{64, 1, 1};

    LR_TRIANGLE_RASTERIZER
    LR_VISIBILITY_TARGETS

    void main() { rasterizeTriangles(); }
};

struct TC_KERNEL RasterVisibilityTiles {
    static constexpr char fileLocation[] = "lr_raster_visibility_tiles";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{64, 1, 1};

    LR_TRIANGLE_RASTERIZER
    LR_TILE_RASTERIZER
    LR_VISIBILITY_TARGETS

    void main() { rasterizeTile(); }
};

// One workgroup: the number of the first tile of each large triangle a raster pass listed,
// their tiles in all, and the workgroup counts of the tile stage, one workgroup per tile.
struct TC_KERNEL ScanLargeTriangles {
    static constexpr char fileLocation[] = "lr_scan_large_triangles";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    BufferBinding<uint, 0> large;
    BufferBinding<DispatchIndirectCommand, 1> tileArguments;
    Shared<std::array<uint, 256>> tiles;
    Shared<uint> base;

    void main() {
        uint local = gl_LocalInvocationIndex;
        uint count = min(large[0], largeTriangleCapacity());
        if (local == 0u) {
            base.value = 0u;
        }
        barrier();
        for (uint chunk = 0u; chunk < count; chunk += 256u) {
            uint listed = chunk + local;
            uint own = 0u;
            if (listed < count) {
                uint extent = large[largeTriangleWord(listed) + 2u];
                own = (extent & 0xFFFFu) * (extent >> 16u);
            }
            tiles[local] = own;
            barrier();
            for (uint offset = 1u; offset < 256u; offset <<= 1u) {
                uint before = local >= offset ? tiles[local - offset] : 0u;
                barrier();
                tiles[local] += before;
                barrier();
            }
            if (listed < count) {
                large[largeTriangleWord(listed) + 3u] = base.value + tiles[local] - own;
            }
            barrier();
            if (local == 255u) {
                base.value += tiles[255];
            }
            barrier();
        }
        if (local == 0u) {
            uint total = base.value;
            large[1] = total;
            tileArguments[0].x = min(total, rasterTilesPerRow());
            tileArguments[0].y = (total + rasterTilesPerRow() - 1u) / rasterTilesPerRow();
            tileArguments[0].z = 1u;
        }
    }
};

inline u64 emptyPackedVisibility() { return 0xFFFFFFFFFFFFFFFFull; }

struct TC_KERNEL ClearPackedVisibility {
    static constexpr char fileLocation[] = "lr_clear_packed_visibility";
    uvec3 local_size{64, 1, 1};

    BufferBinding<u64, 0> packed;
    Uniform<uint, 0> count{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < count.value) {
            packed[i] = emptyPackedVisibility();
        }
    }
};

// The visibility buffer in one pass.
struct TC_KERNEL RasterPackedVisibility {
    static constexpr char fileLocation[] = "lr_raster_packed_visibility";
    uvec3 local_size{64, 1, 1};

    LR_TRIANGLE_RASTERIZER
    LR_PACKED_VISIBILITY_TARGETS

    void main() { rasterizeTriangles(); }
};

struct TC_KERNEL RasterPackedVisibilityTiles {
    static constexpr char fileLocation[] = "lr_raster_packed_visibility_tiles";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{64, 1, 1};

    LR_TRIANGLE_RASTERIZER
    LR_TILE_RASTERIZER
    LR_PACKED_VISIBILITY_TARGETS

    void main() { rasterizeTile(); }
};

// The packed visibility buffer as the two-pass rasterizer leaves it: a pixel nothing covers
// gets depth 0xFFFFFFFF and noTriangle().
struct TC_KERNEL UnpackVisibility {
    static constexpr char fileLocation[] = "lr_unpack_visibility";
    uvec3 local_size{64, 1, 1};

    BufferBinding<u64, 0> packed;
    BufferBinding<uint, 1> depth;
    BufferBinding<uint, 2> triangles;
    Uniform<uint, 0> count{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < count.value) {
            u64 value = packed[i];
            depth[i] = uint(value >> 32u);
            triangles[i] = uint(value & 0xFFFFFFFFull);
        }
    }
};

// Copies the counts in visible[0] to where the host reads them after the frame.
// totals[0] = (visible instances, visible clusters, their triangles, 0): those of `visible`,
// and of `late` too where there are `lists` 2.
struct TC_KERNEL CopyVisibleTotals {
    static constexpr char fileLocation[] = "lr_copy_visible_totals";
    uvec3 local_size{1, 1, 1};

    BufferBinding<uvec4, 0> instances;
    BufferBinding<uvec4, 1> visible;
    BufferBinding<uvec4, 2> late;
    BufferBinding<uvec4, 3> totals;
    Uniform<uint, 0> lists{};

    void main() {
        uvec2 drawn = visible[0].xy();
        if (lists.value == 2u) {
            drawn = drawn + late[0].xy();
        }
        totals[0] = uvec4{instances[0].x, drawn.x, drawn.y, 0u};
    }
};

} // namespace lr::render::kernels
