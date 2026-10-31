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

// triangleCounts[i]: the triangles of instance i if it may be visible and its material
// blends (`blended` 1) or does not (`blended` 0), else 0.
struct TC_KERNEL CullInstances {
    static constexpr char fileLocation[] = "lr_cull_instances";
    uvec3 local_size{64, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    MaterialTable<2> materials;
    BufferBinding<uint, 3> triangleCounts;
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
            triangleCounts[i] = 0u;
        } else {
            triangleCounts[i] = mesh.triangleCount;
        }
    }
};

inline uint compactGroupSize() { return 256u; }

// One workgroup: visible[0] = (visible instances, visible triangles, 0, 0), visible[1 + k]
// = (instance, first visible triangle, first index, first vertex of its mesh) for the
// k-th visible instance in instance order, and the workgroup counts of the raster passes,
// one invocation per visible triangle. Scans each chunk of 256 instances in shared memory.
struct TC_KERNEL CompactVisible {
    static constexpr char fileLocation[] = "lr_compact_visible";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    BufferBinding<uint, 2> triangleCounts;
    BufferBinding<uvec4, 3> visible;
    BufferBinding<DispatchIndirectCommand, 4> rasterArguments;
    Uniform<uint, 0> instanceCount{};
    Shared<std::array<uint, 256>> slots;
    Shared<std::array<uint, 256>> triangles;
    Shared<uint> slotBase;
    Shared<uint> triangleBase;

    void main() {
        uint local = gl_LocalInvocationIndex;
        if (local == 0u) {
            slotBase.value = 0u;
            triangleBase.value = 0u;
        }
        barrier();
        for (uint chunk = 0u; chunk < instanceCount.value; chunk += 256u) {
            uint instance = chunk + local;
            uint count = instance < instanceCount.value ? triangleCounts[instance] : 0u;
            slots[local] = count > 0u ? 1u : 0u;
            triangles[local] = count;
            barrier();
            for (uint offset = 1u; offset < 256u; offset <<= 1u) {
                uint slotsBefore = local >= offset ? slots[local - offset] : 0u;
                uint trianglesBefore = local >= offset ? triangles[local - offset] : 0u;
                barrier();
                slots[local] += slotsBefore;
                triangles[local] += trianglesBefore;
                barrier();
            }
            if (count > 0u) {
                uint slot = slotBase.value + slots[local] - 1u;
                GpuMesh mesh = meshes[instances[instance].mesh];
                visible[1u + slot] =
                    uvec4{instance, triangleBase.value + triangles[local] - count, mesh.firstIndex, mesh.firstVertex};
            }
            barrier();
            if (local == 255u) {
                slotBase.value += slots[255];
                triangleBase.value += triangles[255];
            }
            barrier();
        }
        if (local == 0u) {
            visible[0] = uvec4{slotBase.value, triangleBase.value, 0u, 0u};
            rasterArguments[0].x = (triangleBase.value + 63u) / 64u;
            rasterArguments[0].y = 1u;
            rasterArguments[0].z = 1u;
        }
    }
};

// CompactVisible's outputs, scanning within subgroups; shared memory only holds the
// subgroups' totals, of which each invocation adds up those before its own.
struct TC_KERNEL CompactVisibleSubgroups {
    static constexpr char fileLocation[] = "lr_compact_visible_subgroups";
    static constexpr bool usesBarrier = true;
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{256, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    BufferBinding<uint, 2> triangleCounts;
    BufferBinding<uvec4, 3> visible;
    BufferBinding<DispatchIndirectCommand, 4> rasterArguments;
    Uniform<uint, 0> instanceCount{};
    // Per subgroup, of which there are at most 256 (of one invocation each).
    Shared<std::array<uint, 256>> slotTotals;
    Shared<std::array<uint, 256>> triangleTotals;
    Shared<uint> slotBase;
    Shared<uint> triangleBase;

    void main() {
        uint local = gl_LocalInvocationIndex;
        if (local == 0u) {
            slotBase.value = 0u;
            triangleBase.value = 0u;
        }
        barrier();
        for (uint chunk = 0u; chunk < instanceCount.value; chunk += 256u) {
            uint instance = chunk + local;
            uint count = instance < instanceCount.value ? triangleCounts[instance] : 0u;
            uint flag = count > 0u ? 1u : 0u;
            uint slotsBefore = subgroupExclusiveAdd(flag);
            uint trianglesBefore = subgroupExclusiveAdd(count);
            uint slotSum = subgroupAdd(flag);
            uint triangleSum = subgroupAdd(count);
            if (subgroupElect()) {
                slotTotals[gl_SubgroupID] = slotSum;
                triangleTotals[gl_SubgroupID] = triangleSum;
            }
            barrier();
            for (uint s = 0u; s < gl_SubgroupID; ++s) {
                slotsBefore += slotTotals[s];
                trianglesBefore += triangleTotals[s];
            }
            if (count > 0u) {
                GpuMesh mesh = meshes[instances[instance].mesh];
                visible[1u + slotBase.value + slotsBefore] =
                    uvec4{instance, triangleBase.value + trianglesBefore, mesh.firstIndex, mesh.firstVertex};
            }
            barrier();
            if (local == 0u) {
                for (uint s = 0u; s < gl_NumSubgroups; ++s) {
                    slotBase.value += slotTotals[s];
                    triangleBase.value += triangleTotals[s];
                }
            }
            barrier();
        }
        if (local == 0u) {
            visible[0] = uvec4{slotBase.value, triangleBase.value, 0u, 0u};
            rasterArguments[0].x = (triangleBase.value + 63u) / 64u;
            rasterArguments[0].y = 1u;
            rasterArguments[0].z = 1u;
        }
    }
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
// count - 2 triangles from polygon[0].
struct RasterTriangle {
    std::array<RasterVertex, 4> polygon;
    uint count;
    uint triangle;
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
            rasterize(set.polygon[0], set.polygon[k - 1u], set.polygon[k], set.triangle, set.doubleSided,              \
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
struct TC_KERNEL CopyVisibleTotals {
    static constexpr char fileLocation[] = "lr_copy_visible_totals";
    uvec3 local_size{1, 1, 1};

    BufferBinding<uvec4, 0> visible;
    BufferBinding<uvec2, 1> totals;

    void main() { totals[0] = visible[0].xy(); }
};

} // namespace lr::render::kernels
