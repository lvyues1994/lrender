#pragma once

// The visibility buffer: cull instances, compact the visible ones in instance order, then
// rasterize their triangles twice, first keeping the nearest depth per pixel, then the
// lowest visible-triangle number among the triangles at that depth. The second pass runs
// the same program, so it computes bit-identical depths.

#include <lr/render/GpuTypes.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

struct TC_KERNEL ClearUint {
    static constexpr char fileLocation[] = "lr_clear_uint";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> target;
    Uniform<uint, 0> fill{};
    Uniform<uint, 1> count{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < count.value) {
            target[i] = fill.value;
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

// triangleCounts[i]: the triangles of instance i if it may be visible, else 0.
struct TC_KERNEL CullInstances {
    static constexpr char fileLocation[] = "lr_cull_instances";
    uvec3 local_size{64, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    BufferBinding<uint, 2> triangleCounts;
    Uniform<mat4, 0> viewProjection{};
    Uniform<uint, 1> instanceCount{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= instanceCount.value) {
            return;
        }
        uint meshIndex = instances[i].mesh;
        GpuMesh mesh = meshes[meshIndex];
        mat4 toClip = viewProjection.value * instances[i].objectToWorld.value;
        if (outsideFrustum(toClip, mesh.boundsMin, mesh.boundsMax)) {
            triangleCounts[i] = 0u;
        } else {
            triangleCounts[i] = mesh.triangleCount;
        }
    }
};

// One workgroup: visible[0] = (visible instances, visible triangles), visible[1 + k] =
// (instance, first visible triangle) for the k-th visible instance in instance order, and
// the workgroup counts of the raster passes, one invocation per visible triangle.
struct TC_KERNEL CompactVisible {
    static constexpr char fileLocation[] = "lr_compact_visible";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    BufferBinding<uint, 0> triangleCounts;
    BufferBinding<uvec2, 1> visible;
    BufferBinding<DispatchIndirectCommand, 2> rasterArguments;
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
                visible[1u + slot] = uvec2{instance, triangleBase.value + triangles[local] - count};
            }
            barrier();
            if (local == 255u) {
                slotBase.value += slots[255];
                triangleBase.value += triangles[255];
            }
            barrier();
        }
        if (local == 0u) {
            visible[0] = uvec2{slotBase.value, triangleBase.value};
            rasterArguments[0].x = (triangleBase.value + 63u) / 64u;
            rasterArguments[0].y = 1u;
            rasterArguments[0].z = 1u;
        }
    }
};

struct TC_KERNEL RasterVisibility {
    static constexpr char fileLocation[] = "lr_raster_visibility";
    uvec3 local_size{64, 1, 1};

    InstanceTable<0> instances;
    MeshTable<1> meshes;
    VertexTable<2> vertices;
    BufferBinding<uint, 3> indices;
    MaterialTable<4> materials;
    BufferBinding<uvec2, 5> visible;
    BufferBinding<uint, 6> depth;
    BufferBinding<uint, 7> triangles;
    Uniform<mat4, 0> viewProjection{};
    Uniform<ivec2, 1> size{};
    Uniform<uint, 2> writeTriangles{}; // 0: the depth pass, 1: the triangle pass

    // The visible instance whose triangles include visible triangle `triangle`.
    uint findSlot(uint triangle) {
        uint low = 0u;
        uint high = visible[0].x;
        while (high - low > 1u) {
            uint middle = (low + high) / 2u;
            if (visible[1u + middle].y <= triangle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }

    // Pixels across, rows down from the top, and clip depth.
    vec3 toScreen(vec4 clip) {
        float inverseW = 1.0f / clip.w;
        return vec3{(clip.x * inverseW * 0.5f + 0.5f) * float(size.value.x),
                    (0.5f - clip.y * inverseW * 0.5f) * float(size.value.y), clip.z * inverseW};
    }

    void coverPixel(uint pixel, uint depthBits, uint triangle) {
        if (writeTriangles.value == 0u) {
            atomicMin(depth[pixel], depthBits);
        } else if (depth[pixel] == depthBits) {
            atomicMin(triangles[pixel], triangle);
        }
    }

    // Covers the pixels whose centres lie inside the triangle (edges included).
    void rasterize(vec4 clipA, vec4 clipB, vec4 clipC, uint triangle, bool doubleSided) {
        vec3 a = toScreen(clipA);
        vec3 b = toScreen(clipB);
        vec3 c = toScreen(clipC);
        float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        // Rows run downwards, so a front face (counter-clockwise in clip space) has negative area.
        if (area == 0.0f or (area > 0.0f and not doubleSided)) {
            return;
        }
        if (area < 0.0f) {
            vec3 swap = b;
            b = c;
            c = swap;
            area = -area;
        }
        int x0 = max(int(ceil(min(min(a.x, b.x), c.x) - 0.5f)), 0);
        int x1 = min(int(floor(max(max(a.x, b.x), c.x) - 0.5f)), size.value.x - 1);
        int y0 = max(int(ceil(min(min(a.y, b.y), c.y) - 0.5f)), 0);
        int y1 = min(int(floor(max(max(a.y, b.y), c.y) - 0.5f)), size.value.y - 1);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                float px = float(x) + 0.5f;
                float py = float(y) + 0.5f;
                float wa = (c.x - b.x) * (py - b.y) - (c.y - b.y) * (px - b.x);
                float wb = (a.x - c.x) * (py - c.y) - (a.y - c.y) * (px - c.x);
                float wc = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
                if (wa >= 0.0f and wb >= 0.0f and wc >= 0.0f) {
                    float z = (wa * a.z + wb * b.z + wc * c.z) / area;
                    coverPixel(uint(y * size.value.x + x), floatBitsToUint(max(z, 0.0f)), triangle);
                }
            }
        }
    }

    void main() {
        uint triangle = gl_GlobalInvocationID.x;
        if (triangle >= visible[0].y) {
            return;
        }
        uint slot = findSlot(triangle);
        uvec2 entry = visible[1u + slot];
        uint instance = entry.x;
        uint meshIndex = instances[instance].mesh;
        uint materialIndex = instances[instance].material;
        GpuMesh mesh = meshes[meshIndex];
        bool doubleSided = (materials[materialIndex].flags & materialDoubleSided()) != 0u;
        mat4 toClip = viewProjection.value * instances[instance].objectToWorld.value;
        uint first = mesh.firstIndex + 3u * (triangle - entry.y);
        std::array<vec4, 3> corners;
        for (uint k = 0u; k < 3u; ++k) {
            uint vertex = mesh.firstVertex + indices[first + k];
            corners[k] = toClip * vec4(vertices[vertex].position, 1.0f);
        }
        // Clip against the near plane, clip z = 0; everything behind it has z < 0.
        std::array<vec4, 4> polygon;
        uint count = 0u;
        for (uint k = 0u; k < 3u; ++k) {
            vec4 from = corners[k];
            vec4 to = corners[(k + 1u) % 3u];
            if (from.z >= 0.0f) {
                polygon[count] = from;
                count += 1u;
            }
            if ((from.z >= 0.0f) != (to.z >= 0.0f)) {
                polygon[count] = mix(from, to, from.z / (from.z - to.z));
                count += 1u;
            }
        }
        for (uint k = 2u; k < count; ++k) {
            rasterize(polygon[0], polygon[k - 1u], polygon[k], triangle, doubleSided);
        }
    }
};

// Copies visible[0] to where the host reads it after the frame.
struct TC_KERNEL CopyVisibleTotals {
    static constexpr char fileLocation[] = "lr_copy_visible_totals";
    uvec3 local_size{1, 1, 1};

    BufferBinding<uvec2, 0> visible;
    BufferBinding<uvec2, 1> totals;

    void main() { totals[0] = visible[0]; }
};

} // namespace lr::render::kernels
