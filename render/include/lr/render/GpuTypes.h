#pragma once

// The data kernels share with the host, written once in C++ and transpiled with the
// kernels. Everything here must stay within what compute_shaders' transpiler accepts.

#include <tc/tc.hpp>

namespace lr::render {

// Table layouts, chosen with probes/layout_probe; host tables and kernel bindings must use
// the same one. Device times on an RTX 5070 Ti (Vulkan) with tables well past its cache,
// 4M instances and 8.4M vertices: culling reads an instance's mesh and transform, 0.49 ms
// as Aos and 0.27 ms as Soa<>; rasterization and traversal read corner positions, 0.54 ms
// as Aos (5 words), 0.47 ms as Soa<> and 0.66 ms padded to 8 words; shading reads whole
// corners, 0.53, 0.53 and 0.66 ms. On lavapipe Aos reads positions some 15% faster; on the
// CPU backend the layouts do not matter.
using VertexLayout = tc::Soa<>;
using InstanceLayout = tc::Soa<>;

struct GpuVertex {
    tc::vec3 position;
    tc::Encoded<tc::vec3, tc::OctahedralNormal> normal;
    tc::Encoded<tc::vec2, tc::Half2> uv;
};

// One mesh's triangles: indices[firstIndex + 3 t + k] + firstVertex is corner k of triangle t.
struct GpuMesh {
    tc::uint firstVertex;
    tc::uint firstIndex;
    tc::uint triangleCount;
    tc::vec3 boundsMin;
    tc::vec3 boundsMax;
};

struct GpuInstance {
    tc::Encoded<tc::mat4, tc::Affine3x4> objectToWorld;
    tc::Encoded<tc::mat4, tc::Affine3x4> worldToObject;
    tc::uint mesh;
    tc::uint material;
};

struct GpuMaterial {
    tc::vec4 baseColor;
    tc::vec3 emissive;
    float metallic;
    float roughness;
    float alphaCutoff;
    tc::uint flags; // materialDoubleSided()
};

struct GpuLight {
    tc::uint kind;        // lightDirectional() or lightPoint()
    tc::vec3 radiance;    // color times illuminance (directional) or intensity (point)
    tc::vec3 vector;      // direction the light travels, or the point's position
    float range;          // point lights: fades out to nothing here; 0 for no limit
};

// A node of a bounding volume hierarchy. An inner node's children are the nodes `first`
// and `first + 1`; a leaf holds `count` primitives from `first`.
struct GpuBvhNode {
    tc::vec3 boundsMin;
    tc::uint first;
    tc::vec3 boundsMax;
    tc::uint count; // 0 for inner nodes
};

// A triangle of a mesh's hierarchy: its corners' places in the vertex table.
struct GpuBvhTriangle {
    tc::uvec3 corners;
};

// Constants as functions: the transpiler turns free functions into GLSL functions.
inline tc::uint materialDoubleSided() { return 1u; }
inline tc::uint lightDirectional() { return 0u; }
inline tc::uint lightPoint() { return 1u; }
// The visibility buffer of a pixel no triangle covers.
inline tc::uint noTriangle() { return 0xFFFFFFFFu; }
// The hierarchy root of a mesh without triangles.
inline tc::uint noBvhNode() { return 0xFFFFFFFFu; }
// Hierarchies are at most this many levels deep below the root, so traversal fits in a
// stack of bvhMaxDepth() + 1 entries.
inline constexpr tc::uint bvhMaxDepth() { return 31u; }

template <tc::uint B>
using VertexTable = tc::TableBinding<GpuVertex, B, VertexLayout>;
template <tc::uint B>
using MeshTable = tc::TableBinding<GpuMesh, B>;
template <tc::uint B>
using InstanceTable = tc::TableBinding<GpuInstance, B, InstanceLayout>;
template <tc::uint B>
using MaterialTable = tc::TableBinding<GpuMaterial, B>;
template <tc::uint B>
using LightTable = tc::TableBinding<GpuLight, B>;
template <tc::uint B>
using BvhNodeTable = tc::TableBinding<GpuBvhNode, B>;
template <tc::uint B>
using BvhTriangleTable = tc::TableBinding<GpuBvhTriangle, B>;

// HDR colour between shading and the final pass.
using HdrFormat = tc::gpu::RGBA16F;
using HdrPixel = typename tc::GPUFormatTraits<HdrFormat>::StoragePixel;
// The path tracer's running mean of the samples of each pixel.
using AccumulationFormat = tc::gpu::RGBA32F;
using AccumulationPixel = typename tc::GPUFormatTraits<AccumulationFormat>::StoragePixel;

// The scene as the GPU thread holds it, for passes to bind (attach() takes non-const
// resources; passes do not change them). Counts are the live elements; the tables may be
// larger.
struct SceneTables {
    tc::TableResource<GpuVertex, VertexLayout> *vertices = nullptr;
    tc::BufferResource<tc::uint> *indices = nullptr;
    tc::TableResource<GpuMesh> *meshes = nullptr;
    tc::TableResource<GpuInstance, InstanceLayout> *instances = nullptr;
    tc::TableResource<GpuMaterial> *materials = nullptr;
    tc::TableResource<GpuLight> *lights = nullptr;
    tc::uint instanceCount = 0;
    tc::uint lightCount = 0;

    // Two levels of hierarchies, null until a pipeline that traces rays has run (see
    // SceneRequirements): one per mesh over its triangles, rooted at blasRoots[mesh] in
    // blasNodes with leaves indexing blasTriangles, and one over the instances, rooted at
    // tlasNodes[0] with one instance index per leaf.
    tc::TableResource<GpuBvhNode> *blasNodes = nullptr;
    tc::TableResource<GpuBvhTriangle> *blasTriangles = nullptr;
    tc::BufferResource<tc::uint> *blasRoots = nullptr;
    tc::TableResource<GpuBvhNode> *tlasNodes = nullptr;
    tc::uint tlasNodeCount = 0;
};

} // namespace lr::render
