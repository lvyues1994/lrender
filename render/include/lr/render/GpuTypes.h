#pragma once

// The data kernels share with the host, written once in C++ and transpiled with the
// kernels. Everything here must stay within what compute_shaders' transpiler accepts.

#include <tc/tc.hpp>

#include <array>
#include <cstdint>

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
    tc::uint vertexCount;  // from firstVertex
    tc::uint firstCluster; // its clusters, in the cluster table
    tc::uint clusterCount;
};

// A run of a mesh's triangles that culling takes as one: triangles [firstTriangle,
// firstTriangle + triangleCount) of the mesh, at most clusterMaxTriangles() of them over at
// most clusterMaxVertices() vertices. In the mesh's space: the box around them, and, to
// cull a run whose triangles all face away (meshoptimizer's test), a sphere around them and
// a cone of their normals. A viewer at p sees none of them when dot(center - p, coneAxis)
// >= coneCutoff * length(center - p) + radius, which a cutoff of 1 never meets.
struct GpuCluster {
    tc::vec3 boundsMin;
    tc::uint firstTriangle;
    tc::vec3 boundsMax;
    tc::uint triangleCount;
    tc::vec3 center;
    float radius;
    tc::vec3 coneAxis;
    float coneCutoff;
};

inline tc::uint clusterMaxTriangles() { return 124u; }
inline tc::uint clusterMaxVertices() { return 64u; }

struct GpuInstance {
    tc::Encoded<tc::mat4, tc::Affine3x4> objectToWorld;
    tc::Encoded<tc::mat4, tc::Affine3x4> worldToObject;
    tc::uint mesh;
    tc::uint material;
    // Where the previous frame had the instance, for motion vectors; objectToWorld when it
    // did not move or is new.
    tc::Encoded<tc::mat4, tc::Affine3x4> previousObjectToWorld;
};

struct GpuMaterial {
    tc::vec4 baseColor;
    tc::vec3 emissive;
    float metallic;
    float roughness;
    float alphaCutoff;
    tc::uint flags; // materialDoubleSided(), materialAlphaMask(), materialAlphaBlend()
    float normalScale;
    // Base colour, metallic-roughness, emissive and normal: textureRef() of each, or
    // noTexture() while it is absent.
    tc::uvec4 textures;
    // The texture coordinates every texture reads: (dot(uvRow0, (u, v, 1)), dot(uvRow1, ...)).
    tc::vec3 uvRow0;
    tc::vec3 uvRow1;
    tc::uint model; // the material model that shades it; 0 is metallic-roughness
    tc::vec4 parameters; // MaterialDesc::modelParameters
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

// A triangle of a mesh's hierarchy: its corners' places in the vertex table, and which of
// the mesh's triangles it is.
struct GpuBvhTriangle {
    tc::uvec3 corners;
    tc::uint primitive;
};

// Constants as functions: the transpiler turns free functions into GLSL functions.
inline tc::uint materialDoubleSided() { return 1u; }
inline tc::uint materialAlphaMask() { return 2u; }
inline tc::uint materialAlphaBlend() { return 4u; }
inline tc::uint lightDirectional() { return 0u; }
inline tc::uint lightPoint() { return 1u; }
// The visibility buffer of a pixel no triangle covers.
inline tc::uint noTriangle() { return 0xFFFFFFFFu; }
// The hierarchy root of a mesh without triangles.
inline tc::uint noBvhNode() { return 0xFFFFFFFFu; }
// Hierarchies are at most this many levels deep below the root, so traversal fits in a
// stack of bvhMaxDepth() + 1 entries.
inline constexpr tc::uint bvhMaxDepth() { return 31u; }
// The masks of ray-query instances: rays trace rayMaskScene(); instances of meshes without
// triangles refer to a placeholder bottom level with rayMaskGap(), which no ray traces.
inline tc::uint rayMaskScene() { return 1u; }
inline tc::uint rayMaskGap() { return 2u; }
// TopLevelInstance flags (VkGeometryInstanceFlagBitsKHR).
inline tc::uint instanceCullDisable() { return 1u; }
inline tc::uint instanceFlipFacing() { return 2u; }
inline tc::uint instanceForceOpaque() { return 4u; }
inline tc::uint instanceForceNoOpaque() { return 8u; }

// Textures live in pools, 2D array images of one format and one square power-of-two size
// with full mip chains, sampled through a fixed set of bindings: colorPoolSlots() RGBA8
// pools (texels in the texture's own encoding, sRGB or linear) and compressedPoolSlots()
// pools of block-compressed formats.
inline constexpr tc::uint colorPoolSlots() { return 8u; }
inline constexpr tc::uint compressedPoolSlots() { return 4u; }
inline constexpr tc::uint texturePoolSlots() { return colorPoolSlots() + compressedPoolSlots(); }
inline tc::uint wrapRepeat() { return 0u; }
inline tc::uint wrapClampToEdge() { return 1u; }
inline tc::uint wrapMirroredRepeat() { return 2u; }
inline tc::uint noTexture() { return 0xFFFFFFFFu; }

// Where a texture is: bits 0-3 its pool slot, 4-15 its layer, 16-19 log2 of its size, 20
// whether its texels are sRGB-encoded, 21-22 and 23-24 how it wraps in u and in v.
inline tc::uint textureRef(tc::uint slot, tc::uint layer, tc::uint sizeLog2, bool srgb, tc::uint wrapU,
                           tc::uint wrapV) {
    return slot | layer << 4u | sizeLog2 << 16u | (srgb ? 1u : 0u) << 20u | wrapU << 21u | wrapV << 23u;
}
inline tc::uint textureSlot(tc::uint ref) { return ref & 15u; }
inline tc::uint textureLayer(tc::uint ref) { return (ref >> 4u) & 4095u; }
inline tc::uint textureSizeLog2(tc::uint ref) { return (ref >> 16u) & 15u; }
inline bool textureIsSrgb(tc::uint ref) { return ((ref >> 20u) & 1u) != 0u; }
inline tc::uint textureWrapU(tc::uint ref) { return (ref >> 21u) & 3u; }
inline tc::uint textureWrapV(tc::uint ref) { return (ref >> 23u) & 3u; }

// The shadow map buffer: a header of shadowHeaderWords() words, then per cascade
// resolution^2 depths (floatBitsToUint of the nearest depth, 0xFFFFFFFF where nothing
// was drawn). Header words: [0] cascades, [1] resolution, then per cascade c from
// shadowCascadeWord(c): its world-to-shadow matrix (16 floats, column-major, mapping to
// texture coordinates in x and y and depth in z) and the world size of one of its texels.
inline constexpr tc::uint maxShadowCascades() { return 4u; }
inline constexpr tc::uint shadowHeaderWords() { return 128u; }
inline tc::uint shadowCascadeWord(tc::uint cascade) { return 8u + 20u * cascade; }

// Material models a renderer holds at most, the built-in one included.
inline constexpr tc::uint maxMaterialModels() { return 16u; }
// The pixels each material model shades, in one buffer of uvec2: entry m < maxMaterialModels()
// is (pixels of model m, where its list starts), and entry maxMaterialModels() + start + i
// is (pixel index, visible triangle) of the model's i-th pixel.
inline tc::uint modelListEntry(tc::uint start, tc::uint i) { return maxMaterialModels() + start + i; }

// The transparent fragment lists of a view of `pixels` pixels, in one buffer of words: the
// head of each pixel's list (noFragment() for none), the number of fragments pushed, then
// from fragmentWords(pixels) three words per fragment: floatBitsToUint of its depth, the
// next fragment of its pixel, its visible triangle.
inline tc::uint noFragment() { return 0xFFFFFFFFu; }
inline tc::uint fragmentCount(tc::uint pixels) { return pixels; }
inline tc::uint fragmentWords(tc::uint pixels) { return (pixels + 4u) & ~3u; }

template <tc::uint B>
using VertexTable = tc::TableBinding<GpuVertex, B, VertexLayout>;
template <tc::uint B>
using MeshTable = tc::TableBinding<GpuMesh, B>;
template <tc::uint B>
using ClusterTable = tc::TableBinding<GpuCluster, B>;
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

using ColorPoolFormat = tc::gpu::RGBA8;
using ColorPoolPixel = tc::cpu::RGBA8;
using ColorPoolImage = tc::DeviceImage<ColorPoolFormat, tc::Dim::D2Array, ColorPoolPixel>;
using CompressedPoolImage = tc::DeviceCompressedImage<tc::Dim::D2Array>;
// What a GGX lobe with Schlick's Fresnel reflects of a uniform environment, F0 A + B, as
// (A, B) over (n . v, roughness) on texel centres.
using EnvironmentBrdfFormat = tc::gpu::RG16F;
using EnvironmentBrdfPixel = tc::cpu::RG16F;
using EnvironmentBrdfImage = tc::DeviceImage<EnvironmentBrdfFormat, tc::Dim::D2, EnvironmentBrdfPixel>;
inline constexpr tc::uint environmentBrdfSize() { return 32u; }

// What kernels that sample materials bind besides the tables. Slots without a pool hold a
// small image nothing reads.
struct MaterialTextures {
    std::array<ColorPoolImage const *, colorPoolSlots()> colorPools{};
    std::array<CompressedPoolImage const *, compressedPoolSlots()> compressedPools{};
    EnvironmentBrdfImage const *environmentBrdf = nullptr;
    // Changes whenever textures finish streaming in or go away.
    std::uint64_t residency = 0;
};

// The scene as the GPU thread holds it, for passes to bind (attach() takes non-const
// resources; passes do not change them). Counts are the live elements; the tables may be
// larger.
struct SceneTables {
    tc::TableResource<GpuVertex, VertexLayout> *vertices = nullptr;
    tc::BufferResource<tc::uint> *indices = nullptr;
    tc::TableResource<GpuMesh> *meshes = nullptr;
    tc::TableResource<GpuCluster> *clusters = nullptr;
    tc::TableResource<GpuInstance, InstanceLayout> *instances = nullptr;
    tc::TableResource<GpuMaterial> *materials = nullptr;
    tc::TableResource<GpuLight> *lights = nullptr;
    tc::uint instanceCount = 0;
    tc::uint lightCount = 0;

    // Without ray queries, two levels of hierarchies, null until a pipeline that traces rays
    // has run (see FeatureRequirements): one per mesh over its triangles, rooted at blasRoots[mesh] in
    // blasNodes with leaves indexing blasTriangles, and one over the instances, rooted at
    // tlasNodes[0] with one instance index per leaf.
    tc::TableResource<GpuBvhNode> *blasNodes = nullptr;
    tc::TableResource<GpuBvhTriangle> *blasTriangles = nullptr;
    tc::BufferResource<tc::uint> *blasRoots = nullptr;
    tc::TableResource<GpuBvhNode> *tlasNodes = nullptr;
    tc::uint tlasNodeCount = 0;

    // With ray queries (RendererCapabilities::rayQueries) instead: the device's acceleration
    // structure over the instances, null until a pipeline that traces rays has run. Its
    // instances' custom indices are their places in the instance table, and the primitive
    // index of a hit is the triangle of the instance's mesh.
    tc::TopLevelAS const *topLevel = nullptr;

    MaterialTextures textures;
};

} // namespace lr::render
