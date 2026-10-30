#pragma once

// The scene tables' read patterns, one kernel instance per candidate layout, for
// layout_probe to time on a device.

#include "kernels/Visibility.h"

#include <lr/render/GpuTypes.h>

namespace lr::probes {

using namespace tc;

// Every record padded to a multiple of 4 words, so that each starts 16-byte aligned and
// moves in uvec4 loads (tc's PaddedAos test layout): a 5-word vertex takes 8.
struct PaddedAos {
    static constexpr bool contiguousRecords = true;

    static constexpr std::size_t stride(std::uint32_t const recordWords) noexcept {
        return (recordWords + 3u) / 4u * 4u;
    }
    static constexpr std::size_t dataWords(std::size_t const count, std::uint32_t const recordWords) noexcept {
        return count * stride(recordWords);
    }
    static constexpr std::size_t fieldAddress(std::size_t const index, std::uint32_t const recordWords,
                                              std::uint32_t const fieldOffset, std::uint32_t /*fieldWords*/) noexcept {
        return index * stride(recordWords) + fieldOffset;
    }
    static constexpr std::uint32_t fieldAlignment(std::uint32_t /*recordWords*/, std::uint32_t const fieldOffset,
                                                  std::uint32_t /*fieldWords*/) noexcept {
        return fieldOffset % 4u == 0u ? 4u : fieldOffset % 2u == 0u ? 2u : 1u;
    }
};

// The culling pass: an instance's mesh and transform, the mesh's bounds.
template <class Layout>
struct TC_KERNEL CullReads {
    static constexpr char const *fileLocation = tc::instanceName<CullReads>();
    uvec3 local_size{64, 1, 1};

    TableBinding<render::GpuInstance, 0, Layout> instances;
    render::MeshTable<1> meshes;
    BufferBinding<uint, 2> results;
    Uniform<mat4, 0> viewProjection{};
    Uniform<uint, 1> count{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= count.value) {
            return;
        }
        uint meshIndex = instances[i].mesh;
        render::GpuMesh mesh = meshes[meshIndex];
        mat4 toClip = viewProjection.value * instances[i].objectToWorld.value;
        results[i] = render::kernels::outsideFrustum(toClip, mesh.boundsMin, mesh.boundsMax) ? 0u : mesh.triangleCount;
    }
};

TC_KERNEL_INSTANCE("lr_probe_cull_aos", CullReads<Aos>);
TC_KERNEL_INSTANCE("lr_probe_cull_soa", CullReads<Soa<>>);

// Rasterization and traversal: the positions of a triangle's corners.
template <class Layout>
struct TC_KERNEL PositionReads {
    static constexpr char const *fileLocation = tc::instanceName<PositionReads>();
    uvec3 local_size{64, 1, 1};

    TableBinding<render::GpuVertex, 0, Layout> vertices;
    BufferBinding<uint, 1> indices;
    BufferBinding<float, 2> results;
    Uniform<uint, 0> count{};

    void main() {
        uint t = gl_GlobalInvocationID.x;
        if (t >= count.value) {
            return;
        }
        vec3 a = vertices[indices[3u * t]].position;
        vec3 b = vertices[indices[3u * t + 1u]].position;
        vec3 c = vertices[indices[3u * t + 2u]].position;
        results[t] = length(cross(b - a, c - a));
    }
};

TC_KERNEL_INSTANCE("lr_probe_positions_aos", PositionReads<Aos>);
TC_KERNEL_INSTANCE("lr_probe_positions_soa", PositionReads<Soa<>>);
TC_KERNEL_INSTANCE("lr_probe_positions_padded", PositionReads<PaddedAos>);

// Shading: whole vertices.
template <class Layout>
struct TC_KERNEL VertexReads {
    static constexpr char const *fileLocation = tc::instanceName<VertexReads>();
    uvec3 local_size{64, 1, 1};

    TableBinding<render::GpuVertex, 0, Layout> vertices;
    BufferBinding<uint, 1> indices;
    BufferBinding<float, 2> results;
    Uniform<uint, 0> count{};

    void main() {
        uint t = gl_GlobalInvocationID.x;
        if (t >= count.value) {
            return;
        }
        float sum = 0.0f;
        for (uint k = 0u; k < 3u; ++k) {
            render::GpuVertex vertex = vertices[indices[3u * t + k]];
            sum += dot(vertex.position, vertex.normal.value) + vertex.uv.value.x + vertex.uv.value.y;
        }
        results[t] = sum;
    }
};

TC_KERNEL_INSTANCE("lr_probe_vertices_aos", VertexReads<Aos>);
TC_KERNEL_INSTANCE("lr_probe_vertices_soa", VertexReads<Soa<>>);
TC_KERNEL_INSTANCE("lr_probe_vertices_padded", VertexReads<PaddedAos>);

} // namespace lr::probes
