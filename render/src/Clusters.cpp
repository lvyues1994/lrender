#include "Clusters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace lr::render {

namespace {

// Below this the normals spread too far for the cone to cull anything (meshoptimizer's
// threshold).
constexpr auto kNarrowestSpread = 0.1f;

std::uint32_t corner(core::MeshData const &mesh, std::size_t const triangle, std::size_t const k) {
    return mesh.indices[triangle * 3 + k];
}

GpuCluster clusterOf(core::MeshData const &mesh, std::size_t const first, std::size_t const end) {
    auto low = core::vec3{std::numeric_limits<float>::max()};
    auto high = core::vec3{std::numeric_limits<float>::lowest()};
    auto normals = core::vec3{0.0f};
    for (auto triangle = first; triangle < end; ++triangle) {
        auto const &a = mesh.positions[corner(mesh, triangle, 0)];
        auto const &b = mesh.positions[corner(mesh, triangle, 1)];
        auto const &c = mesh.positions[corner(mesh, triangle, 2)];
        low = tc::min(low, tc::min(a, tc::min(b, c)));
        high = tc::max(high, tc::max(a, tc::max(b, c)));
        auto const normal = tc::cross(b - a, c - a);
        auto const length = tc::length(normal);
        if (length > 0.0f) {
            normals = normals + normal / length;
        }
    }
    auto const center = (low + high) * 0.5f;
    auto radius = 0.0f;
    auto spread = 1.0f; // the least cosine between the axis and a normal
    auto const axisLength = tc::length(normals);
    auto const axis = axisLength > 0.0f ? normals / axisLength : core::vec3{0.0f, 0.0f, 1.0f};
    for (auto triangle = first; triangle < end; ++triangle) {
        auto const &a = mesh.positions[corner(mesh, triangle, 0)];
        auto const &b = mesh.positions[corner(mesh, triangle, 1)];
        auto const &c = mesh.positions[corner(mesh, triangle, 2)];
        radius = std::max({radius, tc::length(a - center), tc::length(b - center), tc::length(c - center)});
        auto const normal = tc::cross(b - a, c - a);
        auto const length = tc::length(normal);
        if (length > 0.0f) {
            spread = std::min(spread, tc::dot(axis, normal / length));
        }
    }
    auto const cutoff = axisLength > 0.0f and spread > kNarrowestSpread ? std::sqrt(1.0f - spread * spread) : 1.0f;
    return GpuCluster{low,    static_cast<tc::uint>(first),
                      high,   static_cast<tc::uint>(end - first),
                      center, radius,
                      axis,   cutoff};
}

} // namespace

std::vector<GpuCluster> buildClusters(core::MeshData const &mesh) {
    auto clusters = std::vector<GpuCluster>{};
    auto const triangles = mesh.indices.size() / 3;
    // The cluster that last counted each vertex.
    auto counted = std::vector<std::uint32_t>(mesh.positions.size(), std::numeric_limits<std::uint32_t>::max());
    auto first = std::size_t{0};
    while (first < triangles) {
        auto const id = static_cast<std::uint32_t>(clusters.size());
        auto vertices = std::uint32_t{0};
        auto end = first;
        while (end < triangles and end - first < clusterMaxTriangles()) {
            auto const a = corner(mesh, end, 0);
            auto const b = corner(mesh, end, 1);
            auto const c = corner(mesh, end, 2);
            auto const added = (counted[a] != id ? 1u : 0u) + (counted[b] != id and b != a ? 1u : 0u)
                             + (counted[c] != id and c != a and c != b ? 1u : 0u);
            if (vertices + added > clusterMaxVertices()) {
                break;
            }
            counted[a] = id;
            counted[b] = id;
            counted[c] = id;
            vertices += added;
            ++end;
        }
        clusters.push_back(clusterOf(mesh, first, end));
        first = end;
    }
    return clusters;
}

} // namespace lr::render
