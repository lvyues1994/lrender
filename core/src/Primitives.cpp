#include <lr/core/Primitives.h>

#include <array>
#include <cmath>

namespace lr::core {

namespace {

// Appends a quad whose corners wind counter-clockwise seen from where `normal` points.
void appendQuad(MeshData &mesh, vec3 const &origin, vec3 const &right, vec3 const &up, vec3 const &normal) {
    auto const first = static_cast<std::uint32_t>(mesh.positions.size());
    auto const corners = std::array<vec2, 4>{vec2{0.0f, 0.0f}, vec2{1.0f, 0.0f}, vec2{1.0f, 1.0f}, vec2{0.0f, 1.0f}};
    for (auto const &corner : corners) {
        mesh.positions.push_back(origin + right * corner.x + up * corner.y);
        mesh.normals.push_back(normal);
        mesh.uvs.push_back(vec2{corner.x, 1.0f - corner.y});
    }
    for (auto const index : {0u, 1u, 2u, 0u, 2u, 3u}) {
        mesh.indices.push_back(first + index);
    }
}

} // namespace

MeshData makeBox(vec3 const &halfExtents) {
    auto mesh = MeshData{};
    auto const h = halfExtents;
    auto const x = vec3{2.0f * h.x, 0.0f, 0.0f};
    auto const y = vec3{0.0f, 2.0f * h.y, 0.0f};
    auto const z = vec3{0.0f, 0.0f, 2.0f * h.z};
    appendQuad(mesh, vec3{-h.x, -h.y, h.z}, x, y, vec3{0.0f, 0.0f, 1.0f});
    appendQuad(mesh, vec3{h.x, -h.y, -h.z}, -x, y, vec3{0.0f, 0.0f, -1.0f});
    appendQuad(mesh, vec3{h.x, -h.y, h.z}, -z, y, vec3{1.0f, 0.0f, 0.0f});
    appendQuad(mesh, vec3{-h.x, -h.y, -h.z}, z, y, vec3{-1.0f, 0.0f, 0.0f});
    appendQuad(mesh, vec3{-h.x, h.y, h.z}, x, -z, vec3{0.0f, 1.0f, 0.0f});
    appendQuad(mesh, vec3{-h.x, -h.y, -h.z}, x, z, vec3{0.0f, -1.0f, 0.0f});
    return mesh;
}

MeshData makeSphere(float const radius, std::uint32_t const segments, std::uint32_t const rings) {
    auto mesh = MeshData{};
    for (auto ring = 0u; ring <= rings; ++ring) {
        auto const v = static_cast<float>(ring) / static_cast<float>(rings);
        auto const polar = v * kPi;
        for (auto segment = 0u; segment <= segments; ++segment) {
            auto const u = static_cast<float>(segment) / static_cast<float>(segments);
            auto const azimuth = u * 2.0f * kPi;
            auto const normal =
                vec3{std::sin(polar) * std::sin(azimuth), std::cos(polar), std::sin(polar) * std::cos(azimuth)};
            mesh.positions.push_back(normal * radius);
            mesh.normals.push_back(normal);
            mesh.uvs.push_back(vec2{u, v});
        }
    }
    auto const stride = segments + 1u;
    for (auto ring = 0u; ring < rings; ++ring) {
        for (auto segment = 0u; segment < segments; ++segment) {
            auto const a = ring * stride + segment;
            auto const b = a + stride;
            for (auto const index : {a, b, a + 1u, a + 1u, b, b + 1u}) {
                mesh.indices.push_back(index);
            }
        }
    }
    return mesh;
}

MeshData makePlane(vec2 const &halfExtents) {
    auto mesh = MeshData{};
    appendQuad(mesh, vec3{-halfExtents.x, 0.0f, halfExtents.y}, vec3{2.0f * halfExtents.x, 0.0f, 0.0f},
               vec3{0.0f, 0.0f, -2.0f * halfExtents.y}, vec3{0.0f, 1.0f, 0.0f});
    return mesh;
}

MeshData makeQuad(vec2 const &halfExtents) {
    auto mesh = MeshData{};
    appendQuad(mesh, vec3{-halfExtents.x, -halfExtents.y, 0.0f}, vec3{2.0f * halfExtents.x, 0.0f, 0.0f},
               vec3{0.0f, 2.0f * halfExtents.y, 0.0f}, vec3{0.0f, 0.0f, 1.0f});
    return mesh;
}

} // namespace lr::core
