#include "Clusters.h"
#include "RenderTestSupport.h"
#include "RenderWorld.h"

#include <lr/core/Primitives.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace {

using namespace lr;

// Triangles that share no corners: the vertex limit closes each cluster.
core::MeshData separateTriangles(std::uint32_t const count) {
    auto mesh = core::MeshData{};
    for (auto index = std::uint32_t{0}; index < count; ++index) {
        auto const x = static_cast<float>(index % 16);
        auto const y = static_cast<float>(index / 16);
        for (auto const &corner : {core::vec3{x, y, 0.0f}, core::vec3{x + 1.0f, y, 0.0f}, core::vec3{x, y + 1.0f, 0.0f}}) {
            mesh.indices.push_back(static_cast<std::uint32_t>(mesh.positions.size()));
            mesh.positions.push_back(corner);
            mesh.normals.push_back(core::vec3{0.0f, 0.0f, 1.0f});
        }
    }
    return mesh;
}

std::vector<core::MeshData> sampleMeshes() {
    return {core::makeSphere(1.0f, 48, 24), core::makeBox(core::vec3{0.5f, 1.0f, 2.0f}),
            core::makePlane(core::vec2{3.0f}), separateTriangles(100)};
}

core::vec3 cornerOf(core::MeshData const &mesh, std::uint32_t const triangle, std::uint32_t const k) {
    return mesh.positions[mesh.indices[std::size_t{triangle} * 3 + k]];
}

TEST_CASE("clusters take a mesh's triangles in order, within the limits, each in a box around them") {
    for (auto const &mesh : sampleMeshes()) {
        auto const triangles = static_cast<std::uint32_t>(mesh.indices.size() / 3);
        CAPTURE(triangles);
        auto const clusters = render::buildClusters(mesh);

        auto next = std::uint32_t{0};
        for (auto const &cluster : clusters) {
            CHECK(cluster.firstTriangle == next);
            CHECK(cluster.triangleCount >= 1);
            CHECK(cluster.triangleCount <= render::clusterMaxTriangles());
            auto vertices = std::set<std::uint32_t>{};
            for (auto triangle = cluster.firstTriangle; triangle < cluster.firstTriangle + cluster.triangleCount;
                 ++triangle) {
                for (auto k = 0u; k < 3u; ++k) {
                    vertices.insert(mesh.indices[std::size_t{triangle} * 3 + k]);
                    auto const p = cornerOf(mesh, triangle, k);
                    CHECK(p == tc::max(p, cluster.boundsMin));
                    CHECK(p == tc::min(p, cluster.boundsMax));
                    CHECK(tc::length(p - cluster.center) <= cluster.radius * 1.0001f + 1e-6f);
                }
            }
            CHECK(vertices.size() <= render::clusterMaxVertices());
            next += cluster.triangleCount;
        }
        CHECK(next == triangles);
    }
    CHECK(render::buildClusters(separateTriangles(100)).size() == 5); // 21 triangles of 3 vertices each
}

TEST_CASE("a cluster the normal cone culls from a viewpoint shows that viewpoint only backs of triangles") {
    auto culled = 0;
    for (auto const &mesh : sampleMeshes()) {
        auto const clusters = render::buildClusters(mesh);
        for (auto step = 0; step < 64; ++step) {
            auto const angle = 0.1f + 0.7f * static_cast<float>(step);
            auto const height = std::sin(1.3f * static_cast<float>(step));
            auto const distance = 1.5f + 0.5f * static_cast<float>(step % 8);
            auto const viewer = core::vec3{std::cos(angle), height, std::sin(angle)} * distance;
            for (auto const &cluster : clusters) {
                auto const toCenter = cluster.center - viewer;
                if (tc::dot(toCenter, cluster.coneAxis) < cluster.coneCutoff * tc::length(toCenter) + cluster.radius) {
                    continue;
                }
                ++culled;
                for (auto triangle = cluster.firstTriangle; triangle < cluster.firstTriangle + cluster.triangleCount;
                     ++triangle) {
                    auto const a = cornerOf(mesh, triangle, 0);
                    auto const normal = tc::cross(cornerOf(mesh, triangle, 1) - a, cornerOf(mesh, triangle, 2) - a);
                    CHECK(tc::dot(normal, a - viewer) >= 0.0f);
                }
            }
        }
    }
    MESSAGE(culled << " clusters culled");
    CHECK(culled > 100);
}

TEST_CASE("the world puts each mesh's clusters in the cluster table, and a full update again") {
    auto world = render::RenderWorld{};
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeSphere(1.0f, 48, 24))},
                    {core::MeshId{1}, test::share(core::makeBox(core::vec3{1.0f}))}};
    auto const update = world.apply(delta);

    REQUIRE(update.geometry.size() == 2);
    REQUIRE(update.meshes.size() == 2);
    auto const &sphere = update.meshes[0].value;
    auto const &box = update.meshes[1].value;
    CHECK(sphere.firstCluster == 0);
    CHECK(sphere.clusterCount == update.geometry[0].clusters.size());
    CHECK(box.firstCluster == sphere.clusterCount);
    CHECK(box.clusterCount == 1);
    CHECK(update.geometry[1].firstCluster == box.firstCluster);
    CHECK(update.capacity.clusters >= sphere.clusterCount + box.clusterCount);

    auto const full = world.fullUpdate();
    auto clusters = std::size_t{0};
    for (auto const &geometry : full.geometry) {
        clusters += geometry.clusters.size();
    }
    CHECK(clusters == sphere.clusterCount + box.clusterCount);
}

} // namespace
