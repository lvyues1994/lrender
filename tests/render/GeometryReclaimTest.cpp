#include "Clusters.h"
#include "DeviceScene.h"
#include "RangeAllocator.h"
#include "RenderTestSupport.h"
#include "RenderWorld.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace core = lr::core;
namespace render = lr::render;
using lr::test::RenderHarness;

TEST_CASE("ranges go to the smallest free range they fit, released neighbours merge, and the end gives back") {
    auto ranges = render::RangeAllocator{};
    CHECK(ranges.allocate(10) == 0);
    CHECK(ranges.allocate(5) == 10);
    CHECK(ranges.allocate(20) == 15);
    CHECK(ranges.allocate(3) == 35);
    CHECK(ranges.allocate(0) == 0);
    ranges.release(0, 10);
    ranges.release(15, 20);
    CHECK(ranges.used() == 8);
    CHECK(ranges.wasted() == 30);
    CHECK(ranges.allocate(8) == 0);  // the smaller of the two that fit
    CHECK(ranges.allocate(15) == 15); // only the larger fits
    CHECK(ranges.allocate(2) == 8);   // what is left of the first
    ranges.release(10, 5);
    ranges.release(8, 2);
    CHECK(ranges.allocate(7) == 8); // the two released neighbours, merged
    ranges.release(35, 3);
    CHECK(ranges.end() == 30); // with the free range before it, the end comes back
    ranges.release(15, 15);
    CHECK(ranges.end() == 15);
    CHECK(ranges.used() == 15);
    CHECK(ranges.wasted() == 0);
    CHECK_THROWS_AS(ranges.release(14, 2), std::logic_error);
}

// What the GPU thread's tables hold after each update, each exactly as large as its
// capacity: what lies past it must not matter.
struct TableModel {
    std::vector<render::GpuVertex> vertices;
    std::vector<tc::uint> indices;
    std::vector<render::GpuCluster> clusters;
    std::vector<render::GpuMesh> meshes;
    std::vector<render::GpuBvhNode> blasNodes;
    std::vector<render::GpuBvhTriangle> blasTriangles;
    std::vector<tc::uint> blasRoots;

    template <class T, class Move, class Rebase>
    static void move(std::vector<T> &table, std::vector<Move> const &moves, Rebase const &rebase) {
        for (auto const &move : moves) {
            auto const &range = render::rangeOf(move);
            for (auto index = std::uint32_t{0}; index < range.count; ++index) {
                table.at(range.to + index) = rebase(move, table.at(range.from + index));
            }
        }
    }

    template <class T>
    static void write(std::vector<T> &table, std::vector<render::Indexed<T>> const &writes) {
        for (auto const &entry : writes) {
            table.at(entry.index) = entry.value;
        }
    }

    void apply(render::SceneUpdate const &update) {
        auto const same = [](auto const &, auto const &value) { return value; };
        auto const &moves = update.moves;
        move(vertices, moves.vertices, same);
        move(indices, moves.indices, same);
        move(clusters, moves.clusters, same);
        move(blasNodes, moves.blasNodes, [](render::NodeMove const &m, render::GpuBvhNode node) {
            node.first = node.count == 0 ? node.first - m.nodes.from + m.nodes.to
                                         : node.first - m.fromTriangle + m.toTriangle;
            return node;
        });
        move(blasTriangles, moves.blasTriangles, [](render::TriangleMove const &m, render::GpuBvhTriangle triangle) {
            triangle.corners = triangle.corners - tc::uvec3{m.fromVertex} + tc::uvec3{m.toVertex};
            return triangle;
        });
        auto const &capacity = update.capacity;
        vertices.resize(capacity.vertices);
        indices.resize(capacity.indices);
        clusters.resize(capacity.clusters);
        meshes.resize(capacity.meshes);
        blasNodes.resize(capacity.blasNodes);
        blasTriangles.resize(capacity.blasTriangles);
        blasRoots.resize(capacity.meshes);
        for (auto const &geometry : update.geometry) {
            std::copy(geometry.vertices.begin(), geometry.vertices.end(), vertices.begin() + geometry.firstVertex);
            std::copy(geometry.indices.begin(), geometry.indices.end(), indices.begin() + geometry.firstIndex);
            std::copy(geometry.clusters.begin(), geometry.clusters.end(), clusters.begin() + geometry.firstCluster);
        }
        write(meshes, update.meshes);
        write(blasNodes, update.blasNodes);
        write(blasTriangles, update.blasTriangles);
        write(blasRoots, update.blasRoots);
    }
};

// TableModel's tables, or a DeviceScene's, as holds() reads them.
struct ModelTables {
    TableModel const &tables;

    std::size_t vertexCount() const { return tables.vertices.size(); }
    std::size_t indexCount() const { return tables.indices.size(); }
    std::size_t clusterCount() const { return tables.clusters.size(); }
    std::size_t nodeCount() const { return tables.blasNodes.size(); }
    std::size_t triangleCount() const { return tables.blasTriangles.size(); }
    render::GpuMesh mesh(std::uint32_t const slot) const { return tables.meshes.at(slot); }
    render::GpuVertex vertex(std::size_t const index) const { return tables.vertices[index]; }
    tc::uint index(std::size_t const at) const { return tables.indices[at]; }
    render::GpuCluster cluster(std::size_t const index) const { return tables.clusters[index]; }
    render::GpuBvhNode node(std::size_t const index) const { return tables.blasNodes[index]; }
    render::GpuBvhTriangle triangle(std::size_t const index) const { return tables.blasTriangles[index]; }
    tc::uint root(std::uint32_t const slot) const { return tables.blasRoots.at(slot); }
};

struct DeviceTables {
    render::SceneTables tables;

    std::size_t vertexCount() const { return tables.vertices->size(); }
    std::size_t indexCount() const { return tables.indices->size(); }
    std::size_t clusterCount() const { return tables.clusters->size(); }
    std::size_t nodeCount() const { return tables.blasNodes->size(); }
    std::size_t triangleCount() const { return tables.blasTriangles->size(); }
    render::GpuMesh mesh(std::uint32_t const slot) const { return tables.meshes->get(slot); }
    render::GpuVertex vertex(std::size_t const index) const { return tables.vertices->get(index); }
    tc::uint index(std::size_t const at) const { return (*tables.indices)[static_cast<std::int32_t>(at)]; }
    render::GpuCluster cluster(std::size_t const index) const { return tables.clusters->get(index); }
    render::GpuBvhNode node(std::size_t const index) const { return tables.blasNodes->get(index); }
    render::GpuBvhTriangle triangle(std::size_t const index) const { return tables.blasTriangles->get(index); }
    tc::uint root(std::uint32_t const slot) const { return (*tables.blasRoots)[static_cast<std::int32_t>(slot)]; }
};

// Whether the tables hold `data` where the mesh in `slot` says, hierarchy included.
template <class Tables>
bool holds(Tables const &tables, std::uint32_t const slot, core::MeshData const &data) {
    auto const mesh = tables.mesh(slot);
    if (mesh.vertexCount != data.positions.size() or 3 * mesh.triangleCount != data.indices.size()
        or mesh.firstVertex + mesh.vertexCount > tables.vertexCount()
        or mesh.firstIndex + 3 * mesh.triangleCount > tables.indexCount()
        or mesh.firstCluster + mesh.clusterCount > tables.clusterCount()) {
        return false;
    }
    for (auto index = std::uint32_t{0}; index < mesh.vertexCount; ++index) {
        if (not(tables.vertex(mesh.firstVertex + index).position == data.positions[index])) {
            return false;
        }
    }
    for (auto index = std::uint32_t{0}; index < 3 * mesh.triangleCount; ++index) {
        if (tables.index(mesh.firstIndex + index) != data.indices[index]) {
            return false;
        }
    }
    auto const clusters = render::buildClusters(data);
    if (clusters.size() != mesh.clusterCount) {
        return false;
    }
    for (auto index = std::size_t{0}; index < clusters.size(); ++index) {
        auto const held = tables.cluster(mesh.firstCluster + index);
        if (held.firstTriangle != clusters[index].firstTriangle or held.triangleCount != clusters[index].triangleCount
            or not(held.boundsMin == clusters[index].boundsMin)) {
            return false;
        }
    }
    auto const root = tables.root(slot);
    if (mesh.triangleCount == 0) {
        return root == render::noBvhNode();
    }
    // Every triangle once, with its corners where the mesh's vertices are.
    auto seen = std::set<tc::uint>{};
    auto stack = std::vector<tc::uint>{root};
    while (not stack.empty()) {
        auto const at = stack.back();
        stack.pop_back();
        if (at >= tables.nodeCount()) {
            return false;
        }
        auto const node = tables.node(at);
        if (node.count == 0) {
            stack.push_back(node.first);
            stack.push_back(node.first + 1);
            continue;
        }
        for (auto k = tc::uint{0}; k < node.count; ++k) {
            if (node.first + k >= tables.triangleCount()) {
                return false;
            }
            auto const triangle = tables.triangle(node.first + k);
            auto const first = std::size_t{triangle.primitive} * 3;
            if (first + 2 >= data.indices.size() or not seen.insert(triangle.primitive).second
                or triangle.corners != tc::uvec3{mesh.firstVertex + data.indices[first],
                                                 mesh.firstVertex + data.indices[first + 1],
                                                 mesh.firstVertex + data.indices[first + 2]}) {
                return false;
            }
        }
    }
    return seen.size() == mesh.triangleCount;
}

// A mesh only its id makes: a sphere, a box or nothing, somewhere of its own.
// A sphere of random detail, its random numbers drawn in this order on every compiler.
core::MeshData sphere(std::mt19937 &random, std::uint32_t const segments, std::uint32_t const rings) {
    auto const across = segments + static_cast<std::uint32_t>(random() % 40);
    auto const down = rings + static_cast<std::uint32_t>(random() % 20);
    return core::makeSphere(0.5f, across, down);
}

core::MeshHandle meshFor(std::uint32_t const id, std::mt19937 &random) {
    auto const kind = random() % 8;
    auto mesh = kind == 0 ? core::MeshData{} : kind < 3 ? core::makeBox(core::vec3{0.5f}) : sphere(random, 6, 4);
    for (auto &position : mesh.positions) {
        position.x += 3.0f * static_cast<float>(id);
    }
    return std::make_shared<core::MeshData const>(std::move(mesh));
}

std::uint32_t roundUp(std::uint32_t const value) {
    auto result = std::uint32_t{1};
    while (result < value) {
        result *= 2;
    }
    return result;
}

TEST_CASE("meshes loaded and unloaded a thousand times keep the tables as large as what they hold needs") {
    auto random = std::mt19937{7};
    auto world = render::RenderWorld{};
    auto tables = TableModel{};
    auto request = tc::DeviceRequest{};
    request.backend = "cpu";
    auto const device = tc::createDevice(request);
    auto scene = render::DeviceScene{};
    auto textures = render::TexturePool{render::TexturePoolConfig{}};
    auto retired = render::RetireList{};
    auto start = render::SceneUpdate{};
    world.keepHierarchies(start);
    tables.apply(start);
    scene.apply(*device, start, textures, retired);
    auto live = std::map<std::uint32_t, core::MeshHandle>{};
    auto compactions = 0;
    auto mostVertices = std::uint32_t{0};
    auto mostIndices = std::uint32_t{0};
    auto mostTriangles = std::uint32_t{0};
    auto largestCapacity = render::SceneCapacity{};
    auto wrong = 0;
    auto wrongOnDevice = 0;
    for (auto round = 0; round < 1000; ++round) {
        auto delta = core::RenderWorldDelta{};
        auto upserted = std::map<std::uint32_t, core::MeshHandle>{};
        for (auto operation = 0u; operation < 1 + random() % 3; ++operation) {
            auto const id = static_cast<std::uint32_t>(random() % 24);
            if (live.count(id) != 0 and upserted.count(id) == 0 and random() % 3 == 0) {
                delta.removed.meshes.push_back(core::MeshId{id});
                live.erase(id);
            } else if (std::find(delta.removed.meshes.begin(), delta.removed.meshes.end(), core::MeshId{id})
                       == delta.removed.meshes.end()) {
                auto mesh = meshFor(id, random);
                delta.meshes.push_back({core::MeshId{id}, mesh}); // a second upsert of an id replaces the first
                upserted[id] = mesh;
                live[id] = mesh;
            }
        }
        auto const update = world.apply(delta);
        tables.apply(update);
        scene.apply(*device, update, textures, retired);
        retired.clear();
        compactions += update.moves.vertices.empty() and update.moves.blasNodes.empty() ? 0 : 1;
        auto vertices = std::uint32_t{0};
        auto indices = std::uint32_t{0};
        for (auto const &[id, mesh] : live) {
            vertices += static_cast<std::uint32_t>(mesh->positions.size());
            indices += static_cast<std::uint32_t>(mesh->indices.size());
            auto const slot = world.meshSlot(core::MeshId{id});
            wrong += slot and holds(ModelTables{tables}, *slot, *mesh) ? 0 : 1;
            wrongOnDevice += slot and holds(DeviceTables{scene.tables()}, *slot, *mesh) ? 0 : 1;
        }
        mostVertices = std::max(mostVertices, vertices);
        mostIndices = std::max(mostIndices, indices);
        mostTriangles = std::max(mostTriangles, indices / 3);
        largestCapacity.vertices = std::max(largestCapacity.vertices, update.capacity.vertices);
        largestCapacity.indices = std::max(largestCapacity.indices, update.capacity.indices);
        largestCapacity.blasTriangles = std::max(largestCapacity.blasTriangles, update.capacity.blasTriangles);
    }
    MESSAGE(compactions << " compactions; at most " << mostVertices << " vertices held, in tables of up to "
                        << largestCapacity.vertices);
    CHECK(wrong == 0);
    CHECK(wrongOnDevice == 0);
    CHECK(compactions > 0);
    // Free ranges are packed away once they outgrow what is held, before the next meshes go in.
    CHECK(largestCapacity.vertices <= roundUp(3 * mostVertices));
    CHECK(largestCapacity.indices <= roundUp(3 * mostIndices));
    CHECK(largestCapacity.blasTriangles <= roundUp(3 * mostTriangles));
}

// The meshes the instances use, last as `final`, while others come and go around them.
constexpr auto kKeptMeshes = 3u;

core::RenderWorldDelta keptScene(std::vector<core::MeshHandle> const &kept) {
    auto delta = core::RenderWorldDelta{};
    for (auto index = 0u; index < kKeptMeshes; ++index) {
        delta.meshes.push_back({core::MeshId{index}, kept[index]});
    }
    auto lit = core::MaterialDesc{};
    lit.baseColor = core::vec4{0.8f, 0.6f, 0.4f, 1.0f};
    delta.materials = {{core::MaterialId{0}, lit}};
    for (auto index = 0u; index < 2 * kKeptMeshes; ++index) {
        delta.instances.push_back(
            {core::InstanceId{index}, core::MeshId{index % kKeptMeshes}, core::MaterialId{0},
             lr::test::translation(core::vec3{-1.5f + 1.0f * static_cast<float>(index % 3), 0.0f,
                                              -1.0f * static_cast<float>(index / 3)})});
    }
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.3f, -1.0f, -0.5f};
    delta.lights = {{core::LightId{0}, sun}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.2f, 0.25f, 0.3f}};
    return delta;
}

std::vector<core::MeshHandle> finalMeshes() {
    auto const share = [](core::MeshData mesh) { return std::make_shared<core::MeshData const>(std::move(mesh)); };
    return {share(core::makeSphere(0.45f, 20, 10)), share(core::makeBox(core::vec3{0.35f})),
            share(core::makeSphere(0.3f, 12, 6))};
}

struct Rendered {
    std::vector<std::uint32_t> raster;
    std::vector<std::uint32_t> traced;
    render::SceneMemory memory;
    render::SceneMemory most; // at any frame before
};

constexpr auto kExtent = core::ivec2{40, 32};

// The final scene's frames from the visibility pipeline and the path tracer.
Rendered renderFinal(RenderHarness &harness, core::RenderWorldDelta delta) {
    auto &targets = harness.renderer->targets();
    auto const raster = targets.createImage(render::ImageTargetDesc{kExtent});
    auto const traced = targets.createImage(render::ImageTargetDesc{kExtent});
    auto const camera = lr::test::cameraAt(core::vec3{0.3f, 1.6f, 3.0f}, core::vec3{0.0f, 0.0f, -0.5f});
    auto rasterView = lr::test::viewOf(raster, camera, core::ViewId{10});
    auto tracedView = lr::test::viewOf(traced, camera, core::ViewId{11});
    tracedView.pipeline = harness.renderer->pipelines().pathTraceReference();
    auto result = harness.render(render::FrameDesc{std::move(delta), {rasterView, tracedView}});
    return Rendered{result.views[0].image->pixels, result.views[1].image->pixels, result.sceneMemory, {}};
}

// Big meshes come and go, and the kept ones change, a frame at a time with a path traced
// view so that the hierarchies are kept all along; then the kept ones get their final
// meshes and the others go.
Rendered renderAfterChurn(RenderHarness &harness) {
    auto random = std::mt19937{11};
    auto const small = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{4, 4}});
    auto view = lr::test::viewOf(small, lr::test::cameraAt(core::vec3{0.0f, 1.0f, 3.0f}, core::vec3{0.0f}));
    view.pipeline = harness.renderer->pipelines().pathTraceReference();
    auto kept = std::vector<core::MeshHandle>{};
    for (auto index = 0u; index < kKeptMeshes; ++index) {
        kept.push_back(meshFor(index, random));
    }
    harness.render(render::FrameDesc{keptScene(kept), {view}});
    auto others = std::set<std::uint32_t>{};
    auto most = render::SceneMemory{};
    for (auto frame = 0; frame < 24; ++frame) {
        auto delta = core::RenderWorldDelta{};
        for (auto operation = 0; operation < 3; ++operation) {
            auto const id = 100u + static_cast<std::uint32_t>(random() % 12);
            if (others.count(id) != 0 and random() % 2 == 0) {
                delta.removed.meshes.push_back(core::MeshId{id});
                others.erase(id);
            } else if (std::find(delta.removed.meshes.begin(), delta.removed.meshes.end(), core::MeshId{id})
                       == delta.removed.meshes.end()) {
                delta.meshes.push_back(
                    {core::MeshId{id}, std::make_shared<core::MeshData const>(sphere(random, 24, 12))});
                others.insert(id);
            }
        }
        if (frame % 5 == 0) {
            auto const index = static_cast<std::uint32_t>(random() % kKeptMeshes);
            delta.meshes.push_back({core::MeshId{index}, meshFor(index, random)});
        }
        auto const memory = harness.render(render::FrameDesc{std::move(delta), {view}}).sceneMemory;
        most.geometry = std::max(most.geometry, memory.geometry);
        most.clusters = std::max(most.clusters, memory.clusters);
    }
    auto last = core::RenderWorldDelta{};
    auto const final = finalMeshes();
    for (auto index = 0u; index < kKeptMeshes; ++index) {
        last.meshes.push_back({core::MeshId{index}, final[index]});
    }
    for (auto const id : others) {
        last.removed.meshes.push_back(core::MeshId{id});
    }
    harness.render(render::FrameDesc{std::move(last), {view}});
    auto rendered = renderFinal(harness, core::RenderWorldDelta{});
    rendered.most = most;
    return rendered;
}

TEST_CASE("after meshes come and go, frames are those of a renderer that only ever had what stayed") {
    for (auto const rayQueries : {render::VariantUse::Off, render::VariantUse::Auto}) {
        auto options = lr::test::HarnessOptions{};
        options.rayQueries = rayQueries;
        lr::test::forEachBackend(options, [&](RenderHarness &churned) {
            if (rayQueries == render::VariantUse::Auto and not churned.renderer->capabilities().rayQueries) {
                return;
            }
            CAPTURE(rayQueries == render::VariantUse::Auto);
            auto const after = renderAfterChurn(churned);
            auto fresh = RenderHarness{[&] {
                auto same = options;
                same.backend = churned.renderer->deviceInfo().backend;
                same.device = lr::test::deviceFor(same.backend);
                return same;
            }()};
            auto const expected = renderFinal(fresh, keptScene(finalMeshes()));
            CHECK(after.raster == expected.raster);
            CHECK(after.traced == expected.traced);
            MESSAGE("geometry " << after.memory.geometry << " bytes after churn, at most " << after.most.geometry
                                << ", " << expected.memory.geometry << " fresh");
            // The tables shrank from what the churn took. They hold at most as much free as
            // taken, and are at most four times as large as that before they shrink.
            CHECK(8 * after.memory.geometry <= after.most.geometry);
            CHECK(after.memory.geometry <= 8 * expected.memory.geometry);
            CHECK(after.memory.clusters <= 8 * expected.memory.clusters);
            // Mesh slots stay as many as there were at once, 16 here against 4, and so do the
            // roots of the hierarchies (kept without ray queries).
            CHECK(after.memory.hierarchies <= expected.memory.hierarchies + (16 - 4) * sizeof(tc::uint));
        });
    }
}

} // namespace
