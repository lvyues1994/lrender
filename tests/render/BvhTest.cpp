// White-box tests of the hierarchies: their structure, and a traversal on the host built
// from the path tracer's own box, child-order and triangle tests.

#include "Bvh.h"
#include "RenderWorld.h"
#include "kernels/PathTrace.h"

#include <lr/core/Primitives.h>
#include <lr/runtime/Runtime.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace lr;
using render::GpuBvhNode;

struct Triangle {
    core::vec3 a;
    core::vec3 b;
    core::vec3 c;
};

std::vector<Triangle> randomTriangles(std::mt19937 &random, std::size_t const count) {
    auto place = std::uniform_real_distribution<float>{-1.0f, 1.0f};
    auto size = std::uniform_real_distribution<float>{-0.2f, 0.2f};
    auto triangles = std::vector<Triangle>{};
    for (auto index = std::size_t{0}; index < count; ++index) {
        auto const centre = core::vec3{place(random), place(random), place(random)};
        triangles.push_back(Triangle{centre + core::vec3{size(random), size(random), size(random)},
                                     centre + core::vec3{size(random), size(random), size(random)},
                                     centre + core::vec3{size(random), size(random), size(random)}});
    }
    return triangles;
}

std::vector<core::Aabb> boundsOf(std::vector<Triangle> const &triangles) {
    auto bounds = std::vector<core::Aabb>{};
    for (auto const &t : triangles) {
        bounds.push_back(core::Aabb{tc::min(tc::min(t.a, t.b), t.c), tc::max(tc::max(t.a, t.b), t.c)});
    }
    return bounds;
}

bool contains(core::Aabb const &outer, core::Aabb const &inner) {
    return outer.min.x <= inner.min.x and outer.min.y <= inner.min.y and outer.min.z <= inner.min.z
       and outer.max.x >= inner.max.x and outer.max.y >= inner.max.y and outer.max.z >= inner.max.z;
}

// What walking a hierarchy from its root finds.
struct Walk {
    std::vector<std::uint32_t> leafHits; // per place in Bvh::order
    std::uint32_t deepestLeaf = 0;
    std::uint32_t largestLeaf = 0;
    bool nested = true; // every node's bounds hold its children's, every leaf's its primitives'
    std::size_t nodesVisited = 0;
};

Walk walk(render::Bvh const &bvh, std::vector<core::Aabb> const &bounds) {
    struct Entry {
        std::uint32_t node;
        std::uint32_t depth;
    };
    auto result = Walk{};
    result.leafHits.assign(bvh.order.size(), 0);
    auto pending = std::vector<Entry>{Entry{0, 0}};
    while (not pending.empty()) {
        auto const entry = pending.back();
        pending.pop_back();
        auto const &node = bvh.nodes.at(entry.node);
        result.nodesVisited += 1;
        auto const box = render::boundsOf(node);
        if (node.count > 0) {
            result.deepestLeaf = std::max(result.deepestLeaf, entry.depth);
            result.largestLeaf = std::max(result.largestLeaf, node.count);
            for (auto place = node.first; place < node.first + node.count; ++place) {
                result.leafHits.at(place) += 1;
                result.nested = result.nested and contains(box, bounds[bvh.order[place]]);
            }
            continue;
        }
        for (auto const child : {node.first, node.first + 1}) {
            result.nested = result.nested and contains(box, render::boundsOf(bvh.nodes.at(child)));
            pending.push_back(Entry{child, entry.depth + 1});
        }
    }
    return result;
}

void checkStructure(std::vector<core::Aabb> const &bounds, std::uint32_t const maxLeafSize) {
    auto const bvh = render::buildBvh(bounds, render::BvhBuildConfig{maxLeafSize});
    auto const found = walk(bvh, bounds);

    CHECK(found.nested);
    CHECK(found.deepestLeaf <= render::bvhMaxDepth());
    CHECK(found.largestLeaf <= maxLeafSize);
    CHECK(found.nodesVisited == bvh.nodes.size());
    CHECK(std::all_of(found.leafHits.begin(), found.leafHits.end(), [](std::uint32_t const hits) { return hits == 1; }));
    auto order = bvh.order;
    std::sort(order.begin(), order.end());
    auto all = std::vector<std::uint32_t>(bounds.size());
    for (auto index = std::size_t{0}; index < all.size(); ++index) {
        all[index] = static_cast<std::uint32_t>(index);
    }
    CHECK(order == all);
}

TEST_CASE("every primitive is in exactly one leaf, and every node holds its children") {
    auto random = std::mt19937{7};
    for (auto const count : {std::size_t{1}, std::size_t{2}, std::size_t{5}, std::size_t{1000}}) {
        CAPTURE(count);
        checkStructure(boundsOf(randomTriangles(random, count)), 4);
        checkStructure(boundsOf(randomTriangles(random, count)), 1);
    }
    auto sphere = std::vector<Triangle>{};
    auto const mesh = core::makeSphere(1.0f, 32, 16);
    for (auto first = std::size_t{0}; first < mesh.indices.size(); first += 3) {
        sphere.push_back(Triangle{mesh.positions[mesh.indices[first]], mesh.positions[mesh.indices[first + 1]],
                                  mesh.positions[mesh.indices[first + 2]]});
    }
    checkStructure(boundsOf(sphere), 4);
}

TEST_CASE("hierarchies stay within the depth limit where the heuristic alone would not") {
    // Each box twice as far out as the one before: the heuristic splits one box off at a time.
    auto spread = std::vector<core::Aabb>{};
    for (auto index = 0; index < 100; ++index) {
        auto const x = std::ldexp(1.0f, index);
        spread.push_back(core::Aabb{core::vec3{x, 0.0f, 0.0f}, core::vec3{x * 1.5f, 1.0f, 1.0f}});
    }
    checkStructure(spread, 4);
    checkStructure(spread, 1);
    // Equal centres give the heuristic nothing to split.
    checkStructure(std::vector<core::Aabb>(5000, core::Aabb{core::vec3{0.0f}, core::vec3{1.0f}}), 4);
    CHECK(render::buildBvh({}, render::BvhBuildConfig{4}).nodes.empty());
}

template <class T>
std::vector<T> gather(std::vector<render::Indexed<T>> const &writes) {
    auto size = std::size_t{0};
    for (auto const &write : writes) {
        size = std::max(size, std::size_t{write.index} + 1);
    }
    auto values = std::vector<T>(size);
    for (auto const &write : writes) {
        values[write.index] = write.value;
    }
    return values;
}

// The scene as one update that built it all writes it.
struct HostScene {
    explicit HostScene(render::SceneUpdate const &update)
        : meshes(gather(update.meshes)), instances(gather(update.instances)), blasNodes(gather(update.blasNodes)),
          blasTriangles(gather(update.blasTriangles)), blasRoots(gather(update.blasRoots)),
          tlasNodes(gather(update.tlasNodes)), tlasNodeCount(update.tlasNodeCount) {
        for (auto const &geometry : update.geometry) {
            positions.resize(std::max<std::size_t>(positions.size(), geometry.firstVertex + geometry.vertices.size()));
            indices.resize(std::max<std::size_t>(indices.size(), geometry.firstIndex + geometry.indices.size()));
            for (auto index = std::size_t{0}; index < geometry.vertices.size(); ++index) {
                positions[geometry.firstVertex + index] = geometry.vertices[index].position;
            }
            std::copy(geometry.indices.begin(), geometry.indices.end(), indices.begin() + geometry.firstIndex);
        }
    }

    std::vector<core::vec3> positions;
    std::vector<std::uint32_t> indices;
    std::vector<render::GpuMesh> meshes;
    std::vector<render::GpuInstance> instances;
    std::vector<GpuBvhNode> blasNodes;
    std::vector<render::GpuBvhTriangle> blasTriangles;
    std::vector<tc::uint> blasRoots;
    std::vector<GpuBvhNode> tlasNodes;
    std::uint32_t tlasNodeCount;
};

constexpr auto kFar = 1e30f;

// The path tracer's traversal: nearest first, a stack of later nodes, the ray in each
// instance's object space.
template <class Visit>
void traverse(std::vector<GpuBvhNode> const &nodes, std::uint32_t const root, core::vec3 const &origin,
              core::vec3 const &direction, float const &far, Visit const &visitLeaf) {
    namespace k = render::kernels;
    auto const inverse = k::safeInverse(direction);
    if (k::boxEntry(origin, inverse, nodes.at(root).boundsMin, nodes.at(root).boundsMax, far) < 0.0f) {
        return;
    }
    auto stack = std::array<std::uint32_t, render::bvhMaxDepth() + 1>{};
    auto depth = std::size_t{0};
    auto node = root;
    for (;;) {
        auto const &visited = nodes.at(node);
        auto next = render::noBvhNode();
        if (visited.count > 0) {
            visitLeaf(visited);
        } else {
            auto const &left = nodes.at(visited.first);
            auto const &right = nodes.at(visited.first + 1);
            auto const order = k::orderChildren(visited.first, k::boxEntry(origin, inverse, left.boundsMin, left.boundsMax, far),
                                                k::boxEntry(origin, inverse, right.boundsMin, right.boundsMax, far));
            next = order.next;
            if (order.later != render::noBvhNode()) {
                stack.at(depth++) = order.later;
            }
        }
        if (next == render::noBvhNode()) {
            if (depth == 0) {
                return;
            }
            next = stack[--depth];
        }
        node = next;
    }
}

float traceHost(HostScene const &scene, core::vec3 const &origin, core::vec3 const &direction) {
    auto nearest = kFar;
    if (scene.tlasNodeCount == 0) {
        return nearest;
    }
    traverse(scene.tlasNodes, 0, origin, direction, nearest, [&](GpuBvhNode const &leaf) {
        auto const &instance = scene.instances.at(leaf.first);
        auto const toObject = instance.worldToObject.value;
        auto const objectOrigin = (toObject * core::vec4{origin, 1.0f}).xyz();
        auto const objectDirection = core::mat3{toObject} * direction;
        auto const root = scene.blasRoots.at(instance.mesh);
        REQUIRE(root != render::noBvhNode());
        traverse(scene.blasNodes, root, objectOrigin, objectDirection, nearest, [&](GpuBvhNode const &blasLeaf) {
            for (auto place = blasLeaf.first; place < blasLeaf.first + blasLeaf.count; ++place) {
                auto const corners = scene.blasTriangles.at(place).corners;
                auto const hit = render::kernels::intersectTriangle(objectOrigin, objectDirection,
                                                                    scene.positions.at(corners.x),
                                                                    scene.positions.at(corners.y),
                                                                    scene.positions.at(corners.z));
                if (hit.distance > 0.0f and hit.distance < nearest) {
                    nearest = hit.distance;
                }
            }
        });
    });
    return nearest;
}

// Every triangle of every instance, in world space.
std::vector<Triangle> worldTriangles(HostScene const &scene) {
    auto triangles = std::vector<Triangle>{};
    for (auto const &instance : scene.instances) {
        auto const &mesh = scene.meshes.at(instance.mesh);
        auto const toWorld = instance.objectToWorld.value;
        auto const corner = [&](std::size_t const index) {
            return (toWorld * core::vec4{scene.positions.at(mesh.firstVertex + scene.indices.at(index)), 1.0f}).xyz();
        };
        for (auto triangle = std::size_t{0}; triangle < mesh.triangleCount; ++triangle) {
            auto const first = mesh.firstIndex + 3 * triangle;
            triangles.push_back(Triangle{corner(first), corner(first + 1), corner(first + 2)});
        }
    }
    return triangles;
}

float traceBruteForce(std::vector<Triangle> const &triangles, core::vec3 const &origin, core::vec3 const &direction) {
    auto nearest = kFar;
    for (auto const &triangle : triangles) {
        auto const hit = render::kernels::intersectTriangle(origin, direction, triangle.a, triangle.b, triangle.c);
        if (hit.distance > 0.0f and hit.distance < nearest) {
            nearest = hit.distance;
        }
    }
    return nearest;
}

core::MeshData soup(std::vector<Triangle> const &triangles) {
    auto mesh = core::MeshData{};
    for (auto const &t : triangles) {
        for (auto const &corner : {t.a, t.b, t.c}) {
            mesh.indices.push_back(static_cast<std::uint32_t>(mesh.positions.size()));
            mesh.positions.push_back(corner);
            mesh.normals.push_back(core::vec3{0.0f, 1.0f, 0.0f});
        }
    }
    return mesh;
}

core::RenderWorldDelta scatteredInstances(std::mt19937 &random) {
    auto delta = core::RenderWorldDelta{};
    auto share = [](core::MeshData mesh) { return std::make_shared<core::MeshData const>(std::move(mesh)); };
    delta.meshes = {{core::MeshId{0}, share(core::makeSphere(0.5f, 16, 8))},
                    {core::MeshId{1}, share(core::makeBox(core::vec3{0.3f, 0.6f, 0.2f}))},
                    {core::MeshId{2}, share(soup(randomTriangles(random, 200)))},
                    {core::MeshId{3}, share(core::MeshData{})}};
    delta.materials = {{core::MaterialId{0}, core::MaterialDesc{}}};
    auto place = std::uniform_real_distribution<float>{-3.0f, 3.0f};
    auto angle = std::uniform_real_distribution<float>{0.0f, 6.0f};
    auto scale = std::uniform_real_distribution<float>{0.5f, 2.0f};
    for (auto index = 0u; index < 24u; ++index) {
        auto transform = core::Transform{};
        transform.translation = core::vec3{place(random), place(random), place(random)};
        transform.rotation = core::Rotation::axisAngle(core::vec3{place(random), place(random), 1.0f}, angle(random));
        // Every fifth instance is mirrored.
        transform.scale = core::vec3{scale(random) * (index % 5 == 0 ? -1.0f : 1.0f), scale(random), scale(random)};
        delta.instances.push_back(
            {core::InstanceId{index}, core::MeshId{index % 4}, core::MaterialId{0}, core::matrixOf(transform)});
    }
    return delta;
}

TEST_CASE("the two-level traversal finds the hits that testing every triangle finds") {
    auto random = std::mt19937{11};
    auto world = render::RenderWorld{};
    auto update = world.apply(scatteredInstances(random));
    world.keepHierarchies(update);
    auto const scene = HostScene{update};
    auto const triangles = worldTriangles(scene);
    REQUIRE(scene.tlasNodeCount > 0);

    auto place = std::uniform_real_distribution<float>{-5.0f, 5.0f};
    auto hits = 0;
    auto disagreements = 0;
    for (auto ray = 0; ray < 1000; ++ray) {
        auto const origin = core::vec3{place(random), place(random), place(random)};
        auto const direction = tc::normalize(core::vec3{place(random), place(random), place(random)});
        auto const expected = traceBruteForce(triangles, origin, direction);
        auto const found = traceHost(scene, origin, direction);
        hits += expected < kFar ? 1 : 0;
        auto const agree = expected == kFar ? found == kFar : std::abs(found - expected) <= 1e-4f * (1.0f + expected);
        disagreements += agree ? 0 : 1;
    }
    CHECK(hits > 100);
    CHECK(disagreements == 0);
}

TEST_CASE("hierarchies follow the deltas after they are first built") {
    auto random = std::mt19937{3};
    auto world = render::RenderWorld{};
    auto first = world.apply(scatteredInstances(random));
    CHECK(first.blasNodes.empty());
    CHECK(first.capacity.blasNodes == 0);
    world.keepHierarchies(first);
    REQUIRE(first.blasRoots.size() == 4);
    CHECK(gather(first.blasRoots)[3] == render::noBvhNode()); // the empty mesh
    CHECK(first.tlasNodeCount == 2 * 18 - 1);                  // instances of the other three meshes

    auto lightOnly = core::RenderWorldDelta{};
    lightOnly.lights = {{core::LightId{0}, core::DirectionalLight{}}};
    auto const versionBefore = world.version();
    auto const afterLight = world.apply(lightOnly);
    CHECK(afterLight.tlasNodes.empty());
    CHECK(afterLight.blasNodes.empty());
    CHECK(afterLight.tlasNodeCount == first.tlasNodeCount);
    CHECK(world.version() == versionBefore + 1);
    world.apply(core::RenderWorldDelta{});
    CHECK(world.version() == versionBefore + 1);

    auto moved = core::RenderWorldDelta{};
    moved.instances = {{core::InstanceId{1}, core::MeshId{1}, core::MaterialId{0}, core::mat4{1.0f}}};
    moved.meshes = {{core::MeshId{0}, std::make_shared<core::MeshData const>(core::makeBox(core::vec3{1.0f}))}};
    auto const afterMove = world.apply(moved);
    CHECK(afterMove.tlasNodes.size() == afterMove.tlasNodeCount);
    REQUIRE(afterMove.blasRoots.size() == 1);
    CHECK(afterMove.blasRoots[0].index == first.meshes[0].index);
    // In the place the sphere's larger hierarchy gave back.
    CHECK(afterMove.blasRoots[0].value == gather(first.blasRoots)[first.meshes[0].index]);
    REQUIRE(afterMove.blasTriangles.size() == 12);
    CHECK(afterMove.blasTriangles[0].index == 0);
}

bool sameNodes(std::vector<render::Indexed<render::GpuBvhNode>> const &a,
               std::vector<render::Indexed<render::GpuBvhNode>> const &b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](auto const &x, auto const &y) {
        return x.index == y.index and x.value.boundsMin == y.value.boundsMin and x.value.boundsMax == y.value.boundsMax
           and x.value.first == y.value.first and x.value.count == y.value.count;
    });
}

// The frame pool encodes the meshes and builds their hierarchies at once; what it makes
// lands where the serial order puts it.
TEST_CASE("a world that works on the frame pool makes the updates a serial one makes") {
    auto config = runtime::RuntimeConfig{};
    config.framePoolThreads = 3;
    auto const runtime = runtime::makeRuntime(config);
    auto const parallel = render::makeFramePoolFor(*runtime);
    auto const conversions = render::makeImmediateConversions();
    auto pooled =
        render::RenderWorld{render::TextureLimits{}, *parallel, *conversions, render::ConversionDelivery::Deterministic};
    auto serial = render::RenderWorld{};
    auto random = std::mt19937{5};
    auto const delta = scatteredInstances(random);
    auto again = core::RenderWorldDelta{};
    again.meshes = {delta.meshes[2], delta.meshes[0]};

    for (auto const *step : std::vector<core::RenderWorldDelta const *>{&delta, &again}) {
        auto a = pooled.apply(*step);
        auto b = serial.apply(*step);
        if (step == &delta) {
            pooled.keepHierarchies(a);
            serial.keepHierarchies(b);
        }
        REQUIRE(a.geometry.size() == b.geometry.size());
        for (auto index = std::size_t{0}; index < a.geometry.size(); ++index) {
            auto const &x = a.geometry[index];
            auto const &y = b.geometry[index];
            CHECK(x.firstVertex == y.firstVertex);
            CHECK(x.firstIndex == y.firstIndex);
            CHECK(x.indices == y.indices);
            CHECK(x.firstCluster == y.firstCluster);
            auto const sameClusters = std::equal(
                x.clusters.begin(), x.clusters.end(), y.clusters.begin(), y.clusters.end(), [](auto const &p, auto const &q) {
                    return p.boundsMin == q.boundsMin and p.boundsMax == q.boundsMax and p.center == q.center
                       and p.radius == q.radius and p.coneAxis == q.coneAxis and p.coneCutoff == q.coneCutoff
                       and p.firstTriangle == q.firstTriangle and p.triangleCount == q.triangleCount;
                });
            CHECK(sameClusters);
            auto const sameVertices = std::equal(
                x.vertices.begin(), x.vertices.end(), y.vertices.begin(), y.vertices.end(), [](auto const &p, auto const &q) {
                    return p.position == q.position and p.normal.value == q.normal.value and p.uv.value == q.uv.value;
                });
            CHECK(sameVertices);
        }
        auto const sameMeshes = std::equal(a.meshes.begin(), a.meshes.end(), b.meshes.begin(), b.meshes.end(),
                                           [](auto const &x, auto const &y) {
                                               return x.index == y.index and x.value.firstVertex == y.value.firstVertex
                                                  and x.value.boundsMin == y.value.boundsMin
                                                  and x.value.vertexCount == y.value.vertexCount;
                                           });
        CHECK(sameMeshes);
        CHECK(sameNodes(a.blasNodes, b.blasNodes));
        CHECK(sameNodes(a.tlasNodes, b.tlasNodes));
        CHECK(gather(a.blasRoots) == gather(b.blasRoots));
        CHECK(a.blasTriangles.size() == b.blasTriangles.size());
    }
}

} // namespace
