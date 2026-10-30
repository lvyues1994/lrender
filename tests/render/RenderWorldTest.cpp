#include "RenderWorld.h"

#include <lr/core/Primitives.h>

#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>

namespace {

using namespace lr;

core::MeshHandle quad() { return std::make_shared<core::MeshData const>(core::makeQuad(core::vec2{1.0f})); }

core::RenderWorldDelta oneQuad() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{10}, quad()}};
    delta.materials = {{core::MaterialId{20}, core::MaterialDesc{}}};
    delta.instances = {{core::InstanceId{30}, core::MeshId{10}, core::MaterialId{20}, core::mat4{1.0f}}};
    return delta;
}

TEST_CASE("a new mesh, material and instance become geometry and table writes") {
    auto world = render::RenderWorld{};

    auto const update = world.apply(oneQuad());

    REQUIRE(update.geometry.size() == 1);
    CHECK(update.geometry[0].vertices.size() == 4);
    CHECK(update.geometry[0].indices.size() == 6);
    REQUIRE(update.meshes.size() == 1);
    CHECK(update.meshes[0].value.triangleCount == 2);
    REQUIRE(update.instances.size() == 1);
    CHECK(update.instances[0].value.mesh == update.meshes[0].index);
    CHECK(update.instanceCount == 1);
    CHECK(update.capacity.vertices >= 4);
}

TEST_CASE("removing an instance moves the last one into its place") {
    auto world = render::RenderWorld{};
    auto delta = oneQuad();
    delta.instances.push_back({core::InstanceId{31}, core::MeshId{10}, core::MaterialId{20}, core::mat4{2.0f}});
    delta.instances.push_back({core::InstanceId{32}, core::MeshId{10}, core::MaterialId{20}, core::mat4{3.0f}});
    world.apply(delta);
    auto removal = core::RenderWorldDelta{};
    removal.removed.instances = {core::InstanceId{30}};

    auto const update = world.apply(removal);

    CHECK(update.instanceCount == 2);
    REQUIRE(update.instances.size() == 1);
    CHECK(update.instances[0].index == 0);
    CHECK(update.instances[0].value.objectToWorld.value[0][0] == doctest::Approx(3.0f));
}

TEST_CASE("a delta that refers to unknown ids is rejected and changes nothing") {
    auto world = render::RenderWorld{};
    world.apply(oneQuad());
    auto bad = core::RenderWorldDelta{};
    bad.instances = {{core::InstanceId{31}, core::MeshId{11}, core::MaterialId{20}, core::mat4{1.0f}}};

    CHECK_THROWS_AS(world.apply(bad), std::invalid_argument);
    CHECK(world.instanceCount() == 1);
}

TEST_CASE("a mesh in use cannot be removed, but can go together with its instances") {
    auto world = render::RenderWorld{};
    world.apply(oneQuad());
    auto removeMesh = core::RenderWorldDelta{};
    removeMesh.removed.meshes = {core::MeshId{10}};

    CHECK_THROWS_AS(world.apply(removeMesh), std::invalid_argument);

    removeMesh.removed.instances = {core::InstanceId{30}};
    CHECK_NOTHROW(world.apply(removeMesh));
    CHECK(world.instanceCount() == 0);
}

TEST_CASE("malformed meshes are rejected") {
    auto world = render::RenderWorld{};
    auto mesh = core::makeQuad(core::vec2{1.0f});
    mesh.indices.push_back(99);
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{1}, std::make_shared<core::MeshData const>(mesh)}};

    CHECK_THROWS_AS(world.apply(delta), std::invalid_argument);
}

TEST_CASE("a replaced mesh keeps its slot and gets new geometry") {
    auto world = render::RenderWorld{};
    auto const first = world.apply(oneQuad());
    auto replace = core::RenderWorldDelta{};
    replace.meshes = {{core::MeshId{10}, std::make_shared<core::MeshData const>(core::makeBox(core::vec3{1.0f}))}};

    auto const second = world.apply(replace);

    REQUIRE(second.meshes.size() == 1);
    CHECK(second.meshes[0].index == first.meshes[0].index);
    CHECK(second.meshes[0].value.triangleCount == 12);
    CHECK(second.geometry[0].firstVertex >= 4);
}

} // namespace
