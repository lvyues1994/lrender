#include "RenderWorld.h"

#include <lr/core/Primitives.h>

#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>
#include <vector>

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

TEST_CASE("an instance keeps where the previous frame had it, for motion") {
    auto world = render::RenderWorld{};
    auto const added = world.apply(oneQuad());
    REQUIRE(added.instances.size() == 1);
    CHECK(added.instances[0].value.previousObjectToWorld.value[3][0] == 0.0f);

    auto moved = core::mat4{1.0f};
    moved[3][0] = 2.0f;
    auto move = core::RenderWorldDelta{};
    move.instances = {{core::InstanceId{30}, core::MeshId{10}, core::MaterialId{20}, moved}};
    auto const afterMove = world.apply(move);
    REQUIRE(afterMove.instances.size() == 1);
    CHECK(afterMove.instances[0].value.objectToWorld.value[3][0] == 2.0f);
    CHECK(afterMove.instances[0].value.previousObjectToWorld.value[3][0] == 0.0f);

    // The next frame finds it where this one left it, and the one after has nothing to write.
    auto const settled = world.apply(core::RenderWorldDelta{});
    REQUIRE(settled.instances.size() == 1);
    CHECK(settled.instances[0].value.previousObjectToWorld.value[3][0] == 2.0f);
    CHECK(world.apply(core::RenderWorldDelta{}).instances.empty());
    CHECK(world.version() == 2);
}

TEST_CASE("materials of a material model the world does not allow are rejected") {
    auto world = render::RenderWorld{};
    auto delta = oneQuad();
    delta.materials[0].material.model = core::MaterialModelId{1};
    delta.materials[0].material.modelParameters = core::vec4{0.5f};

    CHECK_THROWS_AS(world.apply(delta), std::invalid_argument);
    CHECK(world.instanceCount() == 0);

    world.allowMaterialModels(2);
    auto const update = world.apply(delta);
    REQUIRE(update.materials.size() == 1);
    CHECK(update.materials[0].value.gpu.model == 1);
    CHECK(update.materials[0].value.gpu.parameters.y == 0.5f);
}

TEST_CASE("a delta that refers to unknown ids is rejected and changes nothing") {
    auto world = render::RenderWorld{};
    world.apply(oneQuad());
    auto bad = core::RenderWorldDelta{};
    bad.instances = {{core::InstanceId{31}, core::MeshId{11}, core::MaterialId{20}, core::mat4{1.0f}}};

    CHECK_THROWS_AS(world.apply(bad), std::invalid_argument);
    CHECK(world.instanceCount() == 1);
}

TEST_CASE("a delta removes what the world has or what it upserts itself, and nothing else") {
    auto world = render::RenderWorld{};
    auto start = oneQuad();
    start.lights = {{core::LightId{40}, core::PointLight{}}};
    world.apply(start);
    auto unknown = std::vector<core::RenderWorldDelta>(3);
    unknown[0].removed.instances = {core::InstanceId{31}};
    unknown[1].removed.lights = {core::LightId{41}};
    unknown[2].removed.materials = {core::MaterialId{21}};
    for (auto const &delta : unknown) {
        CHECK_THROWS_AS(world.apply(delta), std::invalid_argument);
    }

    auto passing = core::RenderWorldDelta{};
    passing.instances = {{core::InstanceId{32}, core::MeshId{10}, core::MaterialId{20}, core::mat4{1.0f}}};
    passing.lights = {{core::LightId{42}, core::PointLight{}}};
    passing.removed.instances = {core::InstanceId{30}, core::InstanceId{32}};
    passing.removed.lights = {core::LightId{40}, core::LightId{42}};
    CHECK_NOTHROW(world.apply(passing));
    CHECK(world.instanceCount() == 0);
    CHECK(world.lightCount() == 0);
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

TEST_CASE("a full update writes everything the world holds now, hierarchies included") {
    auto world = render::RenderWorld{};
    auto delta = oneQuad();
    delta.instances.push_back({core::InstanceId{31}, core::MeshId{10}, core::MaterialId{20}, core::mat4{2.0f}});
    delta.lights = {{core::LightId{40}, core::DirectionalLight{}}};
    auto unused = render::SceneUpdate{};
    world.keepHierarchies(unused);
    world.apply(delta);
    auto change = core::RenderWorldDelta{};
    change.meshes = {{core::MeshId{10}, std::make_shared<core::MeshData const>(core::makeBox(core::vec3{1.0f}))}};
    change.removed.instances = {core::InstanceId{30}};
    auto const changed = world.apply(change);

    auto const full = world.fullUpdate();

    REQUIRE(full.geometry.size() == 1);
    CHECK(full.geometry[0].firstVertex == changed.geometry[0].firstVertex);
    CHECK(full.geometry[0].vertices.size() == changed.geometry[0].vertices.size());
    REQUIRE(full.meshes.size() == 1);
    CHECK(full.meshes[0].value.triangleCount == 12);
    CHECK(full.materials.size() == 1);
    REQUIRE(full.instances.size() == 1);
    CHECK(full.instances[0].value.objectToWorld.value[0][0] == doctest::Approx(2.0f));
    CHECK(full.instanceCount == 1);
    CHECK(full.lights.size() == 1);
    CHECK(full.lightCount == 1);
    CHECK(full.blasRoots.size() == 1);
    CHECK(full.blasTriangles.size() == 12);
    CHECK(full.tlasNodeCount == 1);
    CHECK(full.capacity.blasNodes >= full.blasNodes.size());
}

core::ImageHandle image(core::ivec2 const extent, std::size_t const bytes) {
    auto data = core::ImageData{};
    data.extent = extent;
    data.levels.emplace_back(bytes, std::uint8_t{128});
    return std::make_shared<core::ImageData const>(std::move(data));
}

TEST_CASE("malformed images and removals of unknown textures are rejected") {
    auto world = render::RenderWorld{};
    auto missing = core::RenderWorldDelta{};
    missing.textures = {{core::TextureId{1}, nullptr, {}}};
    auto shortLevel = core::RenderWorldDelta{};
    shortLevel.textures = {{core::TextureId{1}, image(core::ivec2{4, 4}, 60), {}}};
    auto unknown = core::RenderWorldDelta{};
    unknown.removed.textures = {core::TextureId{9}};

    CHECK_THROWS_AS(world.apply(missing), std::invalid_argument);
    CHECK_THROWS_AS(world.apply(shortLevel), std::invalid_argument);
    CHECK_THROWS_AS(world.apply(unknown), std::invalid_argument);

    auto fine = core::RenderWorldDelta{};
    fine.textures = {{core::TextureId{1}, image(core::ivec2{4, 4}, 64), {}}};
    fine.removed.textures = {core::TextureId{1}};
    CHECK_NOTHROW(world.apply(fine));
}

// glTF's KHR_texture_transform: scale, then rotate, then offset.
TEST_CASE("a material carries the ids of its textures, its alpha mode and its texture transform") {
    auto world = render::RenderWorld{};
    auto material = core::MaterialDesc{};
    material.baseColorTexture = core::TextureId{5};
    material.normalTexture = core::TextureId{7};
    material.alphaMode = core::AlphaMode::Mask;
    material.uvTransform.scale = core::vec2{2.0f, 3.0f};
    material.uvTransform.rotation = core::kPi * 0.5f;
    material.uvTransform.offset = core::vec2{0.5f, 0.25f};
    auto delta = core::RenderWorldDelta{};
    delta.materials = {{core::MaterialId{0}, material}};

    auto const update = world.apply(delta);

    REQUIRE(update.materials.size() == 1);
    auto const &encoded = update.materials[0].value;
    CHECK(encoded.textures[0] == core::TextureId{5});
    CHECK(not encoded.textures[1].isValid());
    CHECK(not encoded.textures[2].isValid());
    CHECK(encoded.textures[3] == core::TextureId{7});
    CHECK(encoded.gpu.textures.x == render::noTexture());
    CHECK(encoded.gpu.flags == render::materialAlphaMask());
    // (u, v) = (1, 0) scales to (2, 0), turns to (0, 2) and moves to (0.5, 2.25).
    auto const uv = core::vec3{1.0f, 0.0f, 1.0f};
    CHECK(tc::dot(encoded.gpu.uvRow0, uv) == doctest::Approx(0.5f));
    CHECK(tc::dot(encoded.gpu.uvRow1, uv) == doctest::Approx(2.25f));
}

TEST_CASE("the world counts blended instances, bounds every instance and finds the light that casts shadows") {
    auto world = render::RenderWorld{};
    auto delta = oneQuad();
    auto glass = core::MaterialDesc{};
    glass.alphaMode = core::AlphaMode::Blend;
    delta.materials.push_back({core::MaterialId{21}, glass});
    auto moved = core::Transform{};
    moved.translation = core::vec3{5.0f, 0.0f, 0.0f};
    delta.instances.push_back({core::InstanceId{31}, core::MeshId{10}, core::MaterialId{21}, core::matrixOf(moved)});
    auto plain = core::DirectionalLight{};
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{0.0f, -2.0f, 0.0f};
    sun.castsShadows = true;
    delta.lights = {{core::LightId{0}, core::PointLight{}}, {core::LightId{1}, plain}, {core::LightId{2}, sun}};

    world.apply(delta);

    CHECK(world.transparentInstanceCount() == 1);
    CHECK(world.bounds().min.x == doctest::Approx(-1.0f));
    CHECK(world.bounds().max.x == doctest::Approx(6.0f));
    REQUIRE(world.shadowLight().has_value());
    CHECK(world.shadowLight()->index == 2);
    CHECK(world.shadowLight()->direction.y == doctest::Approx(-1.0f));

    auto removal = core::RenderWorldDelta{};
    removal.removed.instances = {core::InstanceId{31}};
    world.apply(removal);
    CHECK(world.transparentInstanceCount() == 0);
    CHECK(world.bounds().max.x == doctest::Approx(1.0f));
}

TEST_CASE("the world hands images over to the GPU thread without keeping them, and reports them once lost") {
    auto world = render::RenderWorld{};
    auto const handle = image(core::ivec2{16, 16}, 16 * 16 * 4);
    auto delta = core::RenderWorldDelta{};
    delta.textures = {{core::TextureId{3}, handle, {}}};
    {
        auto update = world.apply(delta);
        REQUIRE(update.textureUploads.size() == 1);
        auto const handedOver = update.textureUploads[0].image == handle;
        CHECK(handedOver);
        delta = core::RenderWorldDelta{};
    }
    CHECK(handle.use_count() == 1);

    world.loseTextures();
    CHECK(world.takeLostTextures() == std::vector<core::TextureId>{core::TextureId{3}});
    CHECK(world.takeLostTextures().empty());
    CHECK(world.fullUpdate().textureUploads.empty());
}

} // namespace
