#include "ScriptTestSupport.h"
#include "asset/AssetTestSupport.h"
#include "render/RenderTestSupport.h"

#include <co2/co2.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace lr::scene;
using lr::asset::AssetLibrary;
using lr::asset::AssetResult;
using lr::core::MeshHandle;
using lr::core::Transform;
using lr::core::vec3;

namespace test = lr::test;
namespace core = lr::core;
namespace render = lr::render;

// A circle of radius 4 around the origin, looking at it.
Transform orbitAround(float const angle) {
    auto const eye = vec3{4.0f * std::sin(angle), 1.0f, 4.0f * std::cos(angle)};
    auto transform = Transform{};
    transform.translation = eye;
    transform.rotation = core::Rotation::lookingAlong(-eye, vec3{0.0f, 1.0f, 0.0f});
    return transform;
}

// The script of docs/architecture.md, as it is written there.
auto orbitCamera(World &world, AssetLibrary &assets, EntityId const camera)
    CO2_BEG(ScriptTask, (world, assets, camera), AssetResult<MeshHandle> mesh; float angle{};) {
    CO2_AWAIT_SET(mesh, assets.loadMesh("models/bunny.glb"));
    if (mesh) world.spawnInstance(*mesh, Transform::identity());
    for (angle = 0.0f;; angle += 0.01f) {
        world.setTransform(camera, orbitAround(angle));
        CO2_AWAIT(nextFrame());
    }
}
CO2_END

test::GltfFiles quadFiles(std::string const &stem) {
    auto gltf = test::TestGltf{};
    gltf.meshes = {test::quadPrimitive(1.0f)};
    gltf.materials = "[{}]";
    return test::writeGltf(gltf, test::GltfLayout::Binary, stem);
}

// Ticks a frame at a time, the loads going on meanwhile, until `done` or for 10 seconds.
template <class Done>
bool tickUntil(ScriptHost &host, Done const &done) {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (not done() and std::chrono::steady_clock::now() < deadline) {
        host.tick(std::chrono::milliseconds{16});
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return done();
}

TEST_CASE("the architecture document's orbitCamera loads a mesh, spawns it and turns the camera each frame") {
    auto const files = quadFiles("models/bunny");
    auto assets = test::AssetHarness{files.files};
    auto world = makeWorld(WorldConfig{});
    auto host = makeScriptHost(ScriptHostConfig{});
    auto const camera = world->createEntity(Transform::identity(), std::nullopt);
    world->setCamera(camera, CameraLens{});

    host->start(orbitCamera(*world, *assets.library, camera));

    REQUIRE(tickUntil(*host, [&] { return world->entityCount() == 2; }));
    auto const before = world->cameraOf(camera);
    host->tick(std::chrono::milliseconds{16});
    auto const after = world->cameraOf(camera);
    CHECK(tc::length(after.position - before.position) > 0.01f);
    CHECK(tc::length(after.position) == doctest::Approx(std::sqrt(17.0f)));
    auto const delta = world->extractDelta();
    CHECK(delta.meshes.size() == 1);
    CHECK(delta.instances.size() == 1);
    CHECK(delta.materials.size() == 1);
    CHECK(host->takeFailures().empty());

    host->shutdown();
    CHECK(host->isIdle());
}

// Puts the file's quad, glowing, under `parent`, offset by `offset`.
auto spawnGlowingQuad(World &world, AssetLibrary &assets, EntityId const parent, vec3 const offset, vec3 const glow)
    CO2_BEG(ScriptTask, (world, assets, parent, offset, glow), AssetResult<MeshHandle> mesh;) {
    CO2_AWAIT_SET(mesh, assets.loadMesh("models/quad.glb"));
    if (not mesh) {
        throw std::runtime_error{mesh.error().message};
    }
    {
        auto &resources = world.resources();
        auto const meshId = resources.addMesh(*mesh);
        auto const material = resources.addMaterial(test::emissive(glow));
        auto local = Transform{};
        local.translation = offset;
        auto const quad = world.createEntity(local, parent);
        world.setRenderable(quad, Renderable{meshId, material});
        resources.releaseMesh(meshId);
        resources.releaseMaterial(material);
    }
}
CO2_END

constexpr auto kSize = 32;

// The quad fills the middle half of the view, glowing; the rest is the black sky.
std::uint32_t largestError(render::ImageRgba8 const &image, vec3 const &glow) {
    auto largest = std::uint32_t{0};
    for (auto y = 0; y < kSize; ++y) {
        for (auto x = 0; x < kSize; ++x) {
            auto const inside = x >= kSize / 4 and x < 3 * kSize / 4 and y >= kSize / 4 and y < 3 * kSize / 4;
            auto const want = test::rgba8(inside ? glow : vec3{0.0f});
            auto const pixel = test::pixelAt(image, x, y);
            for (auto channel = 0u; channel < 3u; ++channel) {
                largest = std::max(largest, test::channelDifference(pixel, want, channel));
            }
        }
    }
    return largest;
}

TEST_CASE("a script spawns a loaded glTF mesh into the world, whose delta renders it on every backend") {
    auto const files = quadFiles("models/quad");
    auto assets = test::AssetHarness{files.files};
    auto world = makeWorld(WorldConfig{});
    auto host = makeScriptHost(ScriptHostConfig{});
    auto parentTransform = Transform{};
    parentTransform.translation = vec3{1.0f, 0.0f, 0.0f};
    auto const parent = world->createEntity(parentTransform, std::nullopt);
    auto const glow = vec3{0.25f, 0.5f, 0.75f};
    host->start(spawnGlowingQuad(*world, *assets.library, parent, vec3{-1.0f, 0.0f, 0.0f}, glow));

    REQUIRE(tickUntil(*host, [&] { return host->isIdle(); }));
    REQUIRE(host->takeFailures().empty());
    auto const halfFilling = test::halfFillingCamera();
    auto cameraTransform = Transform{};
    cameraTransform.translation = halfFilling.position;
    auto const camera = world->createEntity(cameraTransform, std::nullopt);
    auto lens = CameraLens{};
    lens.verticalFov = halfFilling.verticalFov;
    world->setCamera(camera, lens);
    auto const delta = world->extractDelta();
    REQUIRE(delta.instances.size() == 1);

    auto images = std::map<std::string, render::ImageRgba8>{};
    test::forEachBackend(test::HarnessOptions{}, [&](test::RenderHarness &harness) {
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize}});
        auto const frame = harness.render(test::frameOf(delta, test::viewOf(target, world->cameraOf(camera))));
        REQUIRE(frame.views.at(0).image.has_value());
        CHECK(largestError(*frame.views[0].image, glow) <= 1);
        images.emplace(harness.renderer->deviceInfo().backend, *frame.views[0].image);
    });
    REQUIRE(images.count("cpu") == 1);
    for (auto const &entry : images) {
        auto const &backend = entry.first;
        CAPTURE(backend);
        CHECK(test::maxChannelDifference(entry.second, images.at("cpu")) <= 1);
    }
}

} // namespace
