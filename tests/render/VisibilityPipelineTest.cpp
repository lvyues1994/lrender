#include "EnvironmentBrdf.h"
#include "RenderTestSupport.h"
#include "TestKernels.h"
#include "kernels/Visibility.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kSize = 64;
constexpr float kPi = 3.14159265358979f;

core::MeshId const kQuad{0};
core::MaterialId const kRed{0};
core::MaterialId const kBlue{1};
core::vec3 const kRedRadiance{0.6f, 0.1f, 0.05f};
core::vec3 const kBlueRadiance{0.05f, 0.2f, 0.7f};
core::vec3 const kSky{0.1f, 0.2f, 0.3f};

// A camera on +z looking at the origin, and quads facing it.
core::CameraDesc frontCamera() { return test::cameraAt(core::vec3{0.0f, 0.0f, 5.0f}, core::vec3{0.0f}); }

// What an emissive quad facing the camera shows: its emission and the sky its black
// dielectric surface reflects (F0 = 0.04 at the default roughness 0.5), some 2% of it.
std::uint32_t seenFacing(core::vec3 const &emission) {
    auto const scaleBias = render::environmentBrdf(1.0f, 0.5f);
    return test::rgba8(emission + kSky * (0.04f * scaleBias.x + scaleBias.y));
}

core::RenderWorldDelta quadScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{kQuad, test::share(core::makeQuad(core::vec2{0.5f, 0.5f}))}};
    delta.materials = {{kRed, test::emissive(kRedRadiance)}, {kBlue, test::emissive(kBlueRadiance)}};
    delta.environment = core::EnvironmentDesc{kSky};
    return delta;
}

// The pixels whose centres a quad of half size `half` at distance `distance` covers: the
// projection of [-half, half] is centred, so the covered columns and rows are the same.
struct Span {
    int first;
    int last;
};

Span coveredPixels(float const half, float const distance) {
    auto const focal = 1.0 / std::tan(0.4);                         // CameraDesc::verticalFov is 0.8
    auto const extent = focal * static_cast<double>(half / distance); // in NDC
    auto const low = (0.5 - extent * 0.5) * kSize;
    auto const high = (0.5 + extent * 0.5) * kSize;
    return Span{static_cast<int>(std::ceil(low - 0.5)), static_cast<int>(std::floor(high - 0.5))};
}

render::FrameDesc frameOf(core::RenderWorldDelta delta, render::ViewDesc const &view) {
    return render::FrameDesc{std::move(delta), {view}};
}

TEST_CASE("pixels no triangle covers show the sky") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});

    auto const result = harness.render(frameOf(quadScene(), test::viewOf(target, frontCamera())));

    REQUIRE(result.views.size() == 1);
    REQUIRE(result.views[0].image.has_value());
    auto const &image = *result.views[0].image;
    CHECK(image.extent.x == kSize);
    CHECK(image.extent.y == kSize);
    auto const sky = test::rgba8(kSky);
    CHECK(std::all_of(image.pixels.begin(), image.pixels.end(), [sky](std::uint32_t p) { return p == sky; }));
}

TEST_CASE("a quad covers exactly the pixels whose centres its projection contains") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto delta = quadScene();
    delta.instances = {{core::InstanceId{0}, kQuad, kRed, core::mat4{1.0f}}};

    auto const result = harness.render(frameOf(std::move(delta), test::viewOf(target, frontCamera())));

    auto const &image = *result.views[0].image;
    auto const span = coveredPixels(0.5f, 5.0f);
    auto const red = seenFacing(kRedRadiance);
    auto const sky = test::rgba8(kSky);
    auto wrong = 0;
    for (auto y = 0; y < kSize; ++y) {
        for (auto x = 0; x < kSize; ++x) {
            auto const inside = x >= span.first and x <= span.last and y >= span.first and y <= span.last;
            wrong += test::nearPixel(test::pixelAt(image, x, y), inside ? red : sky) ? 0 : 1;
        }
    }
    CHECK(span.last - span.first + 1 == 16);
    CHECK(wrong == 0);
    CHECK(result.views[0].stats.visibleInstances == 1);
    CHECK(result.views[0].stats.visibleTriangles == 2);
}

TEST_CASE("the nearer of two overlapping quads wins") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto delta = quadScene();
    auto const small = core::MeshId{1};
    delta.meshes.push_back({small, test::share(core::makeQuad(core::vec2{0.25f, 0.25f}))});
    // The blue quad comes first, so instance order alone would let the red one win.
    delta.instances = {{core::InstanceId{0}, small, kBlue, test::translation(core::vec3{0.0f, 0.0f, 1.0f})},
                       {core::InstanceId{1}, kQuad, kRed, core::mat4{1.0f}}};

    auto const result = harness.render(frameOf(std::move(delta), test::viewOf(target, frontCamera())));

    auto const &image = *result.views[0].image;
    auto const front = coveredPixels(0.25f, 4.0f);
    auto const back = coveredPixels(0.5f, 5.0f);
    CHECK(test::nearPixel(test::pixelAt(image, kSize / 2, kSize / 2), seenFacing(kBlueRadiance)));
    CHECK(test::nearPixel(test::pixelAt(image, front.first, front.first), seenFacing(kBlueRadiance)));
    CHECK(test::nearPixel(test::pixelAt(image, back.first, kSize / 2), seenFacing(kRedRadiance)));
}

TEST_CASE("instances outside the view are culled") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto delta = quadScene();
    delta.instances = {{core::InstanceId{0}, kQuad, kRed, test::translation(core::vec3{0.0f, 0.0f, 20.0f})},
                       {core::InstanceId{1}, kQuad, kRed, core::mat4{1.0f}},
                       {core::InstanceId{2}, kQuad, kBlue, test::translation(core::vec3{-50.0f, 0.0f, 0.0f})}};

    auto const result = harness.render(frameOf(std::move(delta), test::viewOf(target, frontCamera())));

    CHECK(result.views[0].stats.visibleInstances == 1);
    CHECK(result.views[0].stats.visibleTriangles == 2);
    CHECK(test::nearPixel(test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2), seenFacing(kRedRadiance)));
}

// A rough dielectric plane seen and lit straight from above: the normal, the view and the
// light coincide, so the BRDF reduces to (baseColor + F0 D V) / pi with F0 = 0.04,
// D = 1 / pi and V = 1/4 at roughness 1.
float straightDownRadiance(float const baseColor, float const irradiance) {
    return (baseColor + 0.04f * 0.25f) / kPi * irradiance;
}

core::RenderWorldDelta litPlane(core::LightDesc const &light) {
    auto delta = core::RenderWorldDelta{};
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{0.5f, 0.5f, 0.5f, 1.0f};
    material.roughness = 1.0f;
    delta.meshes = {{kQuad, test::share(core::makePlane(core::vec2{10.0f, 10.0f}))}};
    delta.materials = {{kRed, material}};
    delta.instances = {{core::InstanceId{0}, kQuad, kRed, core::mat4{1.0f}}};
    delta.lights = {{core::LightId{0}, light}};
    return delta;
}

core::CameraDesc overheadCamera() {
    return test::cameraAt(core::vec3{0.0f, 5.0f, 0.0f}, core::vec3{0.0f}, core::vec3{0.0f, 0.0f, -1.0f});
}

TEST_CASE("a directional light lights a plane as the BRDF says") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto sun = core::DirectionalLight{};
    sun.illuminance = 1.0f;

    auto const result = harness.render(frameOf(litPlane(sun), test::viewOf(target, overheadCamera())));

    auto const expected = test::srgb8(straightDownRadiance(0.5f, 1.0f));
    auto const centre = test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2);
    for (auto c = 0u; c < 3u; ++c) {
        CHECK(test::channel(centre, c) >= expected - 1);
        CHECK(test::channel(centre, c) <= expected + 1);
    }
}

// A negative scale reverses the winding of the instance's triangles: the visible side
// must stay visible and keep its outward normal, as glTF requires for mirrored nodes.
TEST_CASE("a mirrored instance keeps its front faces and its normals") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto sun = core::DirectionalLight{};
    sun.illuminance = 1.0f;
    auto delta = litPlane(sun);
    auto mirror = core::Transform{};
    mirror.scale = core::vec3{-1.0f, 1.0f, 1.0f};
    delta.instances[0].objectToWorld = core::matrixOf(mirror);

    auto const result = harness.render(frameOf(std::move(delta), test::viewOf(target, overheadCamera())));

    auto const expected = test::srgb8(straightDownRadiance(0.5f, 1.0f));
    auto const centre = test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2);
    for (auto c = 0u; c < 3u; ++c) {
        CHECK(test::channel(centre, c) >= expected - 1);
        CHECK(test::channel(centre, c) <= expected + 1);
    }
}

TEST_CASE("a point light falls off with the square of the distance") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto lamp = core::PointLight{};
    lamp.position = core::vec3{0.0f, 2.0f, 0.0f};
    lamp.intensity = 4.0f; // 4 / 2^2 = 1 at the point below it

    auto const result = harness.render(frameOf(litPlane(lamp), test::viewOf(target, overheadCamera())));

    auto const expected = test::srgb8(straightDownRadiance(0.5f, 1.0f));
    auto const centre = test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2);
    CHECK(test::channel(centre, 0) >= expected - 1);
    CHECK(test::channel(centre, 0) <= expected + 1);
    auto const corner = test::pixelAt(*result.views[0].image, 0, 0);
    CHECK(test::channel(corner, 0) < test::channel(centre, 0));
}

TEST_CASE("later deltas move, remove and add instances") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto const view = test::viewOf(target, frontCamera());
    auto first = quadScene();
    first.instances = {{core::InstanceId{7}, kQuad, kRed, core::mat4{1.0f}}};
    auto const red = seenFacing(kRedRadiance);
    auto const sky = test::rgba8(kSky);
    auto const centre = [](render::FrameResult const &result) {
        return test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2);
    };

    CHECK(test::nearPixel(centre(harness.render(frameOf(std::move(first), view))), red));

    auto moved = core::RenderWorldDelta{};
    moved.instances = {{core::InstanceId{7}, kQuad, kRed, test::translation(core::vec3{1.0f, 0.0f, 0.0f})}};
    auto const afterMove = harness.render(frameOf(std::move(moved), view));
    CHECK(centre(afterMove) == sky);
    CHECK(test::nearPixel(test::pixelAt(*afterMove.views[0].image, kSize / 2 + 14, kSize / 2), red));

    auto removed = core::RenderWorldDelta{};
    removed.removed.instances = {core::InstanceId{7}};
    auto const afterRemove = harness.render(frameOf(std::move(removed), view));
    CHECK(afterRemove.views[0].stats.visibleInstances == 0);
    CHECK(centre(afterRemove) == sky);

    auto added = core::RenderWorldDelta{};
    added.instances = {{core::InstanceId{8}, kQuad, kBlue, core::mat4{1.0f}}};
    CHECK(test::nearPixel(centre(harness.render(frameOf(std::move(added), view))), seenFacing(kBlueRadiance)));
}

TEST_CASE("one frame renders several views into their own targets") {
    auto harness = RenderHarness{};
    auto &targets = harness.renderer->targets();
    auto const wide = targets.createImage(render::ImageTargetDesc{core::ivec2{48, 24}});
    auto const square = targets.createImage(render::ImageTargetDesc{core::ivec2{32, 32}});
    auto delta = quadScene();
    delta.instances = {{core::InstanceId{0}, kQuad, kRed, core::mat4{1.0f}}};
    auto frame = render::FrameDesc{std::move(delta), {test::viewOf(wide, frontCamera(), core::ViewId{1}),
                                                      test::viewOf(square, frontCamera(), core::ViewId{2})}};

    auto const result = harness.render(std::move(frame));

    REQUIRE(result.views.size() == 2);
    CHECK(result.views[0].view == core::ViewId{1});
    CHECK(result.views[0].image->extent.x == 48);
    CHECK(result.views[0].image->extent.y == 24);
    CHECK(result.views[1].view == core::ViewId{2});
    CHECK(result.views[1].image->extent.x == 32);
    CHECK(test::nearPixel(test::pixelAt(*result.views[1].image, 16, 16), seenFacing(kRedRadiance)));
}

TEST_CASE("resizing a target changes the size of the next frame") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    harness.renderer->targets().resize(target, core::ivec2{20, 10});

    auto const result = harness.render(frameOf(quadScene(), test::viewOf(target, frontCamera())));

    CHECK(result.views[0].image->extent.x == 20);
    CHECK(result.views[0].image->extent.y == 10);
    CHECK(result.views[0].image->pixels.size() == 200);
}

// Overwrites the HDR colour with a constant, through a kernel of its own.
struct FillFeature final : render::RenderFeature {
    explicit FillFeature(core::vec4 const &colour_) : colour(colour_) {}

    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto &pass = graph.addPass("fill");
        pass.write(view.sceneColor);
        pass.record([colour = colour, image = view.sceneColor, extent = view.extent](render::PassContext &context) {
            auto &device = context.device();
            auto kernel = test::FillHdr{};
            device.useKernel(kernel);
            kernel.color.attach(context.image(image));
            device.bindImage(kernel.color);
            kernel.fill = colour;
            device.setUniform(kernel.fill);
            kernel.size = extent;
            device.setUniform(kernel.size);
            context.dispatch(kernel, tc::uvec3{static_cast<tc::uint>((extent.x + 7) / 8),
                                               static_cast<tc::uint>((extent.y + 7) / 8), 1});
        });
    }

    core::vec4 colour;
};

TEST_CASE("a pipeline extended with a feature runs the feature's own kernel") {
    auto harness = RenderHarness{};
    auto &pipelines = harness.renderer->pipelines();
    auto const filled = core::vec3{0.25f, 0.5f, 1.0f};
    auto const pipeline =
        pipelines.create(render::PipelineDesc{pipelines.visibility(), {std::make_shared<FillFeature>(core::vec4{filled, 1.0f})}});
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{16, 16}});
    auto view = test::viewOf(target, frontCamera());
    view.pipeline = pipeline;

    auto const result = harness.render(frameOf(quadScene(), view));

    auto const expected = test::rgba8(filled);
    auto const &pixels = result.views[0].image->pixels;
    CHECK(std::all_of(pixels.begin(), pixels.end(), [expected](std::uint32_t p) { return p == expected; }));
}

TEST_CASE("a frame that refers to an unknown mesh fails, and the next frame still renders") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto const view = test::viewOf(target, frontCamera());
    auto broken = quadScene();
    broken.instances = {{core::InstanceId{0}, core::MeshId{99}, kRed, core::mat4{1.0f}}};

    CHECK_THROWS_AS(harness.render(frameOf(std::move(broken), view)), std::invalid_argument);

    auto fine = quadScene();
    fine.instances = {{core::InstanceId{0}, kQuad, kRed, core::mat4{1.0f}}};
    CHECK(test::nearPixel(test::pixelAt(*harness.render(frameOf(std::move(fine), view)).views[0].image, kSize / 2, kSize / 2),
                          seenFacing(kRedRadiance)));
}

TEST_CASE("submitting a view of an unknown target throws at once") {
    auto harness = RenderHarness{};

    CHECK_THROWS_AS(harness.renderer->submit(frameOf(quadScene(), test::viewOf(core::RenderTargetId{42}, frontCamera()))),
                    std::invalid_argument);
}

// The GPU backends on test::deviceFor() their names against the CPU backend. Skipped where
// those devices are missing, unless LRENDER_TEST_REQUIRE_GPU is set.
TEST_CASE("every GPU backend renders what the CPU backend renders") {
    auto delta = [] {
        auto scene = quadScene();
        auto const small = core::MeshId{1};
        scene.meshes.push_back({small, test::share(core::makeSphere(0.4f, 24, 12))});
        auto lit = core::MaterialDesc{};
        lit.baseColor = core::vec4{0.7f, 0.5f, 0.3f, 1.0f};
        scene.materials.push_back({core::MaterialId{2}, lit});
        scene.instances = {{core::InstanceId{0}, kQuad, kRed, core::mat4{1.0f}},
                           {core::InstanceId{1}, small, core::MaterialId{2}, test::translation(core::vec3{0.3f, 0.2f, 1.0f})}};
        auto sun = core::DirectionalLight{};
        sun.direction = core::vec3{-0.3f, -0.5f, -1.0f};
        scene.lights = {{core::LightId{0}, sun}};
        return scene;
    };
    auto renderWith = [&delta](RenderHarness &harness) {
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
        return *harness.render(frameOf(delta(), test::viewOf(target, frontCamera()))).views[0].image;
    };
    auto cpu = RenderHarness{};
    auto const reference = renderWith(cpu);

    auto const requireGpu = std::getenv("LRENDER_TEST_REQUIRE_GPU") != nullptr;
    for (auto const &backend : tc::availableBackends()) {
        if (backend == "cpu") {
            continue;
        }
        CAPTURE(backend);
        auto harness = std::unique_ptr<RenderHarness>{};
        try {
            harness = std::make_unique<RenderHarness>(backend, test::deviceFor(backend));
        } catch (tc::DeviceError const &error) {
            if (requireGpu) {
                FAIL(error.what());
            }
            MESSAGE("skipped: " << error.what());
            continue;
        }
        CHECK(test::maxChannelDifference(renderWith(*harness), reference) <= 2);
    }
}

core::ImageHandle solidBc1Chain(std::uint16_t const colour, int const size) {
    auto image = core::ImageData{};
    image.extent = core::ivec2{size, size};
    image.compressed = tc::CompressedFormat::Bc1;
    for (auto level = 0u; level < core::fullLevelCount(image.extent); ++level) {
        auto &blocks = image.levels.emplace_back();
        // Endpoints differ, and the indices pick every colour of the block in turn.
        auto const other = static_cast<std::uint16_t>(colour ^ 0x18E3u);
        auto const high = std::max(colour, other);
        auto const low = std::min(colour, other);
        for (auto block = std::size_t{0}; block < core::levelBytes(image, level) / 8; ++block) {
            for (auto const byte : {high & 0xFF, high >> 8, low & 0xFF, low >> 8, 0x1B, 0xE4, 0x1B, 0xE4}) {
                blocks.push_back(static_cast<std::uint8_t>(byte));
            }
        }
    }
    return test::share(std::move(image));
}

// Textures (RGBA8 and BC1, a normal map), an alpha-masked lattice, a blended sphere, a
// sun with shadows, a point light and a sky.
core::RenderWorldDelta texturedScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makePlane(core::vec2{4.0f}))},
                    {core::MeshId{1}, test::share(core::makeBox(core::vec3{0.4f}))},
                    {core::MeshId{2}, test::share(core::makeSphere(0.4f, 24, 12))},
                    {core::MeshId{3}, test::share(core::makeQuad(core::vec2{0.6f}))}};
    delta.textures = {
        {core::TextureId{0}, test::rgba8Image(core::ivec2{64, 64}, core::ColorEncoding::Srgb,
                                              [](int const x, int const y) {
                                                  auto const light = (x / 8 + y / 8) % 2 == 0;
                                                  return light ? tc::ivec4{220, 210, 190, 255} : tc::ivec4{60, 70, 90, 255};
                                              }),
         {}},
        {core::TextureId{1}, solidBc1Chain(0xC3A6u, 16), {}},
        {core::TextureId{2}, test::rgba8Image(core::ivec2{16, 16}, core::ColorEncoding::Linear,
                                              [](int const x, int) {
                                                  return tc::ivec4{x < 8 ? 90 : 165, 128, 240, 255};
                                              }),
         {}},
        {core::TextureId{3}, test::rgba8Image(core::ivec2{32, 32}, core::ColorEncoding::Srgb,
                                              [](int const x, int const y) {
                                                  auto const hole = (x / 4 + y / 4) % 3 == 0;
                                                  return tc::ivec4{80, 160, 90, hole ? 0 : 255};
                                              }),
         {}}};
    auto ground = core::MaterialDesc{};
    ground.baseColorTexture = core::TextureId{0};
    ground.uvTransform.scale = core::vec2{3.0f};
    ground.roughness = 0.8f;
    auto crate = core::MaterialDesc{};
    crate.baseColorTexture = core::TextureId{1};
    crate.normalTexture = core::TextureId{2};
    auto glass = core::MaterialDesc{};
    glass.baseColor = core::vec4{0.7f, 0.85f, 1.0f, 0.35f};
    glass.alphaMode = core::AlphaMode::Blend;
    glass.roughness = 0.1f;
    auto lattice = core::MaterialDesc{};
    lattice.baseColorTexture = core::TextureId{3};
    lattice.alphaMode = core::AlphaMode::Mask;
    lattice.doubleSided = true;
    delta.materials = {{core::MaterialId{0}, ground},
                       {core::MaterialId{1}, crate},
                       {core::MaterialId{2}, glass},
                       {core::MaterialId{3}, lattice}};
    auto turned = core::Transform{};
    turned.translation = core::vec3{-0.8f, 0.4f, 0.0f};
    turned.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, 0.5f);
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}},
                       {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, core::matrixOf(turned)},
                       {core::InstanceId{2}, core::MeshId{2}, core::MaterialId{2},
                        test::translation(core::vec3{0.5f, 0.4f, 0.6f})},
                       {core::InstanceId{3}, core::MeshId{3}, core::MaterialId{3},
                        test::translation(core::vec3{0.4f, 0.6f, -0.8f})}};
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.illuminance = 2.5f;
    sun.castsShadows = true;
    auto lamp = core::PointLight{};
    lamp.position = core::vec3{0.6f, 1.4f, 1.2f};
    lamp.intensity = 2.0f;
    delta.lights = {{core::LightId{0}, sun}, {core::LightId{1}, lamp}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.1f, 0.15f, 0.25f}};
    return delta;
}

struct ImageComparison {
    std::uint32_t largest = 0;
    double meanDifference = 0.0; // per colour channel
    double farFraction = 0.0;    // of the pixels with a channel more than 2 apart
};

ImageComparison compare(render::ImageRgba8 const &a, render::ImageRgba8 const &b) {
    auto result = ImageComparison{};
    auto total = 0.0;
    auto far = 0;
    for (auto index = std::size_t{0}; index < a.pixels.size(); ++index) {
        auto worst = 0u;
        for (auto c = 0u; c < 3u; ++c) {
            auto const difference = test::channelDifference(a.pixels[index], b.pixels[index], c);
            worst = std::max(worst, difference);
            total += difference;
        }
        result.largest = std::max(result.largest, worst);
        far += worst > 2 ? 1 : 0;
    }
    result.meanDifference = total / (3.0 * static_cast<double>(a.pixels.size()));
    result.farFraction = static_cast<double>(far) / static_cast<double>(a.pixels.size());
    return result;
}

// Both pipelines on each GPU backend against the CPU backend. Sampling, filtering and
// BC1 decoding round differently on each device, by a level here and there (the demo
// scene at 960x540 differs by 0.06 levels on average on llvmpipe); rounding can also
// move a pixel to the other side of an edge, a mask's cutoff or a shadow's, which changes
// it by much more, so a few such pixels are allowed.
TEST_CASE("every GPU backend renders a textured, shadowed, transparent scene as the CPU backend does") {
    constexpr auto kPixels = 48;
    constexpr auto kSamples = 4;
    auto const camera = test::cameraAt(core::vec3{0.3f, 1.8f, 3.0f}, core::vec3{0.0f, 0.3f, 0.0f});
    auto renderWith = [&camera, extent = core::ivec2{kPixels, kPixels}](RenderHarness &harness) {
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{extent});
        auto raster = test::viewOf(target, camera);
        auto traced = test::viewOf(target, camera, core::ViewId{1});
        traced.pipeline = harness.renderer->pipelines().pathTraceReference();
        auto first = harness.render(render::FrameDesc{texturedScene(), {raster, traced}});
        auto last = first;
        for (auto frame = 1; frame < kSamples; ++frame) {
            last = harness.render(frameOf({}, traced));
        }
        return std::pair{std::move(*first.views[0].image), std::move(*last.views[0].image)};
    };
    auto cpu = RenderHarness{};
    auto const reference = renderWith(cpu);

    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        if (harness.renderer->deviceInfo().backend == "cpu") {
            return;
        }
        auto const images = renderWith(harness);
        for (auto const &difference : {compare(images.first, reference.first), compare(images.second, reference.second)}) {
            CAPTURE(difference.largest);
            CAPTURE(difference.meanDifference);
            CAPTURE(difference.farFraction);
            CHECK(difference.meanDifference < 0.25);
            CHECK(difference.farFraction < 0.01);
        }
    });
}

// Resources whose passes do not overlap share memory, which changes no pixel: on the CPU
// backend memory a resource takes over from another one holds a poison pattern until its
// first pass writes it, so a pass that read what it had not written would show it.
TEST_CASE("graph resources that share a heap render what resources of their own render, in less memory") {
    auto const camera = test::cameraAt(core::vec3{0.3f, 1.8f, 3.0f}, core::vec3{0.0f, 0.3f, 0.0f});
    auto const renderWith = [&camera](RenderHarness &harness) {
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{48, 40}});
        auto view = test::viewOf(target, camera);
        view.settings.antiAliasing = render::AntiAliasing::Taa;
        view.describeGraph = true;
        harness.render(render::FrameDesc{texturedScene(), {view}});
        harness.render(frameOf({}, view));
        return harness.render(frameOf({}, view));
    };
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        auto options = test::HarnessOptions{};
        options.backend = harness.renderer->deviceInfo().backend;
        options.device = test::deviceFor(options.backend);
        options.aliasGraphResources = false;
        auto separate = RenderHarness{options};
        auto const shared = renderWith(harness);
        auto const own = renderWith(separate);

        auto const &sharing = shared.views.at(0);
        auto const &owning = own.views.at(0);
        CHECK(test::maxChannelDifference(*sharing.image, *owning.image) == 0);
        CAPTURE(sharing.stats.graphBytes);
        CAPTURE(sharing.stats.unaliasedBytes);
        CHECK(sharing.stats.unaliasedBytes == owning.stats.unaliasedBytes);
        CHECK(owning.stats.graphBytes == owning.stats.unaliasedBytes);
        CHECK(sharing.stats.graphBytes < sharing.stats.unaliasedBytes);
        MESSAGE(options.backend << ": " << sharing.stats.graphBytes << " of " << sharing.stats.unaliasedBytes << " bytes");

        REQUIRE(sharing.graph.has_value());
        auto const &graph = *sharing.graph;
        CHECK(graph.rfind("digraph frame {", 0) == 0);
        CHECK(graph.find("cull instances") != std::string::npos);
        CHECK(graph.find("temporal resolve") != std::string::npos);
        CHECK(graph.find("heap at") != std::string::npos);
        CHECK(graph.find("history") != std::string::npos);
        CHECK(not owning.graph->empty());
        CHECK(owning.graph->find("heap at") == std::string::npos);
    });
}

// A grid of quads of 17 x 16 pixels, red and blue like a chequerboard, two triangles each
// whose pixel box is past render::kernels::largeTriangleArea(), and more triangles than a
// raster pass lists for its tile stage: every pixel still shows its own quad.
TEST_CASE("more large triangles than the tile stage lists are all rasterized") {
    constexpr auto kColumns = 128;
    constexpr auto kRows = 65;
    constexpr auto kWidth = 17;
    constexpr auto kHeight = 16;
    static_assert(2 * kColumns * kRows > render::kernels::largeTriangleCapacity());
    auto const extent = core::ivec2{kColumns * kWidth, kRows * kHeight};
    auto const pixel = 2.0f * 5.0f * std::tan(0.4f) / static_cast<float>(extent.y); // at the origin, seen from z 5
    auto delta = quadScene();
    for (auto row = 0; row < kRows; ++row) {
        for (auto column = 0; column < kColumns; ++column) {
            auto const centre = core::vec2{static_cast<float>(2 * column * kWidth + kWidth - extent.x),
                                           static_cast<float>(extent.y - 2 * row * kHeight - kHeight)} * 0.5f;
            auto const place = core::Transform{core::vec3{centre * pixel, 0.0f}, {},
                                               core::vec3{kWidth * pixel, kHeight * pixel, 1.0f}};
            delta.instances.push_back({core::InstanceId{static_cast<std::uint32_t>(row * kColumns + column)}, kQuad,
                                       (row + column) % 2 == 0 ? kRed : kBlue, core::matrixOf(place)});
        }
    }
    auto const red = seenFacing(kRedRadiance);
    auto const blue = seenFacing(kBlueRadiance);
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        // The CPU backend runs the same code, but would take most of a minute over these
        // 2.2 million pixels in a debug build.
        if (harness.renderer->deviceInfo().backend == "cpu") {
            return;
        }
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{extent});
        auto const result = harness.render(frameOf(delta, test::viewOf(target, frontCamera())));
        auto const &image = *result.views.at(0).image;
        auto wrong = 0;
        for (auto y = 0; y < extent.y; ++y) {
            for (auto x = 0; x < extent.x; ++x) {
                auto const expected = (y / kHeight + x / kWidth) % 2 == 0 ? red : blue;
                wrong += test::nearPixel(test::pixelAt(image, x, y), expected) ? 0 : 1;
            }
        }
        CHECK(wrong == 0);
        CHECK(result.views[0].stats.visibleTriangles == 2 * kColumns * kRows);
    });
}

} // namespace
