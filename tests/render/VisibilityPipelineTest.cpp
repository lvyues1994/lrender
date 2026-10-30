#include "RenderTestSupport.h"
#include "TestKernels.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <cmath>
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
    auto const red = test::rgba8(kRedRadiance);
    auto const sky = test::rgba8(kSky);
    auto wrong = 0;
    for (auto y = 0; y < kSize; ++y) {
        for (auto x = 0; x < kSize; ++x) {
            auto const inside = x >= span.first and x <= span.last and y >= span.first and y <= span.last;
            wrong += test::pixelAt(image, x, y) == (inside ? red : sky) ? 0 : 1;
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
    CHECK(test::pixelAt(image, kSize / 2, kSize / 2) == test::rgba8(kBlueRadiance));
    CHECK(test::pixelAt(image, front.first, front.first) == test::rgba8(kBlueRadiance));
    CHECK(test::pixelAt(image, back.first, kSize / 2) == test::rgba8(kRedRadiance));
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
    CHECK(test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2) == test::rgba8(kRedRadiance));
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
    auto const red = test::rgba8(kRedRadiance);
    auto const sky = test::rgba8(kSky);
    auto const centre = [](render::FrameResult const &result) {
        return test::pixelAt(*result.views[0].image, kSize / 2, kSize / 2);
    };

    CHECK(centre(harness.render(frameOf(std::move(first), view))) == red);

    auto moved = core::RenderWorldDelta{};
    moved.instances = {{core::InstanceId{7}, kQuad, kRed, test::translation(core::vec3{1.0f, 0.0f, 0.0f})}};
    auto const afterMove = harness.render(frameOf(std::move(moved), view));
    CHECK(centre(afterMove) == sky);
    CHECK(test::pixelAt(*afterMove.views[0].image, kSize / 2 + 14, kSize / 2) == red);

    auto removed = core::RenderWorldDelta{};
    removed.removed.instances = {core::InstanceId{7}};
    auto const afterRemove = harness.render(frameOf(std::move(removed), view));
    CHECK(afterRemove.views[0].stats.visibleInstances == 0);
    CHECK(centre(afterRemove) == sky);

    auto added = core::RenderWorldDelta{};
    added.instances = {{core::InstanceId{8}, kQuad, kBlue, core::mat4{1.0f}}};
    CHECK(centre(harness.render(frameOf(std::move(added), view))) == test::rgba8(kBlueRadiance));
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
    CHECK(test::pixelAt(*result.views[1].image, 16, 16) == test::rgba8(kRedRadiance));
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
            device.execute(kernel, tc::uvec3{static_cast<tc::uint>((extent.x + 7) / 8),
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
    CHECK(test::pixelAt(*harness.render(frameOf(std::move(fine), view)).views[0].image, kSize / 2, kSize / 2)
          == test::rgba8(kRedRadiance));
}

TEST_CASE("submitting a view of an unknown target throws at once") {
    auto harness = RenderHarness{};

    CHECK_THROWS_AS(harness.renderer->submit(frameOf(quadScene(), test::viewOf(core::RenderTargetId{42}, frontCamera()))),
                    std::invalid_argument);
}

// The GPU backends on Mesa's software devices against the CPU backend. Skipped where those
// devices are missing, unless LRENDER_TEST_REQUIRE_GPU is set.
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
            harness = std::make_unique<RenderHarness>(backend, "llvmpipe");
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

} // namespace
