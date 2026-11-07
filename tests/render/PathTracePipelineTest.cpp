#include "EnvironmentBrdf.h"
#include "RenderTestSupport.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>
#include <lr/render/kernels/Lighting.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kSize = 64;
constexpr float kPi = 3.14159265358979f;

render::FrameDesc frameOf(core::RenderWorldDelta delta, render::ViewDesc const &view) {
    return render::FrameDesc{std::move(delta), {view}};
}

render::ViewDesc pathTraced(RenderHarness &harness, core::RenderTargetId const target, core::CameraDesc const &camera,
                            core::ViewId const id = core::ViewId{0}) {
    auto view = test::viewOf(target, camera, id);
    view.pipeline = harness.renderer->pipelines().pathTraceReference();
    return view;
}

core::MaterialDesc grey(float const baseColor, bool const doubleSided = false) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{baseColor, baseColor, baseColor, 1.0f};
    material.roughness = 1.0f;
    material.doubleSided = doubleSided;
    return material;
}

// What the visibility pipeline's BRDF reflects towards a viewer at cos(theta) = nDotV
// from a uniform sky of radiance 1: the integral over the hemisphere, by the midpoint
// rule (the integrand is smooth at roughness 1).
float reflectedFromUniformSky(float const nDotV, core::MaterialDesc const &material) {
    constexpr auto kSteps = 48;
    auto const normal = core::vec3{0.0f, 0.0f, 1.0f};
    auto const toViewer = core::vec3{std::sqrt(1.0f - nDotV * nDotV), 0.0f, nDotV};
    auto const step = kPi / 2.0f / kSteps;
    auto sum = 0.0;
    for (auto i = 0; i < kSteps; ++i) {
        auto const theta = (static_cast<float>(i) + 0.5f) * step;
        for (auto j = 0; j < 4 * kSteps; ++j) {
            auto const phi = (static_cast<float>(j) + 0.5f) * step;
            auto const toLight =
                core::vec3{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
            auto const reflected =
                render::kernels::reflectedLight(normal, toViewer, toLight, core::vec3{1.0f}, material.baseColor.xyz(),
                                                material.metallic, material.roughness);
            sum += static_cast<double>(reflected.x * std::sin(theta) * step * step);
        }
    }
    return static_cast<float>(sum);
}

// The camera ray through the centre of a pixel of a square view.
core::vec3 centreRay(core::CameraDesc const &camera, int const x, int const y, int const size) {
    auto const scale = std::tan(camera.verticalFov / 2.0f);
    auto const ndcX = (static_cast<float>(x) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f;
    auto const ndcY = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(size) * 2.0f;
    auto const forward = tc::normalize(camera.target - camera.position);
    auto const side = tc::normalize(tc::cross(forward, camera.up));
    auto const up = tc::cross(side, forward);
    return tc::normalize(forward + side * (ndcX * scale) + up * (ndcY * scale));
}

// A diffuse sphere in a uniform sky reflects a fraction `baseColor` of the sky, plus
// what its specular lobe reflects (about 1.3% of the sky facing the viewer, more
// towards the rim), and nothing else: being convex, it never sees itself.
TEST_CASE("a rough sphere in a uniform sky reflects what its BRDF integrates to (white furnace)") {
    constexpr auto kPixels = 24;
    constexpr auto kSamples = 512;
    constexpr auto kSky = 1.0f;
    auto options = test::HarnessOptions{};
    options.framePoolThreads = 4;
    auto harness = RenderHarness{options};
    auto &pipelines = harness.renderer->pipelines();
    auto const probe = std::make_shared<test::HdrProbe>();
    auto const camera = test::cameraAt(core::vec3{0.0f, 0.0f, 4.0f}, core::vec3{0.0f});
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kPixels, kPixels}});
    auto view = test::viewOf(target, camera);
    view.pipeline = pipelines.create(render::PipelineDesc{pipelines.pathTraceReference(), {probe}});
    auto const material = grey(0.5f);
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeSphere(1.0f, 64, 32))}};
    delta.materials = {{core::MaterialId{0}, material}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    delta.environment = core::EnvironmentDesc{core::vec3{kSky}};

    auto last = harness.render(frameOf(std::move(delta), view));
    for (auto frame = 1; frame < kSamples; ++frame) {
        last = harness.render(frameOf(core::RenderWorldDelta{}, view));
    }

    CHECK(last.views[0].stats.accumulatedSamples == kSamples);
    auto const &pixels = *probe->pixels;
    REQUIRE(pixels.size() == kPixels * kPixels);
    auto sphere = 0;
    auto sky = 0;
    auto worst = 0.0f;
    auto relativeSum = 0.0;
    auto squares = 0.0;
    for (auto y = 0; y < kPixels; ++y) {
        for (auto x = 0; x < kPixels; ++x) {
            auto const measured = pixels[static_cast<std::size_t>(y * kPixels + x)].x;
            auto const direction = centreRay(camera, x, y, kPixels);
            auto const along = -tc::dot(camera.position, direction);
            auto const miss = std::sqrt(std::max(tc::dot(camera.position, camera.position) - along * along, 0.0f));
            if (miss > 1.3f) {
                // Two pixels and more off the rim: only sky.
                CHECK(measured == doctest::Approx(kSky).epsilon(1e-3));
                ++sky;
                continue;
            }
            auto const hit = camera.position + direction * (along - std::sqrt(std::max(1.0f - miss * miss, 0.0f)));
            auto const nDotV = -tc::dot(hit, direction);
            if (miss < 1.0f and nDotV >= 0.5f) {
                auto const expected = kSky * reflectedFromUniformSky(nDotV, material);
                auto const relative = (measured - expected) / expected;
                squares += static_cast<double>(relative * relative);
                worst = std::max(worst, std::abs(relative));
                relativeSum += static_cast<double>(relative);
                ++sphere;
            }
        }
    }
    // The pixel means scatter by about 1% (1 sigma) of the radiance: one sample in twenty
    // takes the GGX lobe, and at roughness 1 half of those reflect below the horizon and
    // count nothing. The mean over some 120 pixels is then good to about 0.1%, and leaving
    // the specular lobe out would put it 2.5% low. The geometry adds less than 0.1%: the
    // tessellated sphere's normals, and the paths it ends where a sampled direction points
    // into its facets.
    auto const mean = relativeSum / sphere;
    auto const spread = std::sqrt(squares / sphere);
    CAPTURE(mean);
    CAPTURE(spread);
    CAPTURE(worst);
    CHECK(sphere > 100);
    CHECK(sky > 100);
    CHECK(std::abs(mean) < 0.005);
    CHECK(spread < 0.02);
    CHECK(worst < 0.05f);
}

// What reflectedLight()'s specular term integrates to over the hemisphere for F0 = 0 and
// F0 = 1, by the midpoint rule on a fine grid: the reference for the importance-sampled
// table. Low roughness needs a finer grid around the mirror direction, so the test stays
// at roughness 0.4 and up.
core::vec2 specularAlbedo(float const nDotV, float const roughness) {
    constexpr auto kSteps = 200;
    auto const normal = core::vec3{0.0f, 0.0f, 1.0f};
    auto const toViewer = core::vec3{std::sqrt(1.0f - nDotV * nDotV), 0.0f, nDotV};
    auto const step = kPi / 2.0f / kSteps;
    auto white = 0.0;
    auto black = 0.0;
    for (auto i = 0; i < kSteps; ++i) {
        auto const theta = (static_cast<float>(i) + 0.5f) * step;
        for (auto j = 0; j < 4 * kSteps; ++j) {
            auto const phi = (static_cast<float>(j) + 0.5f) * step;
            auto const toLight =
                core::vec3{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
            auto const weight = static_cast<double>(std::sin(theta) * step * step);
            white += weight * render::kernels::reflectedLight(normal, toViewer, toLight, core::vec3{1.0f},
                                                              core::vec3{1.0f}, 1.0f, roughness)
                                  .x;
            black += weight * render::kernels::reflectedLight(normal, toViewer, toLight, core::vec3{1.0f},
                                                              core::vec3{0.0f}, 1.0f, roughness)
                                  .x;
        }
    }
    return core::vec2{static_cast<float>(white - black), static_cast<float>(black)};
}

// The raster pipeline's ambient specular reads (A, B) from a table built by importance
// sampling; against brute force it is good to a fraction of a percent of F0 A + B.
TEST_CASE("the environment table holds what the shading BRDF integrates to") {
    auto worst = 0.0f;
    for (auto const roughness : {0.4f, 1.0f}) {
        for (auto const nDotV : {0.25f, 0.6f, 1.0f}) {
            CAPTURE(roughness);
            CAPTURE(nDotV);
            auto const exact = specularAlbedo(nDotV, roughness);
            auto const table = render::environmentBrdf(nDotV, roughness);
            for (auto const f0 : {0.04f, 1.0f}) {
                auto const expected = f0 * exact.x + exact.y;
                auto const relative = std::abs(f0 * table.x + table.y - expected) / expected;
                worst = std::max(worst, relative);
                CHECK(relative < 0.01f);
            }
        }
    }
    MESSAGE("environment table: largest relative error " << worst);
}

// One sphere in a uniform sky, as in the white furnace above. Being convex, it never sees
// itself, so the path tracer's mean is the directional albedo of the BRDF, which the
// raster pipeline's ambient term reads from the table: diffuse exactly, specular as
// F0 A + B. The table is good to 0.25% (above) and the path tracer's mean over some 300
// pixel channels to about 0.1%, so the mean difference stays within 0.5% (measured 0.04%
// for the metal, 0.06% for the dielectric). Single pixels scatter by the path tracer's
// noise at 256 samples, about 1.5% (1 sigma) for the dielectric, whose paths pick between
// the diffuse and the specular lobe.
TEST_CASE("the visibility pipeline reflects a uniform sky as the path tracer does, for a metal and a dielectric") {
    constexpr auto kPixels = 24;
    constexpr auto kSamples = 256;
    auto metal = core::MaterialDesc{};
    metal.baseColor = core::vec4{0.95f, 0.64f, 0.54f, 1.0f};
    metal.metallic = 1.0f;
    metal.roughness = 0.35f;
    auto dielectric = grey(0.5f);
    dielectric.roughness = 0.6f;
    for (auto const &material : {metal, dielectric}) {
        CAPTURE(material.metallic);
        auto options = test::HarnessOptions{};
        options.framePoolThreads = 4;
        auto harness = RenderHarness{options};
        auto &pipelines = harness.renderer->pipelines();
        auto const camera = test::cameraAt(core::vec3{0.0f, 0.0f, 4.0f}, core::vec3{0.0f});
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kPixels, kPixels}});
        auto const rasterProbe = std::make_shared<test::HdrProbe>();
        auto const tracedProbe = std::make_shared<test::HdrProbe>();
        auto raster = test::viewOf(target, camera);
        raster.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {rasterProbe}});
        auto traced = test::viewOf(target, camera, core::ViewId{1});
        traced.pipeline = pipelines.create(render::PipelineDesc{pipelines.pathTraceReference(), {tracedProbe}});
        auto delta = core::RenderWorldDelta{};
        delta.meshes = {{core::MeshId{0}, test::share(core::makeSphere(1.0f, 64, 32))}};
        delta.materials = {{core::MaterialId{0}, material}};
        delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
        delta.environment = core::EnvironmentDesc{core::vec3{1.0f}};

        harness.render(render::FrameDesc{std::move(delta), {raster, traced}});
        for (auto frame = 1; frame < kSamples; ++frame) {
            harness.render(frameOf({}, traced));
        }

        auto sphere = 0;
        auto worst = 0.0f;
        auto sum = 0.0;
        auto squares = 0.0;
        for (auto y = 0; y < kPixels; ++y) {
            for (auto x = 0; x < kPixels; ++x) {
                auto const direction = centreRay(camera, x, y, kPixels);
                auto const along = -tc::dot(camera.position, direction);
                auto const miss = std::sqrt(std::max(tc::dot(camera.position, camera.position) - along * along, 0.0f));
                auto const hit = camera.position + direction * (along - std::sqrt(std::max(1.0f - miss * miss, 0.0f)));
                // Pixels nearer the rim than that cover some sky in the path tracer's samples.
                if (miss >= 1.0f or -tc::dot(hit, direction) < 0.5f) {
                    continue;
                }
                auto const index = static_cast<std::size_t>(y * kPixels + x);
                for (auto c = 0; c < 3; ++c) {
                    auto const expected = (*tracedProbe->pixels)[index][c];
                    auto const relative = ((*rasterProbe->pixels)[index][c] - expected) / expected;
                    worst = std::max(worst, std::abs(relative));
                    sum += static_cast<double>(relative);
                    squares += static_cast<double>(relative * relative);
                }
                ++sphere;
            }
        }
        auto const mean = sum / (3.0 * sphere);
        auto const spread = std::sqrt(squares / (3.0 * sphere));
        CAPTURE(mean);
        CAPTURE(spread);
        CAPTURE(worst);
        CHECK(sphere > 80);
        CHECK(std::abs(mean) < 0.005);
        CHECK(spread < 0.02);
        CHECK(worst < 0.07f);
    }
}

// A rough dielectric plane seen and lit straight from above: the normal, the view and the
// light coincide, so the BRDF reduces to (baseColor + F0 D V) / pi with F0 = 0.04,
// D = 1 / pi and V = 1/4 at roughness 1.
float straightDownRadiance(float const baseColor, float const irradiance) {
    return (baseColor + 0.04f * 0.25f) / kPi * irradiance;
}

core::RenderWorldDelta litPlane(core::LightDesc const &light) {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makePlane(core::vec2{10.0f, 10.0f}))}};
    delta.materials = {{core::MaterialId{0}, grey(0.5f)}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    delta.lights = {{core::LightId{0}, light}};
    return delta;
}

core::CameraDesc overheadCamera() {
    return test::cameraAt(core::vec3{0.0f, 5.0f, 0.0f}, core::vec3{0.0f}, core::vec3{0.0f, 0.0f, -1.0f});
}

struct BothPipelines {
    render::ImageRgba8 visibility;
    render::ImageRgba8 pathTraced;
};

// A frame of the visibility pipeline, then `samples` frames of the path tracer.
BothPipelines renderBoth(core::RenderWorldDelta delta, core::CameraDesc const &camera, int const samples = 1) {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto visibility = harness.render(frameOf(std::move(delta), test::viewOf(target, camera)));
    auto traced = render::FrameResult{};
    for (auto frame = 0; frame < samples; ++frame) {
        traced = harness.render(frameOf(core::RenderWorldDelta{}, pathTraced(harness, target, camera)));
    }
    return BothPipelines{std::move(*visibility.views[0].image), std::move(*traced.views[0].image)};
}

// Lights are delta lights met by next-event estimation, and the plane cannot light
// itself, so every sample is exact for the point it sees. The visibility pipeline shades
// the centre of each pixel and the path tracer points anywhere in it, which under the
// point light's falloff can be two levels apart for one sample, so the comparison takes
// the mean of 16.
TEST_CASE("the path tracer lights a plane as the visibility pipeline does") {
    auto sun = core::DirectionalLight{};
    sun.illuminance = 1.0f;
    auto lamp = core::PointLight{};
    lamp.position = core::vec3{0.0f, 2.0f, 0.0f};
    lamp.intensity = 4.0f; // 4 / 2^2 = 1 at the point below it
    for (auto const &light : {core::LightDesc{sun}, core::LightDesc{lamp}}) {
        auto const images = renderBoth(litPlane(light), overheadCamera(), 16);

        auto const expected = test::srgb8(straightDownRadiance(0.5f, 1.0f));
        auto const centre = test::pixelAt(images.pathTraced, kSize / 2, kSize / 2);
        for (auto c = 0u; c < 3u; ++c) {
            CHECK(test::channel(centre, c) >= expected - 1);
            CHECK(test::channel(centre, c) <= expected + 1);
        }
        CHECK(test::maxChannelDifference(images.pathTraced, images.visibility) <= 1);
    }
}

TEST_CASE("shadow rays darken what the visibility pipeline lights") {
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{1.0f, -1.0f, 0.0f};
    auto delta = litPlane(sun);
    // A black box floating over the plane, between the sun and the plane right of it.
    delta.meshes.push_back({core::MeshId{1}, test::share(core::makeBox(core::vec3{0.5f}))});
    delta.materials.push_back({core::MaterialId{1}, grey(0.0f)});
    delta.instances.push_back(
        {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, test::translation(core::vec3{0.0f, 1.5f, 0.0f})});

    auto const images = renderBoth(std::move(delta), overheadCamera());

    // Pixel columns 50 and 13 look at x = 1.25 and x = -1.25 on the plane, past the box.
    auto const shadowed = test::pixelAt(images.pathTraced, 50, kSize / 2);
    auto const lit = test::pixelAt(images.pathTraced, 13, kSize / 2);
    CHECK(test::channel(test::pixelAt(images.visibility, 50, kSize / 2), 0) > 80);
    CHECK(test::channel(shadowed, 0) <= 2);
    for (auto c = 0u; c < 3u; ++c) {
        CHECK(test::channelDifference(lit, test::pixelAt(images.visibility, 13, kSize / 2), c) <= 1);
    }
}

TEST_CASE("single-sided surfaces are invisible from behind and double-sided ones face the viewer, in both pipelines") {
    auto const behind = test::cameraAt(core::vec3{0.0f, 0.0f, -5.0f}, core::vec3{0.0f});
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{0.0f, 0.0f, 1.0f}; // from the camera's side
    auto const sky = core::vec3{0.1f, 0.2f, 0.3f};
    for (auto const doubleSided : {false, true}) {
        CAPTURE(doubleSided);
        auto delta = core::RenderWorldDelta{};
        delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{0.5f}))}};
        delta.materials = {{core::MaterialId{0}, grey(0.5f, doubleSided)}};
        delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
        delta.lights = {{core::LightId{0}, sun}};
        if (not doubleSided) {
            delta.environment = core::EnvironmentDesc{sky};
        }

        auto const images = renderBoth(std::move(delta), behind);

        auto const expected = doubleSided ? test::rgba8(core::vec3{straightDownRadiance(0.5f, 1.0f)}) : test::rgba8(sky);
        auto const centre = test::pixelAt(images.pathTraced, kSize / 2, kSize / 2);
        for (auto c = 0u; c < 3u; ++c) {
            CHECK(test::channelDifference(centre, expected, c) <= 1);
            CHECK(test::channelDifference(test::pixelAt(images.visibility, kSize / 2, kSize / 2), expected, c) <= 1);
        }
    }
}

TEST_CASE("accumulation keeps counting until the camera, the extent or the scene changes") {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{8, 8}});
    auto const camera = test::cameraAt(core::vec3{0.0f, 2.0f, 5.0f}, core::vec3{0.0f});
    auto const view = pathTraced(harness, target, camera);
    auto const samples = [&harness](render::FrameDesc frame) {
        auto const result = harness.render(std::move(frame));
        return result.views.at(0).stats.accumulatedSamples;
    };
    auto sun = core::DirectionalLight{};

    CHECK(samples(frameOf(litPlane(sun), view)) == 1);
    CHECK(samples(frameOf({}, view)) == 2);
    CHECK(samples(frameOf({}, view)) == 3);

    auto moved = view;
    moved.camera.position.x += 0.5f;
    CHECK(samples(frameOf({}, moved)) == 1);
    CHECK(samples(frameOf({}, moved)) == 2);

    // Another pipeline in between leaves the accumulation as it was.
    auto raster = moved;
    raster.pipeline = harness.renderer->pipelines().visibility();
    CHECK(samples(frameOf({}, raster)) == 0);
    CHECK(samples(frameOf({}, moved)) == 3);

    auto brighter = core::RenderWorldDelta{};
    sun.illuminance = 2.0f;
    brighter.lights = {{core::LightId{0}, sun}};
    CHECK(samples(frameOf(std::move(brighter), moved)) == 1);
    CHECK(samples(frameOf({}, moved)) == 2);

    harness.renderer->targets().resize(target, core::ivec2{6, 4});
    CHECK(samples(frameOf({}, moved)) == 1);

    // Views keep their own accumulation.
    auto const other = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{8, 8}});
    auto const both = harness.render(render::FrameDesc{{}, {moved, pathTraced(harness, other, camera, core::ViewId{1})}});
    CHECK(both.views.at(0).stats.accumulatedSamples == 2);
    CHECK(both.views.at(1).stats.accumulatedSamples == 1);
}

// Fills the HDR colour by calling the device directly instead of through the context.
struct UncountedFill final : render::RenderFeature {
    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto &pass = graph.addPass("uncounted fill");
        pass.write(view.sceneColor);
        pass.record([image = view.sceneColor, extent = view.extent](render::PassContext &context) {
            auto &device = context.device();
            auto kernel = test::FillHdr{};
            device.useKernel(kernel);
            kernel.color.attach(context.image(image));
            device.bindImage(kernel.color);
            kernel.fill = core::vec4{1.0f};
            device.setUniform(kernel.fill);
            kernel.size = extent;
            device.setUniform(kernel.size);
            device.execute(kernel, tc::uvec3{1, 1, 1});
        });
    }
};

std::vector<std::string> passNames(render::ViewResult const &view) {
    auto names = std::vector<std::string>{};
    for (auto const &pass : view.passTimings) {
        names.push_back(pass.pass);
    }
    return names;
}

bool allTimed(render::ViewResult const &view) {
    return std::all_of(view.passTimings.begin(), view.passTimings.end(), [](render::PassTiming const &pass) {
        return pass.milliseconds.has_value() and *pass.milliseconds >= 0.0;
    });
}

TEST_CASE("with device timers on, every pass reports the time of its dispatches") {
    auto options = test::HarnessOptions{};
    options.gpuTimers = true;
    options.singlePassVisibility = render::VariantUse::Off;
    auto harness = RenderHarness{options};
    auto &pipelines = harness.renderer->pipelines();
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{16, 16}});
    auto const raster = test::viewOf(target, overheadCamera(), core::ViewId{0});
    auto const traced = pathTraced(harness, target, overheadCamera(), core::ViewId{1});

    auto const result = harness.render(render::FrameDesc{litPlane(core::DirectionalLight{}), {raster, traced}});

    REQUIRE(result.views.size() == 2);
    auto const shading = std::vector<std::string>{"visibility statistics",
                                                  "clear shadow map",
                                                  "assign lights",
                                                  "clear material model counts",
                                                  "classify pixels",
                                                  "allocate material model lists",
                                                  "scatter material model pixels",
                                                  "shade lr_shade_metallic_roughness",
                                                  "tonemap"};
    auto twoPass = std::vector<std::string>{"clear visibility",         "cull instances",
                                            "compact visible instances", "cull clusters",
                                            "compact visible clusters",  "rasterize depth",
                                            "build depth pyramid",       "cull occluded clusters",
                                            "compact late clusters",     "rasterize late depth",
                                            "rasterize triangles",       "rasterize late triangles"};
    twoPass.insert(twoPass.end(), shading.begin(), shading.end());
    CHECK(passNames(result.views[0]) == twoPass);
    CHECK(result.views[0].passTimings[0].dispatches == 2);
    CHECK(passNames(result.views[1]) == std::vector<std::string>{"path trace", "tonemap"});
    CHECK(allTimed(result.views[0]));
    CHECK(allTimed(result.views[1]));

    options.singlePassVisibility = render::VariantUse::Auto;
    auto onePassHarness = RenderHarness{options};
    REQUIRE(onePassHarness.renderer->capabilities().singlePassVisibility);
    auto const onePassTarget =
        onePassHarness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{16, 16}});
    auto onePass = std::vector<std::string>{"cull instances",          "compact visible instances",
                                            "cull clusters",           "compact visible clusters",
                                            "clear packed visibility", "rasterize visibility",
                                            "build depth pyramid",     "cull occluded clusters",
                                            "compact late clusters",   "rasterize late visibility",
                                            "unpack visibility"};
    onePass.insert(onePass.end(), shading.begin(), shading.end());
    auto const onePassResult = onePassHarness.render(
        frameOf(litPlane(core::DirectionalLight{}), test::viewOf(onePassTarget, overheadCamera())));
    CHECK(passNames(onePassResult.views[0]) == onePass);
    CHECK(allTimed(onePassResult.views[0]));

    auto untimed = RenderHarness{};
    auto const untimedTarget = untimed.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{16, 16}});
    CHECK(untimed.render(frameOf(litPlane(core::DirectionalLight{}), test::viewOf(untimedTarget, overheadCamera())))
              .views[0]
              .passTimings.empty());

    // Dispatches the frame graph does not know of leave the timings unattributable.
    auto filled = raster;
    filled.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {std::make_shared<UncountedFill>()}});
    CHECK(harness.render(frameOf({}, filled)).views[0].passTimings.empty());
}

core::RenderWorldDelta smallScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makePlane(core::vec2{3.0f, 3.0f}))},
                    {core::MeshId{1}, test::share(core::makeSphere(0.6f, 24, 12))},
                    {core::MeshId{2}, test::share(core::makeBox(core::vec3{0.4f}))}};
    auto metal = grey(0.9f);
    metal.metallic = 1.0f;
    metal.roughness = 0.4f;
    auto red = grey(0.6f);
    red.baseColor = core::vec4{0.8f, 0.2f, 0.1f, 1.0f};
    red.roughness = 0.7f;
    delta.materials = {{core::MaterialId{0}, grey(0.6f)}, {core::MaterialId{1}, metal}, {core::MaterialId{2}, red},
                       {core::MaterialId{3}, test::emissive(core::vec3{1.5f, 1.0f, 0.5f})}};
    auto turned = core::Transform{};
    turned.translation = core::vec3{-0.8f, 0.4f, 0.3f};
    turned.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, 0.5f);
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}},
                       {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1},
                        test::translation(core::vec3{0.7f, 0.6f, -0.2f})},
                       {core::InstanceId{2}, core::MeshId{2}, core::MaterialId{2}, core::matrixOf(turned)},
                       {core::InstanceId{3}, core::MeshId{2}, core::MaterialId{3},
                        test::translation(core::vec3{0.2f, 0.25f, 1.2f}) * core::matrixOf(core::Transform{
                                                                              core::vec3{0.0f}, {}, core::vec3{0.4f}})}};
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.illuminance = 2.0f;
    auto lamp = core::PointLight{};
    lamp.position = core::vec3{0.5f, 1.5f, 1.5f};
    lamp.intensity = 3.0f;
    lamp.range = 6.0f;
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

// The GPU backends on test::deviceFor() their names against the CPU backend. Skipped where
// those devices are missing, unless LRENDER_TEST_REQUIRE_GPU is set.
constexpr auto kEquivalenceSamples = 4u;
constexpr auto kEquivalencePixels = 48;

TEST_CASE("every GPU backend path traces what the CPU backend path traces") {
    auto const camera = test::cameraAt(core::vec3{0.0f, 2.0f, 4.0f}, core::vec3{0.0f, 0.4f, 0.0f});
    auto renderWith = [&camera](RenderHarness &harness) {
        auto const extent = core::ivec2{kEquivalencePixels, kEquivalencePixels};
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{extent});
        auto const view = pathTraced(harness, target, camera);
        auto result = harness.render(frameOf(smallScene(), view));
        for (auto frame = 1u; frame < kEquivalenceSamples; ++frame) {
            result = harness.render(frameOf({}, view));
        }
        return result.views.at(0);
    };
    auto cpu = RenderHarness{};
    auto const reference = renderWith(cpu);
    REQUIRE(reference.stats.accumulatedSamples == kEquivalenceSamples);

    auto const requireGpu = std::getenv("LRENDER_TEST_REQUIRE_GPU") != nullptr;
    for (auto const &backend : tc::availableBackends()) {
        if (backend == "cpu") {
            continue;
        }
        CAPTURE(backend);
        auto harness = std::unique_ptr<RenderHarness>{};
        try {
            auto options = test::HarnessOptions{};
            options.backend = backend;
            options.device = test::deviceFor(backend);
            options.gpuTimers = true;
            harness = std::make_unique<RenderHarness>(options);
        } catch (tc::DeviceError const &error) {
            if (requireGpu) {
                FAIL(error.what());
            }
            MESSAGE("skipped: " << error.what());
            continue;
        }
        auto const result = renderWith(*harness);
        auto const difference = compare(*result.image, *reference.image);
        CAPTURE(difference.largest);
        CAPTURE(difference.meanDifference);
        CAPTURE(difference.farFraction);
        CHECK(result.stats.accumulatedSamples == kEquivalenceSamples);
        // Every device draws the same random numbers, so the paths are the same ones and
        // their colours differ by rounding: at most a level, a tenth of one on average
        // (measured 0.02 on OpenGL and Vulkan, 0.09 on OpenGL ES). Rounding can still send
        // a ray past the other side of an edge, which changes its pixel by a whole sample;
        // none does here, and a few may on other devices.
        CHECK(difference.meanDifference < 0.25);
        CHECK(difference.farFraction < 0.005);
        if (harness->renderer->deviceInfo().timers) {
            CHECK(passNames(result) == std::vector<std::string>{"path trace", "tonemap"});
        }
    }
}

core::mat4 placed(core::vec3 const &at, float const turn, core::vec3 const &scale = core::vec3{1.0f}) {
    return core::matrixOf(
        core::Transform{at, core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, turn), scale});
}

// smallScene() behind the cases traversal treats apart: first an instance of a mesh
// without triangles, then quads that are masked (one kept, one cut out), blended, mirrored
// (single-sided, facing the camera and away), and double-sided seen from behind.
core::RenderWorldDelta traversalScene() {
    auto delta = smallScene();
    delta.meshes.push_back({core::MeshId{10}, test::share(core::MeshData{})});
    delta.meshes.push_back({core::MeshId{11}, test::share(core::makeQuad(core::vec2{0.3f}))});
    auto const partial = [](core::AlphaMode const mode, float const alpha) {
        auto material = grey(0.7f);
        material.baseColor = core::vec4{0.2f, 0.6f, 0.9f, alpha};
        material.alphaMode = mode;
        return material;
    };
    delta.materials.push_back({core::MaterialId{4}, partial(core::AlphaMode::Mask, 0.7f)});
    delta.materials.push_back({core::MaterialId{5}, partial(core::AlphaMode::Mask, 0.3f)});
    delta.materials.push_back({core::MaterialId{6}, partial(core::AlphaMode::Blend, 0.5f)});
    delta.materials.push_back({core::MaterialId{7}, grey(0.5f, true)});
    auto const pi = 3.14159265f;
    auto const mirrored = core::vec3{-1.0f, 1.0f, 1.0f};
    delta.instances.insert(delta.instances.begin(),
                           {core::InstanceId{10}, core::MeshId{10}, core::MaterialId{0}, core::mat4{1.0f}});
    delta.instances.push_back({core::InstanceId{11}, core::MeshId{11}, core::MaterialId{4}, placed({-0.9f, 0.9f, 1.6f}, 0.0f)});
    delta.instances.push_back({core::InstanceId{12}, core::MeshId{11}, core::MaterialId{5}, placed({-0.3f, 0.9f, 1.6f}, 0.0f)});
    delta.instances.push_back({core::InstanceId{13}, core::MeshId{11}, core::MaterialId{6}, placed({0.3f, 0.9f, 1.6f}, 0.0f)});
    delta.instances.push_back(
        {core::InstanceId{14}, core::MeshId{11}, core::MaterialId{0}, placed({0.9f, 0.9f, 1.6f}, 0.0f, mirrored)});
    delta.instances.push_back(
        {core::InstanceId{15}, core::MeshId{11}, core::MaterialId{0}, placed({-0.9f, 1.5f, 1.0f}, pi, mirrored)});
    delta.instances.push_back({core::InstanceId{16}, core::MeshId{11}, core::MaterialId{7}, placed({0.0f, 1.5f, 1.0f}, pi)});
    return delta;
}

// Then the sphere moves, the lit box and the empty mesh go away, the quads grow and a
// second sphere comes in.
core::RenderWorldDelta traversalChanges() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{11}, test::share(core::makeQuad(core::vec2{0.4f}))}};
    delta.instances = {{core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, test::translation(core::vec3{0.4f, 0.7f, 0.1f})},
                       {core::InstanceId{17}, core::MeshId{1}, core::MaterialId{2}, placed({-0.6f, 0.4f, -0.8f}, 0.0f, core::vec3{0.5f})}};
    delta.removed.instances = {core::InstanceId{3}, core::InstanceId{10}};
    delta.removed.meshes = {core::MeshId{10}};
    return delta;
}

// On each device with ray queries, against lrender's own hierarchies on the same device.
TEST_CASE("the path tracer traces the same paths with ray queries as through lrender's hierarchies") {
    auto const camera = test::cameraAt(core::vec3{0.0f, 2.0f, 4.0f}, core::vec3{0.0f, 0.6f, 0.0f});
    auto const extent = core::ivec2{kEquivalencePixels, kEquivalencePixels};
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        if (not harness.renderer->capabilities().rayQueries) {
            MESSAGE("no ray queries on " << harness.renderer->deviceInfo().deviceName);
            return;
        }
        auto options = test::HarnessOptions{};
        options.backend = harness.renderer->deviceInfo().backend;
        options.device = test::deviceFor(options.backend);
        options.rayQueries = render::VariantUse::Off;
        auto hierarchies = RenderHarness{options};
        REQUIRE_FALSE(hierarchies.renderer->capabilities().rayQueries);
        auto const traced = pathTraced(harness, harness.renderer->targets().createImage(render::ImageTargetDesc{extent}), camera);
        auto const walked =
            pathTraced(hierarchies, hierarchies.renderer->targets().createImage(render::ImageTargetDesc{extent}), camera);
        for (auto const &delta : {traversalScene(), traversalChanges()}) {
            auto rayQueried = harness.render(frameOf(delta, traced));
            auto walkedResult = hierarchies.render(frameOf(delta, walked));
            for (auto frame = 1u; frame < kEquivalenceSamples; ++frame) {
                rayQueried = harness.render(frameOf({}, traced));
                walkedResult = hierarchies.render(frameOf({}, walked));
            }
            CHECK(rayQueried.views.at(0).stats.accumulatedSamples == kEquivalenceSamples);
            auto const difference = compare(*rayQueried.views[0].image, *walkedResult.views.at(0).image);
            CAPTURE(difference.largest);
            CAPTURE(difference.meanDifference);
            CAPTURE(difference.farFraction);
            MESSAGE(options.backend << ": mean difference " << difference.meanDifference << " levels, "
                                    << difference.farFraction * 100.0 << "% of pixels more than 2 apart");
            // The same random numbers decide the same paths: only where the two intersect a
            // ray with a triangle differently by rounding does a path go elsewhere.
            CHECK(difference.meanDifference < 0.25);
            CHECK(difference.farFraction < 0.005);
        }
    });
}

} // namespace
