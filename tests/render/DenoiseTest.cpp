#include "RenderTestSupport.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;
using test::frameOf;

constexpr auto kPixels = 48;

core::MaterialDesc grey(float const baseColor) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{baseColor, baseColor, baseColor, 1.0f};
    material.roughness = 1.0f;
    return material;
}

// A floor with a metal sphere, a red box and a glowing one, under a sun, a lamp and a sky.
core::RenderWorldDelta denoiseScene() {
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
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}},
                       {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1},
                        test::translation(core::vec3{0.7f, 0.6f, -0.2f})},
                       {core::InstanceId{2}, core::MeshId{2}, core::MaterialId{2},
                        test::translation(core::vec3{-0.8f, 0.4f, 0.3f})},
                       {core::InstanceId{3}, core::MeshId{2}, core::MaterialId{3},
                        test::translation(core::vec3{0.2f, 0.25f, 1.2f})}};
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

core::CameraDesc cameraOf(float const along) {
    return test::cameraAt(core::vec3{along, 2.0f, 4.0f}, core::vec3{along * 0.5f, 0.4f, 0.0f});
}

render::ViewDesc viewOf(RenderHarness &harness, render::PipelineId const pipeline, core::CameraDesc const &camera,
                        core::ViewId const id = core::ViewId{0}) {
    auto view = test::viewOf(
        harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kPixels, kPixels}}), camera, id);
    view.pipeline = pipeline;
    return view;
}

// The root mean square of the channel differences, in 8-bit levels, over the pixels
// `counted` keeps.
template <class Counted>
double rmsDifference(render::ImageRgba8 const &a, render::ImageRgba8 const &b, Counted const &counted) {
    auto squares = 0.0;
    auto count = 0;
    for (auto index = std::size_t{0}; index < a.pixels.size(); ++index) {
        if (not counted(index)) {
            continue;
        }
        for (auto c = 0u; c < 3u; ++c) {
            auto const difference = static_cast<double>(test::channelDifference(a.pixels[index], b.pixels[index], c));
            squares += difference * difference;
        }
        ++count;
    }
    return std::sqrt(squares / (3.0 * count));
}

double rmsDifference(render::ImageRgba8 const &a, render::ImageRgba8 const &b) {
    return rmsDifference(a, b, [](std::size_t) { return true; });
}

// Whether no channel of the pixel's 3x3 neighbours is more than `levels` from its own.
bool smoothAt(render::ImageRgba8 const &image, std::size_t const index, std::uint32_t const levels) {
    auto const x = static_cast<int>(index % kPixels);
    auto const y = static_cast<int>(index / kPixels);
    for (auto j = std::max(y - 1, 0); j <= std::min(y + 1, kPixels - 1); ++j) {
        for (auto i = std::max(x - 1, 0); i <= std::min(x + 1, kPixels - 1); ++i) {
            if (not test::nearPixel(test::pixelAt(image, i, j), image.pixels[index], levels)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::string> passNames(render::ViewResult const &view) {
    auto names = std::vector<std::string>{};
    for (auto const &pass : view.passTimings) {
        names.push_back(pass.pass);
    }
    return names;
}

// The reference's mean of `samples` samples at `camera`.
render::ImageRgba8 converged(core::CameraDesc const &camera, std::uint32_t const samples) {
    auto harness = RenderHarness{};
    auto const view = viewOf(harness, harness.renderer->pipelines().pathTraceReference(), camera);
    auto result = harness.render(frameOf(denoiseScene(), view));
    for (auto frame = 1u; frame < samples; ++frame) {
        result = harness.render(frameOf({}, view));
    }
    return std::move(*result.views.at(0).image);
}

TEST_CASE("the denoised path tracer shows the reference's image once its mean holds 64 samples") {
    auto options = test::HarnessOptions{};
    options.gpuTimers = true;
    auto harness = RenderHarness{options};
    auto &pipelines = harness.renderer->pipelines();
    auto const reference = viewOf(harness, pipelines.pathTraceReference(), cameraOf(0.0f), core::ViewId{0});
    auto const interactive = viewOf(harness, pipelines.pathTraceInteractive(), cameraOf(0.0f), core::ViewId{1});

    auto result = harness.render(render::FrameDesc{denoiseScene(), {reference, interactive}});
    auto const passes = passNames(result.views.at(1));
    for (auto const *const pass :
         {"path trace", "temporal motion", "denoise guides", "demodulate illumination", "accumulate illumination",
          "estimate variance", "filter illumination", "remodulate illumination", "tonemap"}) {
        CAPTURE(pass);
        CHECK(std::count(passes.begin(), passes.end(), std::string{pass}) == 1);
    }
    for (auto frame = 1; frame < 64; ++frame) {
        if (frame == 8) {
            CHECK(result.views[1].image->pixels != result.views[0].image->pixels);
        }
        result = harness.render(render::FrameDesc{{}, {reference, interactive}});
    }
    CHECK(result.views[0].stats.accumulatedSamples == 64);
    CHECK(result.views[1].stats.accumulatedSamples == 64);
    CHECK(result.views[1].image->pixels == result.views[0].image->pixels);
}

// Each frame the camera moves, so the mean starts over with one sample; the denoiser
// keeps what the surfaces showed before and filters what it has. Its guides are of pixel
// centres, so edges, which the path tracer's samples spread over each pixel smooth, it
// shows as steps: the comparison leaves them out.
TEST_CASE("while the camera moves, the denoised path tracer is nearer the converged image than one sample") {
    constexpr auto kFrames = 12;
    constexpr auto kStep = 0.04f;
    auto harness = RenderHarness{};
    auto &pipelines = harness.renderer->pipelines();
    auto reference = viewOf(harness, pipelines.pathTraceReference(), cameraOf(0.0f), core::ViewId{0});
    auto interactive = viewOf(harness, pipelines.pathTraceInteractive(), cameraOf(0.0f), core::ViewId{1});
    auto result = harness.render(render::FrameDesc{denoiseScene(), {reference, interactive}});
    for (auto frame = 1; frame < kFrames; ++frame) {
        reference.camera = cameraOf(kStep * static_cast<float>(frame));
        interactive.camera = reference.camera;
        result = harness.render(render::FrameDesc{{}, {reference, interactive}});
    }
    REQUIRE(result.views[0].stats.accumulatedSamples == 1);
    REQUIRE(result.views[1].stats.accumulatedSamples == 1);

    auto const expected = converged(reference.camera, 256);
    auto const smooth = [&expected](std::size_t const index) { return smoothAt(expected, index, 12); };
    auto const edges = [&smooth](std::size_t const index) { return not smooth(index); };
    auto const noisy = rmsDifference(*result.views[0].image, expected, smooth);
    auto const denoised = rmsDifference(*result.views[1].image, expected, smooth);
    auto const noisyEdges = rmsDifference(*result.views[0].image, expected, edges);
    auto const denoisedEdges = rmsDifference(*result.views[1].image, expected, edges);
    MESSAGE("against 256 samples, away from edges: one sample " << noisy << " levels rms, denoised " << denoised
                                                                << "; at edges " << noisyEdges << " and "
                                                                << denoisedEdges);
    CHECK(denoised * 2.5 < noisy);
}

// A camera cut drops what the denoiser kept: the first frame after it is the frame a
// view without a past renders.
TEST_CASE("after a camera cut the denoised path tracer renders what a new view renders") {
    auto harness = RenderHarness{};
    auto view = viewOf(harness, harness.renderer->pipelines().pathTraceInteractive(), cameraOf(-0.5f));
    harness.render(frameOf(denoiseScene(), view));
    for (auto frame = 1; frame < 10; ++frame) {
        harness.render(frameOf({}, view));
    }
    view.camera = cameraOf(0.3f);
    view.cameraCut = true;
    auto const cut = harness.render(frameOf({}, view));

    auto fresh = RenderHarness{};
    auto const freshView = viewOf(fresh, fresh.renderer->pipelines().pathTraceInteractive(), cameraOf(0.3f));
    auto const expected = fresh.render(frameOf(denoiseScene(), freshView));
    CHECK(cut.views.at(0).image->pixels == expected.views.at(0).image->pixels);
}

// The box, grey now, slides across the floor in front of a still camera. Where it was
// three frames before the last the floor shows again, and what the denoiser kept there
// is of the box, at another distance and with another normal: it starts over there
// instead of keeping the box's light. Its brightness there is then near the converged
// image's, nearer than a new view's (which filters the floor's shadow away with its
// light) and than one that kept the box's (measured 3, 18 and 32 levels off).
TEST_CASE("the denoised path tracer leaves no trail behind a moving object") {
    constexpr auto kFrames = 16;
    auto const camera = test::cameraAt(core::vec3{0.0f, 3.0f, 3.0f}, core::vec3{0.0f, 0.0f, 0.0f});
    auto const boxAt = [](int const frame, core::MaterialId const material) {
        return core::InstanceUpsert{core::InstanceId{3}, core::MeshId{2}, material,
                                    test::translation(core::vec3{-1.2f + 0.1f * static_cast<float>(frame), 0.4f, 0.8f})};
    };
    auto const sceneWithBoxAt = [&boxAt](int const frame, core::MaterialId const material) {
        auto scene = denoiseScene();
        scene.materials.push_back({core::MaterialId{4}, test::emissive(core::vec3{0.0f, 20.0f, 0.0f})});
        scene.instances[3] = boxAt(frame, material);
        return scene;
    };
    // The pixels the box covers at `frame`: those of one sample of the reference where it
    // glows green.
    auto const coverage = [&](int const frame) {
        auto harness = RenderHarness{};
        auto const view = viewOf(harness, harness.renderer->pipelines().pathTraceReference(), camera);
        auto const result = harness.render(frameOf(sceneWithBoxAt(frame, core::MaterialId{4}), view));
        auto covered = std::vector<bool>{};
        for (auto const pixel : result.views.at(0).image->pixels) {
            covered.push_back(test::channel(pixel, 1) > 200 and test::channel(pixel, 0) < 100);
        }
        return covered;
    };

    auto harness = RenderHarness{};
    auto const view = viewOf(harness, harness.renderer->pipelines().pathTraceInteractive(), camera);
    auto result = harness.render(frameOf(sceneWithBoxAt(0, core::MaterialId{0}), view));
    for (auto frame = 1; frame < kFrames; ++frame) {
        auto delta = core::RenderWorldDelta{};
        delta.instances = {boxAt(frame, core::MaterialId{0})};
        result = harness.render(frameOf(std::move(delta), view));
    }
    auto truth = RenderHarness{};
    auto const truthView = viewOf(truth, truth.renderer->pipelines().pathTraceReference(), camera);
    auto expected = truth.render(frameOf(sceneWithBoxAt(kFrames - 1, core::MaterialId{0}), truthView));
    for (auto sample = 1; sample < 256; ++sample) {
        expected = truth.render(frameOf({}, truthView));
    }

    auto const was = coverage(kFrames - 4);
    auto const is = coverage(kFrames - 1);
    auto const brightness = [](std::uint32_t const pixel) {
        return static_cast<double>(test::channel(pixel, 0) + test::channel(pixel, 1) + test::channel(pixel, 2)) / 3.0;
    };
    auto uncovered = 0;
    auto slid = 0.0;
    auto converged = 0.0;
    for (auto index = std::size_t{0}; index < was.size(); ++index) {
        if (was[index] and not is[index]) {
            ++uncovered;
            slid += brightness(result.views.at(0).image->pixels[index]);
            converged += brightness(expected.views.at(0).image->pixels[index]);
        }
    }
    REQUIRE(uncovered > 50);
    slid /= uncovered;
    converged /= uncovered;
    MESSAGE(uncovered << " pixels uncovered: " << slid << " levels after the slide, " << converged << " converged");
    CHECK(std::abs(slid - converged) < 8.0);
}

TEST_CASE("every GPU backend denoises what the CPU backend denoises") {
    constexpr auto kFrames = 4;
    auto const renderWith = [](RenderHarness &harness) {
        auto view = viewOf(harness, harness.renderer->pipelines().pathTraceInteractive(), cameraOf(0.0f));
        auto result = harness.render(frameOf(denoiseScene(), view));
        for (auto frame = 1; frame < kFrames; ++frame) {
            view.camera = cameraOf(0.04f * static_cast<float>(frame));
            result = harness.render(frameOf({}, view));
        }
        return std::move(*result.views.at(0).image);
    };
    auto cpu = RenderHarness{};
    auto const reference = renderWith(cpu);
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        if (harness.renderer->deviceInfo().backend == "cpu") {
            return;
        }
        auto const image = renderWith(harness);
        auto const difference = rmsDifference(image, reference);
        MESSAGE(harness.renderer->deviceInfo().backend << ": " << difference << " levels rms from the CPU backend, "
                                                       << test::maxChannelDifference(image, reference) << " at most");
        // Paths that rounding sends elsewhere spread through the filter: measured 0.4 to 0.8
        // levels on Mesa's devices and NVIDIA's.
        CHECK(difference < 1.5);
    });
}

} // namespace
