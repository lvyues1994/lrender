#include "RenderTestSupport.h"
#include "Temporal.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

double radicalInverse(std::uint32_t index, std::uint32_t const base) {
    auto result = 0.0;
    auto scale = 1.0 / base;
    while (index > 0) {
        result += (index % base) * scale;
        index /= base;
        scale /= base;
    }
    return result;
}

// Pixels across and rows down where a view-projection puts a point on a square target.
core::vec2 screenOf(core::mat4 const &viewProjection, core::vec3 const &point, float const size) {
    auto const clip = viewProjection * core::vec4{point, 1.0f};
    return core::vec2{(clip.x / clip.w * 0.5f + 0.5f) * size, (0.5f - clip.y / clip.w * 0.5f) * size};
}

TEST_CASE("the jitter runs through Halton (2, 3) points within half a pixel") {
    for (auto sample = 0u; sample < 2 * render::kTemporalJitters; ++sample) {
        auto const jitter = render::temporalJitter(sample);
        auto const index = 1 + sample % render::kTemporalJitters;
        CHECK(jitter.x == doctest::Approx(radicalInverse(index, 2) - 0.5));
        CHECK(jitter.y == doctest::Approx(radicalInverse(index, 3) - 0.5));
        CHECK(std::abs(jitter.x) < 0.5f);
        CHECK(std::abs(jitter.y) < 0.5f);
    }

    auto const camera = test::cameraAt(core::vec3{1.0f, 2.0f, 5.0f}, core::vec3{0.0f});
    auto const viewProjection = core::projectionMatrix(camera, 1.0f) * core::viewMatrix(camera);
    auto const jitter = render::temporalJitter(3);
    auto const jittered = render::jitteredViewProjection(viewProjection, jitter, core::ivec2{64, 64});
    auto const point = core::vec3{0.3f, -0.2f, 0.4f};
    auto const moved = screenOf(jittered, point, 64.0f) - screenOf(viewProjection, point, 64.0f);
    CHECK(moved.x == doctest::Approx(jitter.x).epsilon(1e-3));
    CHECK(moved.y == doctest::Approx(jitter.y).epsilon(1e-3));
}

TEST_CASE("a view's history goes on over consecutive frames at one extent, and starts over otherwise") {
    auto views = render::TemporalViews{60};
    auto const extent = core::ivec2{32, 32};
    auto const first = core::mat4{1.0f};
    auto const second = core::mat4{2.0f};
    auto const at = [&](std::uint64_t const frame, core::mat4 const &viewProjection, core::ivec2 const size,
                        bool const cut) {
        return views.advance(render::TemporalRequest{core::ViewId{0}, frame, size, viewProjection, cut});
    };

    auto const fresh = at(5, first, extent, false);
    CHECK_FALSE(fresh.continues);
    CHECK(fresh.sample == 0);
    CHECK(fresh.previousViewProjection[0][0] == 1.0f);

    auto const next = at(6, second, extent, false);
    CHECK(next.continues);
    CHECK(next.sample == 1);
    CHECK(next.previousViewProjection[0][0] == 1.0f);
    CHECK(next.viewProjection[0][0] == 2.0f);
    auto const secondJitter = next.jitter == render::temporalJitter(1);
    CHECK(secondJitter);

    CHECK_FALSE(at(8, second, extent, false).continues); // a frame without the view
    CHECK_FALSE(at(9, second, core::ivec2{16, 32}, false).continues);
    CHECK_FALSE(at(10, second, core::ivec2{16, 32}, true).continues);
    CHECK(at(11, second, core::ivec2{16, 32}, false).continues);
    views.restart();
    CHECK_FALSE(at(12, second, core::ivec2{16, 32}, false).continues);
}

// A view of the visibility pipeline with temporal anti-aliasing, whose HDR colour and
// motion each frame reads back.
struct TaaView {
    TaaView(RenderHarness &harness_, int const size, core::CameraDesc const &camera, core::ViewId const id)
        : harness(&harness_) {
        auto &pipelines = harness->renderer->pipelines();
        view = test::viewOf(harness->renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{size, size}}),
                            camera, id);
        view.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {colour, motion}});
        view.settings.antiAliasing = render::AntiAliasing::Taa;
    }

    std::vector<core::vec4> render(core::RenderWorldDelta delta = {}) {
        last = harness->render(test::frameOf(std::move(delta), view));
        return *colour->pixels;
    }

    RenderHarness *harness;
    render::ViewDesc view;
    std::shared_ptr<test::HdrProbe> colour = std::make_shared<test::HdrProbe>();
    std::shared_ptr<test::MotionProbe> motion = std::make_shared<test::MotionProbe>();
    render::FrameResult last;
};

core::MaterialDesc glowing(float const radiance) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    material.emissive = core::vec3{radiance};
    return material;
}

core::mat4 placed(core::vec3 const &translation, float const turn = 0.0f) {
    auto transform = core::Transform{};
    transform.translation = translation;
    transform.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 0.0f, 1.0f}, turn);
    return core::matrixOf(transform);
}

// A view of the plane z = 0 from (0, 0, 4): a point (x, y) is at (half + scale x,
// half - scale y) on a square target of side 2 half.
struct FlatView {
    double half;
    double scale;
};

FlatView flatViewOf(int const size) {
    auto const focal = 1.0 / std::tan(0.4); // CameraDesc::verticalFov is 0.8
    return FlatView{size / 2.0, size / 2.0 * focal / 4.0};
}

core::CameraDesc frontCamera() { return test::cameraAt(core::vec3{0.0f, 0.0f, 4.0f}, core::vec3{0.0f}); }

// The edge: a white quad, far larger than the view, turned by kEdgeAngle, whose edge runs
// through kEdgePoint with the quad on the side its normal points to; no sky.
constexpr auto kEdgeAngle = 0.3;
constexpr auto kEdgePointX = 0.013;
constexpr auto kEdgePointY = 0.021;
constexpr auto kEdgeSize = 32;

core::RenderWorldDelta edgeScene() {
    constexpr auto half = 20.0f;
    auto const normal = core::vec2{static_cast<float>(std::cos(kEdgeAngle)), static_cast<float>(std::sin(kEdgeAngle))};
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{half}))}};
    delta.materials = {{core::MaterialId{0}, glowing(1.0f)}};
    auto const centre = core::vec3{static_cast<float>(kEdgePointX) + normal.x * half,
                                   static_cast<float>(kEdgePointY) + normal.y * half, 0.0f};
    delta.instances = {
        {core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, placed(centre, static_cast<float>(kEdgeAngle))}};
    return delta;
}

// How far a point of the target is on the quad's side of the edge, in world units.
double edgeSide(FlatView const &flat, double const u, double const v) {
    auto const x = (u - flat.half) / flat.scale - kEdgePointX;
    auto const y = -(v - flat.half) / flat.scale - kEdgePointY;
    return x * std::cos(kEdgeAngle) + y * std::sin(kEdgeAngle);
}

// The area of the pixel's square on the quad's side.
double coveredArea(FlatView const &flat, int const i, int const j) {
    auto const corners = std::array<std::array<double, 2>, 4>{
        {{{double(i), double(j)}}, {{i + 1.0, double(j)}}, {{i + 1.0, j + 1.0}}, {{double(i), j + 1.0}}}};
    auto clipped = std::vector<std::array<double, 2>>{};
    for (auto k = std::size_t{0}; k < 4; ++k) {
        auto const &a = corners[k];
        auto const &b = corners[(k + 1) % 4];
        auto const sa = edgeSide(flat, a[0], a[1]);
        auto const sb = edgeSide(flat, b[0], b[1]);
        if (sa >= 0.0) {
            clipped.push_back(a);
        }
        if ((sa >= 0.0) != (sb >= 0.0)) {
            auto const t = sa / (sa - sb);
            clipped.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])});
        }
    }
    auto twice = 0.0;
    for (auto k = std::size_t{0}; k < clipped.size(); ++k) {
        auto const &a = clipped[k];
        auto const &b = clipped[(k + 1) % clipped.size()];
        twice += a[0] * b[1] - a[1] * b[0];
    }
    return std::abs(twice) / 2.0;
}

// What the history holds after `frames` frames of a pixel whose frame n covers it or not:
// the mean of the first temporalFrames() frames, then an exponential average with weight
// 1 / temporalFrames() for the newest. Also the least distance, in pixels, of any of its
// samples from the edge.
struct Accumulated {
    double value;
    double closest;
};

Accumulated accumulate(FlatView const &flat, int const i, int const j, std::uint32_t const frames) {
    auto history = 0.0;
    auto closest = 1e9;
    for (auto n = 0u; n < frames; ++n) {
        auto const jitter = render::temporalJitter(n);
        auto const side = edgeSide(flat, i + 0.5 - jitter.x, j + 0.5 - jitter.y);
        closest = std::min(closest, std::abs(side) * flat.scale);
        auto const sample = side > 0.0 ? 1.0 : 0.0;
        history += (sample - history) / std::min(n + 1.0, 10.0);
    }
    return Accumulated{history, closest};
}

// The history after N frames is the average above, kept in half floats: each frame rounds
// what it stores by at most 2^-12 for values below 1, and the blend carries a rounding on
// with weight at most 0.9, so they add up to less than 2^-12 / 0.1.
constexpr auto kHalfFloatDrift = 2.5e-3;

TEST_CASE("a still straight edge converges to the area of each pixel on the quad's side") {
    constexpr auto kFrames = 32u;
    auto harness = RenderHarness{};
    auto taa = TaaView{harness, kEdgeSize, frontCamera(), core::ViewId{0}};
    auto pixels = taa.render(edgeScene());
    auto const single = pixels;
    for (auto frame = 1u; frame < kFrames; ++frame) {
        pixels = taa.render();
    }

    auto const flat = flatViewOf(kEdgeSize);
    auto worstFromAverage = 0.0;
    auto worstAverage = 0.0; // the average's own distance from the area
    auto edgeError = 0.0;
    auto singleError = 0.0;
    auto edgePixels = 0;
    auto ambiguous = 0;
    for (auto j = 0; j < kEdgeSize; ++j) {
        for (auto i = 0; i < kEdgeSize; ++i) {
            auto const area = coveredArea(flat, i, j);
            auto const expected = accumulate(flat, i, j, kFrames);
            auto const shown = static_cast<double>(pixels[static_cast<std::size_t>(j * kEdgeSize + i)].x);
            if (expected.closest < 1e-3) {
                ++ambiguous; // rounding may put a sample on either side
                continue;
            }
            worstFromAverage = std::max(worstFromAverage, std::abs(shown - expected.value));
            if (area > 0.0 and area < 1.0) {
                worstAverage = std::max(worstAverage, std::abs(expected.value - area));
                edgeError += std::abs(shown - area);
                singleError += std::abs(static_cast<double>(single[static_cast<std::size_t>(j * kEdgeSize + i)].x) - area);
                ++edgePixels;
            } else {
                CHECK(shown == area);
            }
        }
    }
    edgeError /= edgePixels;
    singleError /= edgePixels;
    MESSAGE("after " << kFrames << " frames, " << edgePixels << " edge pixels: mean |value - area| " << edgeError
                     << " (one frame: " << singleError << "), the jitter sequence's own largest " << worstAverage
                     << ", largest distance from the simulated history " << worstFromAverage << ", " << ambiguous
                     << " pixels with a sample on the edge");
    CHECK(ambiguous <= 2);
    CHECK(worstFromAverage < kHalfFloatDrift);
    // 16 Halton points cannot place an edge exactly; the average they make stays within
    // this of the area everywhere.
    CHECK(worstAverage < 0.16);
    CHECK(edgeError < 0.04);
    CHECK(edgeError * 4.0 < singleError);
}

// A white quad over a grey backdrop: still for a while, then moving right kStep pixels a
// frame, then still again.
constexpr auto kGhostSize = 64;
constexpr auto kMoverHalf = 0.25f;
constexpr auto kBackdrop = 0.2f;
constexpr auto kStep = 2.0;

core::RenderWorldDelta moverScene(float const x) {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{kMoverHalf}))},
                    {core::MeshId{1}, test::share(core::makeQuad(core::vec2{5.0f}))}};
    delta.materials = {{core::MaterialId{0}, glowing(1.0f)}, {core::MaterialId{1}, glowing(kBackdrop)}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, placed(core::vec3{x, 0.0f, 0.0f})},
                       {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, placed(core::vec3{0.0f, 0.0f, -0.5f})}};
    return delta;
}

core::RenderWorldDelta moveTo(float const x) {
    auto delta = core::RenderWorldDelta{};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, placed(core::vec3{x, 0.0f, 0.0f})}};
    return delta;
}

// How far outside (positive) or inside (negative) the mover at x a pixel centre is, in
// pixels: the larger of its distances along the axes.
double outsideMover(FlatView const &flat, double const x, int const i, int const j) {
    auto const centreU = flat.half + flat.scale * x;
    auto const reach = flat.scale * kMoverHalf;
    return std::max(std::abs(i + 0.5 - centreU), std::abs(j + 0.5 - flat.half)) - reach;
}

struct GhostMeasure {
    double trail = 0.0;    // largest |value - backdrop| in the trail the mover left
    double interior = 0.0; // largest |value - 1| well inside the mover
};

// The trail: the pixel centres the mover passed over since it was at `start`.
GhostMeasure measure(std::vector<core::vec4> const &pixels, FlatView const &flat, double const x, double const start) {
    auto const reach = flat.scale * kMoverHalf;
    auto const left = flat.half + flat.scale * start - reach;
    auto const right = flat.half + flat.scale * x + reach;
    auto result = GhostMeasure{};
    for (auto j = 0; j < kGhostSize; ++j) {
        for (auto i = 0; i < kGhostSize; ++i) {
            auto const value = static_cast<double>(pixels[static_cast<std::size_t>(j * kGhostSize + i)].x);
            auto const swept = i + 0.5 > left and i + 0.5 < right and std::abs(j + 0.5 - flat.half) < reach;
            // The 3x3 neighbourhood and the jitter reach 1.5 pixels; one more for the edge itself.
            if (swept and outsideMover(flat, x, i, j) > 2.5) {
                result.trail = std::max(result.trail, std::abs(value - kBackdrop));
            }
            if (outsideMover(flat, x, i, j) < -1.5) {
                result.interior = std::max(result.interior, std::abs(value - 1.0));
            }
        }
    }
    return result;
}

TEST_CASE("a moving object leaves no ghost behind") {
    auto harness = RenderHarness{};
    auto taa = TaaView{harness, kGhostSize, frontCamera(), core::ViewId{0}};
    auto const flat = flatViewOf(kGhostSize);
    auto const step = static_cast<float>(kStep / flat.scale);
    auto x = -0.4f;
    taa.render(moverScene(x));
    for (auto frame = 0; frame < 12; ++frame) {
        taa.render();
    }
    auto const start = static_cast<double>(x);
    auto worst = GhostMeasure{};
    for (auto frame = 0; frame < 16; ++frame) {
        x += step;
        auto const pixels = taa.render(moveTo(x));
        auto const motion = *taa.motion->motion;
        auto const centre = static_cast<std::size_t>(kGhostSize / 2 * kGhostSize)
                          + static_cast<std::size_t>(flat.half + flat.scale * x);
        CHECK(motion[centre].x == doctest::Approx(kStep).epsilon(1e-3));
        CHECK(std::abs(motion[centre].y) < 1e-3f);
        auto const ghost = measure(pixels, flat, static_cast<double>(x), start);
        if (frame >= 1) {
            worst.trail = std::max(worst.trail, ghost.trail);
            worst.interior = std::max(worst.interior, ghost.interior);
        }
    }
    auto settled = GhostMeasure{};
    for (auto frame = 0; frame < 3; ++frame) {
        settled = measure(taa.render(), flat, static_cast<double>(x), start);
    }
    MESSAGE("moving " << kStep << " pixels a frame: the trail differs from the backdrop by at most " << worst.trail
                      << ", the mover's inside from white by at most " << worst.interior << "; once still, "
                      << settled.trail << " and " << settled.interior);
    CHECK(worst.trail < 1.0 / 255.0);
    CHECK(worst.interior < 0.01);
    CHECK(settled.trail < 1.0 / 255.0);
}

TEST_CASE("motion follows moving instances and the camera, and is 0 where nothing moves") {
    auto harness = RenderHarness{};
    auto taa = TaaView{harness, kGhostSize, frontCamera(), core::ViewId{0}};
    auto const flat = flatViewOf(kGhostSize);
    taa.render(moverScene(0.0f));
    taa.render();
    auto const still = *taa.motion->motion;
    CHECK(std::all_of(still.begin(), still.end(), [](core::vec2 const &m) { return m.x == 0.0f and m.y == 0.0f; }));

    // The camera moves right by `pan`: the backdrop, 4.5 away, moves left on the screen.
    constexpr auto pan = 0.05f;
    taa.view.camera = test::cameraAt(core::vec3{pan, 0.0f, 4.0f}, core::vec3{pan, 0.0f, 0.0f});
    taa.render();
    auto const panned = *taa.motion->motion;
    auto const backdrop = panned[static_cast<std::size_t>(4 * kGhostSize + 4)];
    CHECK(backdrop.x == doctest::Approx(-flat.scale * 4.0 / 4.5 * pan).epsilon(1e-3));
    CHECK(std::abs(backdrop.y) < 1e-3f);
    auto const mover = panned[static_cast<std::size_t>(kGhostSize / 2 * kGhostSize + kGhostSize / 2)];
    CHECK(mover.x == doctest::Approx(-flat.scale * pan).epsilon(1e-3));
}

// Edges, textures and lights to anti-alias.
core::RenderWorldDelta texturedScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makePlane(core::vec2{4.0f}))},
                    {core::MeshId{1}, test::share(core::makeSphere(0.4f, 24, 12))},
                    {core::MeshId{2}, test::share(core::makeBox(core::vec3{0.3f}))}};
    delta.textures = {{core::TextureId{0}, test::rgba8Image(core::ivec2{64, 64}, core::ColorEncoding::Srgb,
                                                            [](int const x, int const y) {
                                                                auto const light = (x / 8 + y / 8) % 2 == 0;
                                                                return light ? tc::ivec4{220, 210, 190, 255}
                                                                             : tc::ivec4{60, 70, 90, 255};
                                                            }),
                       {}}};
    auto ground = core::MaterialDesc{};
    ground.baseColorTexture = core::TextureId{0};
    ground.uvTransform.scale = core::vec2{3.0f};
    auto metal = core::MaterialDesc{};
    metal.baseColor = core::vec4{0.9f, 0.8f, 0.5f, 1.0f};
    metal.metallic = 1.0f;
    metal.roughness = 0.3f;
    delta.materials = {{core::MaterialId{0}, ground}, {core::MaterialId{1}, metal}, {core::MaterialId{2}, core::MaterialDesc{}}};
    delta.instances = {
        {core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}},
        {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, test::translation(core::vec3{0.4f, 0.4f, 0.2f})},
        {core::InstanceId{2}, core::MeshId{2}, core::MaterialId{2}, placed(core::vec3{-0.6f, 0.3f, -0.2f}, 0.4f)}};
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.illuminance = 2.0f;
    sun.castsShadows = true;
    delta.lights = {{core::LightId{0}, sun}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.1f, 0.15f, 0.25f}};
    return delta;
}

core::CameraDesc sceneCamera() { return test::cameraAt(core::vec3{0.3f, 1.6f, 2.6f}, core::vec3{0.0f, 0.3f, 0.0f}); }

TEST_CASE("a resize or a camera cut starts the history over") {
    auto harness = RenderHarness{};
    auto &targets = harness.renderer->targets();
    auto taa = TaaView{harness, 32, sceneCamera(), core::ViewId{0}};
    taa.render(texturedScene());
    for (auto frame = 0; frame < 5; ++frame) {
        taa.render();
    }

    targets.resize(taa.view.target, core::ivec2{24, 24});
    auto const resized = taa.render();
    auto fresh = TaaView{harness, 24, sceneCamera(), core::ViewId{1}};
    auto const startedOver = resized == fresh.render();
    CHECK(startedOver);

    for (auto frame = 0; frame < 4; ++frame) {
        taa.render();
    }
    auto const elsewhere = test::cameraAt(core::vec3{-1.0f, 1.2f, 2.4f}, core::vec3{0.0f, 0.2f, 0.0f});
    auto elsewhereFresh = TaaView{harness, 24, elsewhere, core::ViewId{2}};
    auto const firstFrame = elsewhereFresh.render();
    taa.view.camera = elsewhere;
    auto continued = TaaView{harness, 24, sceneCamera(), core::ViewId{3}};
    continued.render();
    continued.view.camera = elsewhere;
    auto const blended = continued.render() == firstFrame;
    CHECK_FALSE(blended);
    taa.view.cameraCut = true;
    auto const cut = taa.render() == firstFrame;
    CHECK(cut);
}

std::vector<std::string> passNames(render::ViewResult const &view) {
    auto names = std::vector<std::string>{};
    for (auto const &pass : view.passTimings) {
        names.push_back(pass.pass);
    }
    return names;
}

TEST_CASE("without temporal anti-aliasing nothing temporal runs and still frames stay the same") {
    auto options = test::HarnessOptions{};
    options.gpuTimers = true;
    auto harness = RenderHarness{options};
    auto view = test::viewOf(harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{24, 24}}),
                             sceneCamera());
    harness.render(test::frameOf(texturedScene(), view)); // its textures arrive in the next frame
    auto const first = harness.render(test::frameOf({}, view));
    auto const second = harness.render(test::frameOf({}, view));
    auto const names = passNames(first.views[0]);
    CHECK(std::find(names.begin(), names.end(), "temporal motion") == names.end());
    CHECK(first.views[0].image->pixels == second.views[0].image->pixels);

    view.settings.antiAliasing = render::AntiAliasing::Taa;
    auto const jittered = harness.render(test::frameOf({}, view));
    auto const taaNames = passNames(jittered.views[0]);
    for (auto const *pass : {"temporal motion", "temporal resolve", "show resolved"}) {
        CAPTURE(pass);
        CHECK(std::find(taaNames.begin(), taaNames.end(), pass) != taaNames.end());
    }

    // The path tracer has no use for a jitter.
    view.pipeline = harness.renderer->pipelines().pathTraceReference();
    auto const traced = harness.render(test::frameOf({}, view));
    CHECK(passNames(traced.views[0]) == std::vector<std::string>{"path trace", "tonemap"});
}

TEST_CASE("every GPU backend resolves the same frames as the CPU backend") {
    constexpr auto kFrames = 12;
    auto renderWith = [](RenderHarness &harness) {
        auto taa = TaaView{harness, 48, sceneCamera(), core::ViewId{0}};
        taa.render(texturedScene());
        for (auto frame = 1; frame < kFrames; ++frame) {
            taa.render();
        }
        return std::move(*taa.last.views[0].image);
    };
    auto cpu = RenderHarness{};
    auto const reference = renderWith(cpu);

    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        if (harness.renderer->deviceInfo().backend == "cpu") {
            return;
        }
        auto const image = renderWith(harness);
        auto total = 0.0;
        auto far = 0;
        for (auto index = std::size_t{0}; index < image.pixels.size(); ++index) {
            auto worst = 0u;
            for (auto c = 0u; c < 3u; ++c) {
                auto const difference = test::channelDifference(image.pixels[index], reference.pixels[index], c);
                worst = std::max(worst, difference);
                total += difference;
            }
            far += worst > 2 ? 1 : 0;
        }
        auto const mean = total / (3.0 * static_cast<double>(image.pixels.size()));
        auto const farFraction = static_cast<double>(far) / static_cast<double>(image.pixels.size());
        MESSAGE(harness.renderer->deviceInfo().backend << ": mean difference " << mean << " levels, "
                                                       << farFraction * 100.0 << "% of pixels more than 2 apart");
        // As for single frames (VisibilityPipelineTest): rounding moves a few pixels across
        // edges and shadow boundaries, and the history carries them for some frames.
        CHECK(mean < 0.25);
        CHECK(farFraction < 0.01);
    });
}

} // namespace
