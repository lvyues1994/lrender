#include "RenderTestSupport.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kSize = 64;
core::vec3 const kRed{0.6f, 0.1f, 0.05f};

// Black, so that without lights or sky it shows its emission alone.
core::MaterialDesc glowing(core::vec3 const &emission, float const alpha, core::AlphaMode const mode) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{0.0f, 0.0f, 0.0f, alpha};
    material.emissive = emission;
    material.alphaMode = mode;
    return material;
}

core::mat4 at(core::vec3 const &position) {
    auto transform = core::Transform{};
    transform.translation = position;
    return core::matrixOf(transform);
}

// A red quad filling the view's middle half, seen from +z, with no sky and no lights.
core::RenderWorldDelta redBackdrop() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{1.0f}))}};
    delta.materials = {{core::MaterialId{0}, glowing(kRed, 1.0f, core::AlphaMode::Opaque)}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    return delta;
}

core::vec3 rgb(core::vec4 const &pixel) { return pixel.xyz(); }

core::vec4 hdrAt(std::vector<core::vec4> const &pixels, int const x, int const y) {
    return pixels[static_cast<std::size_t>(y * kSize + x)];
}

bool near(core::vec3 const &a, core::vec3 const &b, float const tolerance) {
    return std::abs(a.x - b.x) <= tolerance and std::abs(a.y - b.y) <= tolerance and std::abs(a.z - b.z) <= tolerance;
}

// The HDR image of one frame of `pipeline` (with an HDR probe added), or the mean of
// `samples` frames of the path tracer.
struct HdrRenderer {
    explicit HdrRenderer(RenderHarness &harness_) : harness(&harness_) {
        target = harness->renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    }

    std::vector<core::vec4> visibility(core::RenderWorldDelta delta, core::CameraDesc const &camera) {
        auto &pipelines = harness->renderer->pipelines();
        auto view = test::viewOf(target, camera);
        view.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {probe}});
        harness->render(test::frameOf(std::move(delta), view));
        return *probe->pixels;
    }

    std::vector<core::vec4> pathTraced(core::RenderWorldDelta delta, core::CameraDesc const &camera, int const samples) {
        auto &pipelines = harness->renderer->pipelines();
        auto view = test::viewOf(target, camera, core::ViewId{1});
        view.pipeline = pipelines.create(render::PipelineDesc{pipelines.pathTraceReference(), {probe}});
        harness->render(test::frameOf(std::move(delta), view));
        for (auto frame = 1; frame < samples; ++frame) {
            harness->render(test::frameOf({}, view));
        }
        return *probe->pixels;
    }

    RenderHarness *harness;
    core::RenderTargetId target;
    std::shared_ptr<test::HdrProbe> probe = std::make_shared<test::HdrProbe>();
};

core::vec3 meanOver(std::vector<core::vec4> const &pixels, int const first, int const last) {
    auto sum = core::vec3{0.0f};
    for (auto y = first; y <= last; ++y) {
        for (auto x = first; x <= last; ++x) {
            sum += rgb(hdrAt(pixels, x, y));
        }
    }
    auto const count = static_cast<float>((last - first + 1) * (last - first + 1));
    return sum / count;
}

// A grey quad with alpha 0.5 in front of the red one: alpha E + (1 - alpha) red. The path
// tracer passes through it on half of its paths; over the 16x16 pixels it covers and 64
// samples each, the mean's noise is some 0.001.
TEST_CASE("a half-transparent quad blends over what lies behind by its alpha, in both pipelines") {
    auto delta = redBackdrop();
    delta.meshes.push_back({core::MeshId{1}, test::share(core::makeQuad(core::vec2{0.5f}))});
    delta.materials.push_back({core::MaterialId{1}, glowing(core::vec3{0.5f}, 0.5f, core::AlphaMode::Blend)});
    delta.instances.push_back({core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, at(core::vec3{0.0f, 0.0f, 1.0f})});
    auto const camera = test::halfFillingCamera(1.0f, 4.0f);
    auto const expected = core::vec3{0.5f} * 0.5f + kRed * 0.5f;
    auto harness = RenderHarness{};
    auto renderer = HdrRenderer{harness};

    auto const raster = renderer.visibility(delta, camera);
    auto const traced = renderer.pathTraced(std::move(delta), camera, 64);

    // The front quad covers about pixels 21 to 42; 24 to 39 are well inside it.
    auto blended = true;
    for (auto y = 24; y < 40; ++y) {
        for (auto x = 24; x < 40; ++x) {
            blended = blended and near(rgb(hdrAt(raster, x, y)), expected, 2e-3f);
        }
    }
    CHECK(blended);
    CHECK(near(rgb(hdrAt(raster, 18, 18)), kRed, 1e-3f));
    auto const mean = meanOver(traced, 24, 39);
    CAPTURE(mean.x);
    CAPTURE(mean.y);
    CHECK(near(mean, expected, 0.01f));
    CHECK(near(rgb(hdrAt(traced, 18, 18)), kRed, 1e-3f));
}

// A green quad (alpha 0.4) nearer than a blue one (alpha 0.6), which overlap in the middle.
core::RenderWorldDelta twoGlasses(bool const greenFirst) {
    auto delta = redBackdrop();
    delta.meshes.push_back({core::MeshId{1}, test::share(core::makeQuad(core::vec2{0.5f}))});
    delta.materials.push_back(
        {core::MaterialId{1}, glowing(core::vec3{0.0f, 0.8f, 0.0f}, 0.4f, core::AlphaMode::Blend)});
    delta.materials.push_back(
        {core::MaterialId{2}, glowing(core::vec3{0.0f, 0.0f, 0.9f}, 0.6f, core::AlphaMode::Blend)});
    auto const green = core::InstanceUpsert{core::InstanceId{greenFirst ? 1u : 2u}, core::MeshId{1},
                                            core::MaterialId{1}, at(core::vec3{-0.2f, 0.0f, 1.0f})};
    auto const blue = core::InstanceUpsert{core::InstanceId{greenFirst ? 2u : 1u}, core::MeshId{1},
                                           core::MaterialId{2}, at(core::vec3{0.2f, 0.0f, 0.5f})};
    delta.instances.push_back(greenFirst ? green : blue);
    delta.instances.push_back(greenFirst ? blue : green);
    return delta;
}

TEST_CASE("overlapping transparent quads blend in depth order whichever comes first, on every backend") {
    auto const camera = test::halfFillingCamera(1.0f, 4.0f);
    auto const behindBlue = core::vec3{0.0f, 0.0f, 0.9f} * 0.6f + kRed * 0.4f;
    auto const expected = core::vec3{0.0f, 0.8f, 0.0f} * 0.4f + behindBlue * 0.6f;
    auto reference = std::vector<core::vec4>{};
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        auto renderer = HdrRenderer{harness};
        auto const greenFirst = renderer.visibility(twoGlasses(true), camera);
        auto const backend = harness.renderer->deviceInfo().backend;
        auto other = RenderHarness{test::RenderHarness::withBackend(backend, test::deviceFor(backend))};
        auto otherRenderer = HdrRenderer{other};
        auto const blueFirst = otherRenderer.visibility(twoGlasses(false), camera);

        auto const sameImage = greenFirst == blueFirst;
        CHECK(sameImage);
        CAPTURE(rgb(hdrAt(greenFirst, kSize / 2, kSize / 2)).y);
        CHECK(near(rgb(hdrAt(greenFirst, kSize / 2, kSize / 2)), expected, 2e-3f));
        if (reference.empty()) {
            reference = greenFirst;
        }
        auto largest = 0.0f;
        for (auto index = std::size_t{0}; index < reference.size(); ++index) {
            for (auto c = 0; c < 3; ++c) {
                largest = std::max(largest, std::abs(reference[index][c] - greenFirst[index][c]));
            }
        }
        CHECK(largest < 2e-3f);
        if (harness.renderer->deviceInfo().backend != "cpu") {
            return;
        }
        // The path tracer passes each quad independently, though they share a mesh: 0.4 of
        // the paths stop at the green one, 0.6 x 0.6 at the blue one, the rest at the red.
        auto const traced = otherRenderer.pathTraced({}, camera, 64);
        auto const overlap = meanOver(traced, 28, 37);
        CAPTURE(overlap.z);
        CHECK(near(overlap, expected, 0.015f));
    });
}

// Alpha 0.2 in the left half of the texture and 0.9 in the right, cut at 0.5.
TEST_CASE("a masked quad is cut out where its alpha falls below the cutoff, in both pipelines") {
    auto delta = redBackdrop();
    delta.meshes.push_back({core::MeshId{1}, test::share(core::makeQuad(core::vec2{0.5f}))});
    auto masked = glowing(core::vec3{0.0f, 0.7f, 0.0f}, 1.0f, core::AlphaMode::Mask);
    masked.baseColorTexture = core::TextureId{0};
    delta.textures = {{core::TextureId{0},
                       test::rgba8Image(core::ivec2{16, 16}, core::ColorEncoding::Srgb,
                                        [](int const x, int) { return tc::ivec4{0, 0, 0, x < 8 ? 51 : 230}; }),
                       {}}};
    delta.materials.push_back({core::MaterialId{1}, masked});
    delta.instances.push_back({core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, at(core::vec3{0.0f, 0.0f, 1.0f})});
    auto const camera = test::halfFillingCamera(1.0f, 4.0f);
    auto harness = RenderHarness{};
    auto renderer = HdrRenderer{harness};
    renderer.visibility(std::move(delta), camera); // the texture arrives in the next frame

    for (auto const &image : {renderer.visibility({}, camera), renderer.pathTraced({}, camera, 4)}) {
        auto cut = true;
        auto kept = true;
        for (auto y = 24; y < 40; ++y) {
            for (auto x = 23; x < 30; ++x) {
                cut = cut and near(rgb(hdrAt(image, x, y)), kRed, 1e-3f);
            }
            for (auto x = 34; x < 41; ++x) {
                kept = kept and near(rgb(hdrAt(image, x, y)), core::vec3{0.0f, 0.7f, 0.0f}, 1e-3f);
            }
        }
        CHECK(cut);
        CHECK(kept);
    }
}

// The quad's corners project onto pixel boundaries, so its diagonal runs through the
// centres of the 32 pixels where x + y = 63: each must blend the quad once, like the rest.
TEST_CASE("the edge two triangles of a transparent quad share is blended once") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &harness) {
        auto delta = core::RenderWorldDelta{};
        delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{1.0f}))}};
        delta.materials = {{core::MaterialId{0}, glowing(core::vec3{0.5f}, 0.5f, core::AlphaMode::Blend)}};
        delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
        auto renderer = HdrRenderer{harness};

        auto const image = renderer.visibility(std::move(delta), test::halfFillingCamera(1.0f, 4.0f));

        auto covered = 0;
        auto once = 0;
        for (auto const &pixel : image) {
            if (pixel.x > 0.0f) {
                ++covered;
                once += std::abs(pixel.x - 0.25f) < 1e-3f ? 1 : 0;
            }
        }
        CHECK(covered == 32 * 32);
        CHECK(once == covered);
    });
}

} // namespace
