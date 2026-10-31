#include "RenderTestSupport.h"
#include "ShadowCascades.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

core::MaterialDesc grey(float const baseColor) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{baseColor, baseColor, baseColor, 1.0f};
    material.roughness = 1.0f;
    return material;
}

core::mat4 boxAt(core::vec3 const &centre, core::vec3 const &scale) {
    auto transform = core::Transform{};
    transform.translation = centre;
    transform.scale = scale;
    return core::matrixOf(transform);
}

// A grey plane, black boxes floating over it, and a sun that casts shadows.
core::RenderWorldDelta occluderScene(float const planeHalf, std::vector<core::mat4> const &boxes) {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makePlane(core::vec2{planeHalf}))},
                    {core::MeshId{1}, test::share(core::makeBox(core::vec3{0.5f}))}};
    delta.materials = {{core::MaterialId{0}, grey(0.5f)}, {core::MaterialId{1}, grey(0.0f)}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    for (auto index = std::size_t{0}; index < boxes.size(); ++index) {
        delta.instances.push_back({core::InstanceId{static_cast<std::uint32_t>(index + 1)}, core::MeshId{1},
                                   core::MaterialId{1}, boxes[index]});
    }
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{1.0f, -1.0f, 0.0f};
    sun.castsShadows = true;
    delta.lights = {{core::LightId{0}, sun}};
    return delta;
}

struct BothPipelines {
    render::ImageRgba8 visibility;
    render::ImageRgba8 pathTraced;
};

// A frame of the visibility pipeline, then `samples` frames of the path tracer, which
// meets the sun by next-event estimation and so has exact, hard shadows.
BothPipelines renderBoth(core::RenderWorldDelta delta, core::CameraDesc const &camera, int const size, int const samples) {
    auto harness = RenderHarness{};
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{size, size}});
    auto view = test::viewOf(target, camera);
    auto visibility = harness.render(test::frameOf(std::move(delta), view));
    view.pipeline = harness.renderer->pipelines().pathTraceReference();
    auto traced = render::FrameResult{};
    for (auto frame = 0; frame < samples; ++frame) {
        traced = harness.render(test::frameOf({}, view));
    }
    return BothPipelines{std::move(*visibility.views[0].image), std::move(*traced.views[0].image)};
}

// Whether the path traced pixel has a neighbour within two pixels more than two levels
// away: an edge of the shadow or of the box, where PCF blurs and the path tracer's
// jittered samples straddle.
bool nearEdge(render::ImageRgba8 const &image, int const x, int const y) {
    auto const centre = test::pixelAt(image, x, y);
    for (auto dy = -2; dy <= 2; ++dy) {
        for (auto dx = -2; dx <= 2; ++dx) {
            auto const nx = std::clamp(x + dx, 0, image.extent.x - 1);
            auto const ny = std::clamp(y + dy, 0, image.extent.y - 1);
            if (not test::nearPixel(test::pixelAt(image, nx, ny), centre, 2)) {
                return true;
            }
        }
    }
    return false;
}

// Seen from above, a box over the plane shadows a square of it; both pipelines must agree
// on every pixel away from the edges: the shadow as dark (the path tracer adds up to
// 2/255 of sunlight the black box's dielectric sides reflect into it, which the raster
// pipeline has no light for), the rest as lit, to a level.
TEST_CASE("the shadow map darkens what the path tracer's shadow rays darken, and nothing else") {
    constexpr auto kSize = 64;
    auto const camera = test::cameraAt(core::vec3{0.0f, 5.0f, 0.0f}, core::vec3{0.0f}, core::vec3{0.0f, 0.0f, -1.0f});
    auto const images = renderBoth(occluderScene(10.0f, {boxAt(core::vec3{0.0f, 1.5f, 0.0f}, core::vec3{1.0f})}),
                                   camera, kSize, 16);

    auto compared = 0;
    auto shadowed = 0;
    auto wrong = 0;
    for (auto y = 0; y < kSize; ++y) {
        for (auto x = 0; x < kSize; ++x) {
            if (nearEdge(images.pathTraced, x, y)) {
                continue;
            }
            auto const traced = test::pixelAt(images.pathTraced, x, y);
            auto const raster = test::pixelAt(images.visibility, x, y);
            ++compared;
            if (test::nearPixel(traced, 0u, 3)) {
                ++shadowed;
                wrong += test::nearPixel(raster, 0u, 3) ? 0 : 1;
            } else {
                wrong += test::nearPixel(raster, traced, 1) ? 0 : 1;
            }
        }
    }
    CAPTURE(compared);
    CAPTURE(shadowed);
    CHECK(compared > kSize * kSize / 2);
    CHECK(shadowed > 30);
    CHECK(wrong == 0);
}

// Seen from the camera: the projection of a point to pixel coordinates.
core::vec2 pixelOf(core::CameraDesc const &camera, int const size, core::vec3 const &point) {
    auto const clip = core::projectionMatrix(camera, 1.0f) * core::viewMatrix(camera) * core::vec4{point, 1.0f};
    return core::vec2{(clip.x / clip.w * 0.5f + 0.5f) * static_cast<float>(size),
                      (0.5f - clip.y / clip.w * 0.5f) * static_cast<float>(size)};
}

// One box near the camera, one 30 away: their shadows fall in the second and the last of
// three cascades (the first ends before the ground comes into view), and both match the
// path tracer's.
TEST_CASE("the cascades shadow receivers near the camera and far from it") {
    constexpr auto kSize = 96;
    auto const camera = test::cameraAt(core::vec3{0.0f, 4.0f, 2.0f}, core::vec3{0.0f, 0.0f, -16.0f});
    // The sun travels along (1, -1, 0): a box's shadow lies its height further along x.
    auto const nearBox = boxAt(core::vec3{-1.0f, 1.0f, -5.0f}, core::vec3{1.0f});
    auto const farBox = boxAt(core::vec3{-2.0f, 2.0f, -30.0f}, core::vec3{2.0f, 2.0f, 12.0f});
    auto const images = renderBoth(occluderScene(100.0f, {nearBox, farBox}), camera, kSize, 4);

    auto request = render::CascadeRequest{};
    request.camera = camera;
    request.lightDirection = tc::normalize(core::vec3{1.0f, -1.0f, 0.0f});
    request.sceneBounds = core::Aabb{core::vec3{-100.0f, -1.0f, -100.0f}, core::vec3{100.0f, 4.0f, 100.0f}};
    auto const cascades = render::fitShadowCascades(request);
    REQUIRE(cascades.size() == 3);
    auto const distanceOf = [&camera](core::vec3 const &point) { return tc::length(point - camera.position); };
    auto const nearShadow = core::vec3{0.0f, 0.0f, -5.0f};
    auto const farShadow = core::vec3{0.0f, 0.0f, -30.0f};
    CHECK(distanceOf(nearShadow) > cascades[0].farDistance);
    CHECK(distanceOf(nearShadow) < cascades[1].farDistance);
    CHECK(distanceOf(farShadow) > cascades[1].farDistance);
    CHECK(distanceOf(farShadow) < cascades[2].farDistance);

    for (auto const &pair : {std::pair{nearShadow, core::vec3{-4.0f, 0.0f, -5.0f}},
                             std::pair{farShadow, core::vec3{-7.0f, 0.0f, -30.0f}}}) {
        auto const point = pair.first;
        auto const lit = pair.second;
        CAPTURE(point.z);
        auto const shadowPixel = pixelOf(camera, kSize, point);
        auto const litPixel = pixelOf(camera, kSize, lit);
        auto const sx = static_cast<int>(shadowPixel.x);
        auto const sy = static_cast<int>(shadowPixel.y);
        auto const lx = static_cast<int>(litPixel.x);
        auto const ly = static_cast<int>(litPixel.y);
        REQUIRE(sx >= 0);
        REQUIRE(sx < kSize);
        REQUIRE(sy >= 0);
        REQUIRE(sy < kSize);
        CHECK(test::channel(test::pixelAt(images.pathTraced, sx, sy), 0) <= 2);
        CHECK(test::channel(test::pixelAt(images.visibility, sx, sy), 0) <= 2);
        CHECK(test::channel(test::pixelAt(images.pathTraced, lx, ly), 0) > 60);
        CHECK(test::nearPixel(test::pixelAt(images.visibility, lx, ly), test::pixelAt(images.pathTraced, lx, ly), 1));
    }
}

// Every cascade holds the corners of its slice of the view frustum, and a camera that
// moves a little moves the cascades by whole texels, so their shadows do not shimmer.
TEST_CASE("cascades hold their slices of the view and move by whole texels") {
    auto request = render::CascadeRequest{};
    request.camera = test::cameraAt(core::vec3{1.0f, 3.0f, 4.0f}, core::vec3{0.0f, 0.0f, -6.0f});
    request.aspect = 1.5f;
    request.lightDirection = tc::normalize(core::vec3{-0.4f, -1.0f, -0.5f});
    request.sceneBounds = core::Aabb{core::vec3{-20.0f, -1.0f, -60.0f}, core::vec3{20.0f, 5.0f, 10.0f}};
    request.settings.cascades = 4;
    request.settings.resolution = 512;
    request.settings.distance = 50.0f;
    auto const cascades = render::fitShadowCascades(request);
    REQUIRE(cascades.size() == 4);

    auto const forward = tc::normalize(request.camera.target - request.camera.position);
    auto const side = tc::normalize(tc::cross(forward, request.camera.up));
    auto const up = tc::cross(side, forward);
    auto const tangent = std::tan(request.camera.verticalFov * 0.5f);
    auto nearDistance = request.camera.nearPlane;
    for (auto const &cascade : cascades) {
        for (auto corner = 0u; corner < 8u; ++corner) {
            auto const distance = (corner & 4u) != 0 ? cascade.farDistance : nearDistance;
            auto const point = request.camera.position + forward * distance
                             + side * (((corner & 1u) != 0 ? 1.0f : -1.0f) * tangent * request.aspect * distance)
                             + up * (((corner & 2u) != 0 ? 1.0f : -1.0f) * tangent * distance);
            auto const at = cascade.worldToShadow * core::vec4{point, 1.0f};
            auto const inside = at.x > 0.0f and at.x < 1.0f and at.y > 0.0f and at.y < 1.0f and at.z > 0.0f and at.z < 1.0f;
            CHECK(inside);
        }
        nearDistance = cascade.farDistance;
    }
    CHECK(cascades.back().farDistance == doctest::Approx(50.0f));
    CHECK(cascades[0].texelSize < cascades[3].texelSize);

    auto moved = request;
    moved.camera.position += core::vec3{0.013f, 0.0f, -0.021f};
    moved.camera.target += core::vec3{0.013f, 0.0f, -0.021f};
    auto const shifted = render::fitShadowCascades(moved);
    for (auto index = std::size_t{0}; index < cascades.size(); ++index) {
        CAPTURE(index);
        CHECK(shifted[index].texelSize == cascades[index].texelSize);
        for (auto const axis : {0, 1}) {
            auto const texels = (shifted[index].worldToShadow[3][axis] - cascades[index].worldToShadow[3][axis]) * 512.0f;
            CHECK(std::abs(texels - std::round(texels)) < 1e-2f);
        }
    }
}

} // namespace
