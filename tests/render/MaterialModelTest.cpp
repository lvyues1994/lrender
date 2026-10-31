#include "RenderTestSupport.h"
#include "TestMaterialModels.h"

#include <lr/core/Primitives.h>
#include <lr/render/MaterialModel.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kWidth = 48;
constexpr auto kHeight = 16;
core::vec4 const kFlat{0.25f, 0.5f, 0.75f, 0.0f};
core::vec4 const kPaint{0.8f, 0.3f, 0.2f, 0.0f};

// Three lit quads side by side seen from +z, under a sun and a sky: the left one's
// material is metallic-roughness, the middle one's `middle` and the right one's `right`.
core::RenderWorldDelta threeQuads(core::MaterialDesc const &middle, core::MaterialDesc const &right) {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{0.45f}))}};
    auto left = core::MaterialDesc{};
    left.baseColor = core::vec4{0.5f, 0.6f, 0.7f, 1.0f};
    left.roughness = 0.3f;
    delta.materials = {{core::MaterialId{0}, left}, {core::MaterialId{1}, middle}, {core::MaterialId{2}, right}};
    delta.instances = {
        {core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, test::translation(core::vec3{-1.0f, 0.0f, 0.0f})},
        {core::InstanceId{1}, core::MeshId{0}, core::MaterialId{1}, core::mat4{1.0f}},
        {core::InstanceId{2}, core::MeshId{0}, core::MaterialId{2}, test::translation(core::vec3{1.0f, 0.0f, 0.0f})}};
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.3f, -0.4f, -1.0f};
    sun.illuminance = 2.0f;
    delta.lights = {{core::LightId{0}, sun}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.1f, 0.15f, 0.2f}};
    return delta;
}

core::CameraDesc frontCamera() {
    auto camera = test::cameraAt(core::vec3{0.0f, 0.0f, 2.0f}, core::vec3{0.0f});
    camera.verticalFov = 0.6f;
    return camera;
}

std::vector<core::vec4> renderHdr(RenderHarness &harness, core::RenderWorldDelta delta) {
    auto const probe = std::make_shared<test::HdrProbe>();
    auto &pipelines = harness.renderer->pipelines();
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kWidth, kHeight}});
    auto view = test::viewOf(target, frontCamera());
    view.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {probe}});
    harness.render(test::frameOf(std::move(delta), view));
    return *probe->pixels;
}

// Columns [first, last) of every row.
bool sameColumns(std::vector<core::vec4> const &a, std::vector<core::vec4> const &b, int const first, int const last) {
    for (auto y = 0; y < kHeight; ++y) {
        for (auto x = first; x < last; ++x) {
            auto const index = static_cast<std::size_t>(y * kWidth + x);
            if (not(a[index] == b[index])) {
                return false;
            }
        }
    }
    return true;
}

core::MaterialDesc ofModel(core::MaterialModelId const model, core::vec4 const &parameters) {
    auto material = core::MaterialDesc{};
    material.model = model;
    material.modelParameters = parameters;
    return material;
}

// The left quad covers columns 5 to 16, the middle one 18 to 29 and the right one 31 to 42.
constexpr auto kLeftEnd = 17;
constexpr auto kMiddleStart = 18;
constexpr auto kMiddleEnd = 30;

TEST_CASE("an application's material models shade their pixels and the built-in model's stay as they were") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &harness) {
        auto &models = harness.renderer->materialModels();
        auto const flat = models.add(render::makeMaterialModel<test::FlatColour>());
        auto const painted = models.add(render::makeMaterialModel<test::PaintedMetallicRoughness>());
        CHECK(flat.value == 1);
        CHECK(painted.value == 2);
        CHECK(models.count() == 3);

        auto builtinPaint = core::MaterialDesc{};
        builtinPaint.baseColor = core::vec4{kPaint.x, kPaint.y, kPaint.z, 1.0f};
        auto const allBuiltin = renderHdr(harness, threeQuads(builtinPaint, builtinPaint));
        auto const mixed = renderHdr(harness, threeQuads(ofModel(painted, kPaint), ofModel(flat, kFlat)));

        CHECK(sameColumns(mixed, allBuiltin, 0, kLeftEnd));
        auto const backend = harness.renderer->deviceInfo().backend;
        auto const middle = mixed[static_cast<std::size_t>(kHeight / 2 * kWidth + kWidth / 2)];
        auto const builtinMiddle = allBuiltin[static_cast<std::size_t>(kHeight / 2 * kWidth + kWidth / 2)];
        CAPTURE(backend);
        // The painted model lights its surface as the built-in one does: on the CPU
        // backend to the bit; GPU compilers may fuse the two programs' arithmetic apart.
        if (backend == "cpu") {
            CHECK(sameColumns(mixed, allBuiltin, kMiddleStart, kMiddleEnd));
        } else {
            CHECK(middle.x == doctest::Approx(builtinMiddle.x).epsilon(1e-3));
            CHECK(middle.y == doctest::Approx(builtinMiddle.y).epsilon(1e-3));
        }
        CHECK(middle.x > middle.y);
        auto const right = mixed[static_cast<std::size_t>(kHeight / 2 * kWidth + kWidth - 8)];
        CHECK(right.x == kFlat.x);
        CHECK(right.y == kFlat.y);
        CHECK(right.z == kFlat.z);
    });
}

TEST_CASE("materials of a material model the renderer does not have are rejected") {
    auto harness = RenderHarness{};
    auto unknown = threeQuads(core::MaterialDesc{}, ofModel(core::MaterialModelId{1}, kFlat));

    CHECK_THROWS_AS(renderHdr(harness, unknown), std::invalid_argument);

    auto const flat = harness.renderer->materialModels().add(render::makeMaterialModel<test::FlatColour>());
    REQUIRE(flat.value == 1);
    auto const shown = renderHdr(harness, unknown);
    CHECK(shown[static_cast<std::size_t>(kHeight / 2 * kWidth + kWidth - 8)].z == kFlat.z);
}

// A model whose kernel nobody transpiled.
struct MissingModel final : render::MaterialModel {
    std::string_view program() const noexcept override { return "lr_test_no_such_model"; }
    void shade(render::ModelShading &) override {}
};

TEST_CASE("a model is added only if the device runs its kernel, and only so many") {
    auto harness = RenderHarness{};
    auto &models = harness.renderer->materialModels();
    auto message = std::string{};
    try {
        models.add(std::make_shared<MissingModel>());
    } catch (tc::DeviceError const &error) {
        message = error.what();
    }
    CHECK(message.find("lr_test_no_such_model") != std::string::npos);
    CHECK(models.count() == 1);
    CHECK_THROWS_AS(models.add(nullptr), std::invalid_argument);

    while (models.count() < render::maxMaterialModels()) {
        models.add(render::makeMaterialModel<test::FlatColour>());
    }
    CHECK_THROWS_AS(models.add(render::makeMaterialModel<test::FlatColour>()), std::length_error);
}

TEST_CASE("transparent fragments are shaded with the metallic-roughness model whatever their material names") {
    auto harness = RenderHarness{};
    auto const flat = harness.renderer->materialModels().add(render::makeMaterialModel<test::FlatColour>());
    auto glass = core::MaterialDesc{};
    glass.baseColor = core::vec4{0.6f, 0.8f, 1.0f, 0.4f};
    glass.alphaMode = core::AlphaMode::Blend;
    auto flatGlass = glass;
    flatGlass.model = flat;
    flatGlass.modelParameters = kFlat;

    auto const builtin = renderHdr(harness, threeQuads(glass, core::MaterialDesc{}));
    auto const named = renderHdr(harness, threeQuads(flatGlass, core::MaterialDesc{}));

    CHECK(sameColumns(builtin, named, 0, kWidth));
}

} // namespace
