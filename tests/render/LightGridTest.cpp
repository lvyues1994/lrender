#include "RenderTestSupport.h"

#include <lr/core/Primitives.h>
#include <lr/render/Features.h>
#include <lr/render/GpuTypes.h>
#include <lr/render/Renderer.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using lr::test::RenderHarness;
namespace core = lr::core;
namespace render = lr::render;

// Not a multiple of the grid's tiles either way.
constexpr auto kExtent = core::ivec2{200, 150};
constexpr auto kPointLights = 300u;
constexpr auto kUnlimitedLight = 137u; // a point light without a range, among the others

// Reads back the view's light grid.
struct LightGridProbe final : render::RenderFeature {
    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto const tiles = [](int const pixels) {
            return static_cast<std::size_t>((pixels + static_cast<int>(render::lightTileSize()) - 1)
                                            / static_cast<int>(render::lightTileSize()));
        };
        cells = view.clusterLights ? tiles(view.extent.x) * tiles(view.extent.y) * render::lightSlices() : 1;
        lights = view.lightCount;
        auto const words = render::lightGridHeaderWords() + std::size_t{render::lightRecordWords()} * lights
                         + cells * render::lightMaskWords(lights);
        lr::test::readBack(graph, "probed light grid", graph.blackboard().get<render::LightGridOutputs>().grid, words,
                           grid);
    }

    // The bits set over all the cells' masks.
    std::size_t heldLights() const {
        auto held = std::size_t{0};
        for (auto word = render::lightGridHeaderWords() + std::size_t{render::lightRecordWords()} * lights;
             word < grid->size(); ++word) {
            for (auto bits = (*grid)[word]; bits != 0u; bits &= bits - 1u) {
                ++held;
            }
        }
        return held;
    }

    std::size_t cells = 0;
    std::uint32_t lights = 0;
    std::shared_ptr<std::vector<tc::uint>> grid = std::make_shared<std::vector<tc::uint>>();
};

// Spheres on a floor, one of them metal and one glass, lit by a shadowed sun and many
// small lights: in front of the camera, behind it, across its near plane and off the
// screen at either side, and one with no range at all.
core::RenderWorldDelta manyLights() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, lr::test::share(core::makePlane(core::vec2{12.0f}))},
                    {core::MeshId{1}, lr::test::share(core::makeSphere(0.45f, 24, 12))}};
    auto rough = core::MaterialDesc{};
    rough.baseColor = core::vec4{0.8f, 0.7f, 0.6f, 1.0f};
    rough.roughness = 0.7f;
    auto metal = core::MaterialDesc{};
    metal.baseColor = core::vec4{0.9f, 0.8f, 0.5f, 1.0f};
    metal.metallic = 1.0f;
    metal.roughness = 0.3f;
    auto glass = core::MaterialDesc{};
    glass.baseColor = core::vec4{0.6f, 0.7f, 0.9f, 0.5f};
    glass.alphaMode = core::AlphaMode::Blend;
    delta.materials = {{core::MaterialId{0}, rough}, {core::MaterialId{1}, metal}, {core::MaterialId{2}, glass}};
    auto next = std::uint32_t{0};
    delta.instances.push_back({core::InstanceId{next++}, core::MeshId{0}, core::MaterialId{0},
                               lr::test::translation(core::vec3{0.0f, -0.5f, 0.0f})});
    for (auto row = 0; row < 4; ++row) {
        for (auto column = 0; column < 4; ++column) {
            auto const material = core::MaterialId{next == 6 ? 1u : next == 11 ? 2u : 0u};
            auto const at = core::vec3{-2.4f + 1.6f * static_cast<float>(column), 0.0f,
                                       -3.0f + 1.6f * static_cast<float>(row)};
            delta.instances.push_back({core::InstanceId{next++}, core::MeshId{1}, material, lr::test::translation(at)});
        }
    }
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.3f, -1.0f, -0.4f};
    sun.illuminance = 0.6f;
    sun.castsShadows = true;
    delta.lights.push_back({core::LightId{0}, sun});
    auto seed = std::uint32_t{12345};
    auto const random = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8u) / static_cast<float>(1u << 24u);
    };
    for (auto index = 0u; index < kPointLights; ++index) {
        auto lamp = core::PointLight{};
        lamp.position = core::vec3{-7.0f + 14.0f * random(), -0.4f + 2.0f * random(), -6.0f + 13.0f * random()};
        lamp.color = core::vec3{0.3f + 0.7f * random(), 0.3f + 0.7f * random(), 0.3f + 0.7f * random()};
        lamp.intensity = 0.5f + 1.5f * random();
        lamp.range = index == kUnlimitedLight ? 0.0f : 0.4f + 1.6f * random();
        delta.lights.push_back({core::LightId{index + 1u}, lamp});
    }
    delta.environment = core::EnvironmentDesc{core::vec3{0.02f, 0.025f, 0.03f}};
    return delta;
}

struct LitFrames {
    std::vector<std::vector<core::vec4>> hdr; // per frame
    std::vector<std::vector<std::uint32_t>> pixels;
    std::size_t heldLights = 0; // of the last frame
    std::size_t cells = 0;
};

// Two still frames, then two with temporal anti-aliasing, whose projection is jittered.
LitFrames renderLit(lr::test::HarnessOptions options, std::string const &backend) {
    options.backend = backend;
    options.device = lr::test::deviceFor(backend);
    auto harness = RenderHarness{options};
    auto const hdr = std::make_shared<lr::test::HdrProbe>();
    auto const grid = std::make_shared<LightGridProbe>();
    auto &pipelines = harness.renderer->pipelines();
    auto const pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {hdr, grid}});
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{kExtent});
    auto frames = LitFrames{};
    auto scene = manyLights();
    for (auto frame = 0; frame < 4; ++frame) {
        auto camera = lr::test::cameraAt(core::vec3{0.6f, 1.4f, 4.2f}, core::vec3{0.0f, -0.2f, -0.5f});
        camera.nearPlane = 0.3f;
        auto view = lr::test::viewOf(target, camera);
        view.pipeline = pipeline;
        view.settings.antiAliasing = frame < 2 ? render::AntiAliasing::None : render::AntiAliasing::Taa;
        auto result = harness.render(lr::test::frameOf(std::exchange(scene, core::RenderWorldDelta{}), view));
        frames.hdr.push_back(*hdr->pixels);
        frames.pixels.push_back(result.views[0].image->pixels);
    }
    frames.heldLights = grid->heldLights();
    frames.cells = grid->cells;
    return frames;
}

bool sameBits(std::vector<core::vec4> const &a, std::vector<core::vec4> const &b) {
    return a.size() == b.size() and std::memcmp(a.data(), b.data(), a.size() * sizeof(core::vec4)) == 0;
}

TEST_CASE("each point is lit by the lights of its cell of the grid exactly as by every light") {
    for (auto const &backend : tc::availableBackends()) {
        CAPTURE(backend);
        auto clustered = LitFrames{};
        auto everyLight = LitFrames{};
        try {
            auto options = lr::test::HarnessOptions{};
            clustered = renderLit(options, backend);
            options.clusterLights = false;
            everyLight = renderLit(options, backend);
        } catch (tc::DeviceError const &error) {
            MESSAGE("skipped: " << error.what());
            continue;
        }
        auto const lights = std::size_t{kPointLights + 1};
        MESSAGE(clustered.heldLights << " lights held over " << clustered.cells << " cells, of " << lights
                                     << " each");
        CHECK(everyLight.heldLights == lights);
        CHECK(clustered.heldLights < clustered.cells * lights / 8);
        // The sun and the light without a range light every point.
        CHECK(clustered.heldLights >= 2 * clustered.cells);
        for (auto frame = std::size_t{0}; frame < clustered.hdr.size(); ++frame) {
            CAPTURE(frame);
            CHECK(sameBits(clustered.hdr[frame], everyLight.hdr[frame]));
            CHECK(clustered.pixels[frame] == everyLight.pixels[frame]);
        }
    }
}

} // namespace
