#include "KernelVariants.h"
#include "RenderTestSupport.h"
#include "TestMaterialModels.h"

#include <lr/core/Primitives.h>
#include <lr/render/MaterialModel.h>
#include <lr/render/Renderer.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kSize = 48;

// What a device with these limits would say of the programs transpiled into the tests.
struct LimitsCheck final : render::KernelCheck {
    explicit LimitsCheck(tc::DeviceLimits const &limits_) : limits(limits_) {}

    std::optional<std::string> whyNot(std::string_view const program) const override {
        auto const found = tc::ProgramRegistry::instance().find(program);
        if (not found) {
            return std::string{"not transpiled"};
        }
        return tc::exceedsLimits(*found, limits);
    }

    tc::DeviceLimits limits;
};

TEST_CASE("each device gets the variants it runs, and none it is told not to use") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &harness) {
        auto const &info = harness.renderer->deviceInfo();
        auto const &chosen = harness.renderer->capabilities();
        MESSAGE(info.backend << " on " << info.deviceName << ": subgroups " << chosen.subgroups << " (of "
                             << chosen.subgroupSize << "), single-pass visibility " << chosen.singlePassVisibility);
        CHECK(chosen.subgroups == tc::uses(info.limits.features, tc::KernelFeature::Subgroups));
        CHECK(chosen.singlePassVisibility == tc::uses(info.limits.features, tc::KernelFeature::Int64Atomics));
        CHECK(chosen.subgroupSize == info.limits.subgroupSize);
        // Mesa's llvmpipe has neither on OpenGL and OpenGL ES; lavapipe has both.
        auto const both = info.backend == "cpu" or info.backend == "vulkan";
        CHECK(chosen.subgroups == both);
        CHECK(chosen.singlePassVisibility == both);
    });

    auto options = test::HarnessOptions{};
    options.subgroups = render::VariantUse::Off;
    options.singlePassVisibility = render::VariantUse::Off;
    auto const forced = RenderHarness{options};
    CHECK_FALSE(forced.renderer->capabilities().subgroups);
    CHECK_FALSE(forced.renderer->capabilities().singlePassVisibility);
}

TEST_CASE("a device short of what a kernel needs is refused, with the kernels named") {
    auto const harness = RenderHarness{};
    auto const limits = harness.renderer->deviceInfo().limits;

    auto const everything = render::chooseCapabilities(LimitsCheck{limits}, render::VariantChoices{});
    CHECK(everything.subgroups);
    CHECK(everything.singlePassVisibility);

    auto plain = limits;
    plain.features = 0;
    auto const fallback = render::chooseCapabilities(LimitsCheck{plain}, render::VariantChoices{});
    CHECK_FALSE(fallback.subgroups);
    CHECK_FALSE(fallback.singlePassVisibility);

    auto cramped = limits;
    cramped.storageBuffers = 4;
    auto message = std::string{};
    try {
        render::chooseCapabilities(LimitsCheck{cramped}, render::VariantChoices{});
    } catch (tc::DeviceError const &error) {
        message = error.what();
    }
    MESSAGE(message);
    CHECK(message.find("lr_shade_metallic_roughness") != std::string::npos);
    CHECK(message.find("lr_blend_transparent") != std::string::npos);
    CHECK(message.find("lr_path_trace") != std::string::npos);
    CHECK(message.find("storage buffers") != std::string::npos);
    CHECK(message.find("lr_tonemap") == std::string::npos);

    // A recreated device must run what was chosen for the first, and the models' kernels.
    CHECK_THROWS_AS(render::verifyCapabilities(LimitsCheck{plain}, everything, {}), tc::DeviceError);
    CHECK_NOTHROW(render::verifyCapabilities(LimitsCheck{plain}, fallback, {}));
    CHECK_THROWS_AS(render::verifyCapabilities(LimitsCheck{limits}, everything, {"lr_test_not_transpiled"}),
                    tc::DeviceError);
}

TEST_CASE("the renderer checks every program of lrender's") {
    auto const &programs = render::lrenderPrograms();
    auto listed = std::set<std::string_view>{programs.required.begin(), programs.required.end()};
    listed.insert(programs.subgroups.begin(), programs.subgroups.end());
    listed.insert(programs.singlePassVisibility.begin(), programs.singlePassVisibility.end());
    for (auto const name : listed) {
        CAPTURE(name);
        CHECK(tc::ProgramRegistry::instance().find(name).has_value());
    }
    for (auto const &program : tc::ProgramRegistry::instance().all()) {
        auto const name = std::string_view{program.name};
        if (name.substr(0, 3) == "lr_" and name.substr(0, 8) != "lr_test_") {
            CAPTURE(name);
            CHECK(listed.count(name) == 1);
        }
    }
}

// More instances than a compaction workgroup holds, some culled; two coincident quads, so
// that triangles tie in depth; a masked lattice; a model of the tests' own; a shadowed sun.
core::RenderWorldDelta busyScene(core::MaterialModelId const flat) {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makePlane(core::vec2{4.0f}))},
                    {core::MeshId{1}, test::share(core::makeBox(core::vec3{0.06f}))},
                    {core::MeshId{2}, test::share(core::makeQuad(core::vec2{0.5f}))},
                    {core::MeshId{3}, test::share(core::makeSphere(0.3f, 16, 8))}};
    delta.textures = {{core::TextureId{0}, test::rgba8Image(core::ivec2{32, 32}, core::ColorEncoding::Srgb,
                                                            [](int const x, int const y) {
                                                                auto const hole = (x / 4 + y / 4) % 3 == 0;
                                                                return tc::ivec4{80, 160, 90, hole ? 0 : 255};
                                                            }),
                       {}}};
    auto lattice = core::MaterialDesc{};
    lattice.baseColorTexture = core::TextureId{0};
    lattice.alphaMode = core::AlphaMode::Mask;
    lattice.doubleSided = true;
    auto red = core::MaterialDesc{};
    red.baseColor = core::vec4{0.8f, 0.2f, 0.1f, 1.0f};
    auto flatGreen = core::MaterialDesc{};
    flatGreen.model = flat;
    flatGreen.modelParameters = core::vec4{0.1f, 0.7f, 0.2f, 0.0f};
    delta.materials = {{core::MaterialId{0}, core::MaterialDesc{}},
                       {core::MaterialId{1}, red},
                       {core::MaterialId{2}, lattice},
                       {core::MaterialId{3}, flatGreen}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    auto next = std::uint32_t{1};
    for (auto row = 0; row < 18; ++row) {
        for (auto column = 0; column < 18; ++column) {
            auto const at = core::vec3{-2.6f + 0.3f * static_cast<float>(column), 0.06f,
                                       -2.6f + 0.3f * static_cast<float>(row)};
            auto const material = core::MaterialId{(row + column) % 2 == 0 ? 1u : 3u};
            delta.instances.push_back({core::InstanceId{next++}, core::MeshId{1}, material, test::translation(at)});
        }
    }
    auto const quad = test::translation(core::vec3{-0.4f, 0.6f, 0.3f});
    delta.instances.push_back({core::InstanceId{next++}, core::MeshId{2}, core::MaterialId{1}, quad});
    delta.instances.push_back({core::InstanceId{next++}, core::MeshId{2}, core::MaterialId{3}, quad});
    delta.instances.push_back({core::InstanceId{next++}, core::MeshId{2}, core::MaterialId{2},
                               test::translation(core::vec3{0.5f, 0.5f, 0.0f})});
    delta.instances.push_back(
        {core::InstanceId{next++}, core::MeshId{3}, core::MaterialId{0}, test::translation(core::vec3{0.3f, 0.3f, 0.6f})});
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.castsShadows = true;
    delta.lights = {{core::LightId{0}, sun}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.1f, 0.15f, 0.25f}};
    return delta;
}

struct ProbedFrame {
    render::ImageRgba8 image;
    std::vector<tc::uint> triangles;
    std::vector<tc::uint> depth;
    std::vector<tc::uvec4> visible; // the visible instances, without what lies past them
    std::uint32_t visibleInstances = 0;
};

ProbedFrame renderBusyScene(RenderHarness &harness) {
    auto const flat = harness.renderer->materialModels().add(render::makeMaterialModel<test::FlatColour>());
    auto const probe = std::make_shared<test::VisibilityProbe>();
    auto &pipelines = harness.renderer->pipelines();
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto view = test::viewOf(target, test::cameraAt(core::vec3{0.2f, 2.2f, 3.0f}, core::vec3{0.0f, 0.2f, 0.0f}));
    view.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {probe}});
    auto result = harness.render(test::frameOf(busyScene(flat), view));
    auto frame = ProbedFrame{std::move(*result.views[0].image), *probe->triangles, *probe->depth, *probe->visible,
                             result.views[0].stats.visibleInstances};
    frame.visible.resize(std::size_t{frame.visibleInstances} + 1);
    return frame;
}

// The same backend and device as `harness`, with `options`' variants.
test::HarnessOptions sameDevice(RenderHarness const &harness, test::HarnessOptions options) {
    options.backend = harness.renderer->deviceInfo().backend;
    options.device = options.backend == "cpu" ? "" : "llvmpipe";
    return options;
}

bool sameQuads(std::vector<tc::uvec4> const &a, std::vector<tc::uvec4> const &b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](tc::uvec4 const &x, tc::uvec4 const &y) { return x == y; });
}

TEST_CASE("one 64-bit pass and two 32-bit passes find the same triangles and depths") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &onePass) {
        if (not onePass.renderer->capabilities().singlePassVisibility) {
            MESSAGE("no 64-bit atomics on " << onePass.renderer->deviceInfo().backend);
            return;
        }
        auto options = sameDevice(onePass, test::HarnessOptions{});
        options.singlePassVisibility = render::VariantUse::Off;
        auto twoPasses = RenderHarness{options};
        REQUIRE_FALSE(twoPasses.renderer->capabilities().singlePassVisibility);

        auto const a = renderBusyScene(onePass);
        auto const b = renderBusyScene(twoPasses);
        auto const covered = std::count_if(a.triangles.begin(), a.triangles.end(),
                                           [](tc::uint const triangle) { return triangle != render::noTriangle(); });
        MESSAGE(covered << " of " << a.triangles.size() << " pixels covered, " << a.visibleInstances
                        << " instances visible");
        CHECK(covered > static_cast<long>(a.triangles.size() / 2));
        // Visible instances from both of the compaction's chunks of 256.
        CHECK(a.visible[1].x < 256);
        CHECK(a.visible.back().x >= 256);
        CHECK(a.triangles == b.triangles);
        CHECK(a.depth == b.depth);
        CHECK(a.image.pixels == b.image.pixels);
    });
}

TEST_CASE("compaction and classification find the same with subgroups as with shared memory alone") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &subgroups) {
        if (not subgroups.renderer->capabilities().subgroups) {
            MESSAGE("no subgroups on " << subgroups.renderer->deviceInfo().backend);
            return;
        }
        auto options = sameDevice(subgroups, test::HarnessOptions{});
        options.subgroups = render::VariantUse::Off;
        auto shared = RenderHarness{options};
        REQUIRE_FALSE(shared.renderer->capabilities().subgroups);

        auto const a = renderBusyScene(subgroups);
        auto const b = renderBusyScene(shared);
        CHECK(a.visibleInstances == b.visibleInstances);
        CHECK(sameQuads(a.visible, b.visible));
        CHECK(a.triangles == b.triangles);
        CHECK(a.image.pixels == b.image.pixels);
    });
}

} // namespace
