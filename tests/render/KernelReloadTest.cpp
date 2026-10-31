#include "RenderTestSupport.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>

#include <tc/program.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

namespace {

using namespace lr;

struct KernelCopy {
    std::filesystem::path directory;
    std::size_t programs = 0;
};

// The build's kernel directory copied, with a comment appended to every OpenGL GLSL file:
// each program with one then differs from what the registry holds, and behaves the same.
// Programs only Vulkan runs (ray queries) have none and stay as they are.
KernelCopy editedCopyOf(std::string const &kernels, int const edit) {
    auto copy = KernelCopy{std::filesystem::path{LR_TEST_OUTPUT_DIR} / ("kernel-reload-" + std::to_string(edit)), 0};
    std::filesystem::remove_all(copy.directory);
    std::filesystem::create_directories(copy.directory);
    auto edited = std::set<std::string>{};
    auto registered = std::set<std::string>{};
    for (auto const &entry : std::filesystem::directory_iterator{kernels}) {
        auto const target = copy.directory / entry.path().filename();
        std::filesystem::copy_file(entry.path(), target);
        auto const name = entry.path().filename().string();
        auto const glsl = name.size() > 5 and name.compare(name.size() - 5, 5, ".comp") == 0
                      and name.find(".vk.") == std::string::npos;
        if (glsl) {
            std::ofstream{target, std::ios::app} << "\n// edited " << edit << "\n";
            edited.insert(name.substr(0, name.find('.')));
        }
        // Builds leave the files of kernels since renamed behind; reloading skips them.
        if (entry.path().extension() == ".interface"
            and tc::ProgramRegistry::instance().find(entry.path().stem().string())) {
            registered.insert(entry.path().stem().string());
        }
    }
    copy.programs = static_cast<std::size_t>(std::count_if(
        registered.begin(), registered.end(), [&edited](std::string const &program) { return edited.count(program) != 0; }));
    return copy;
}

core::RenderWorldDelta litScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeSphere(1.0f, 24, 12))},
                    {core::MeshId{1}, test::share(core::makePlane(core::vec2{4.0f}))}};
    auto shiny = core::MaterialDesc{};
    shiny.baseColor = core::vec4{0.8f, 0.5f, 0.3f, 1.0f};
    shiny.roughness = 0.3f;
    delta.materials = {{core::MaterialId{0}, shiny}, {core::MaterialId{1}, test::emissive(core::vec3{0.1f, 0.2f, 0.1f})}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, test::translation(core::vec3{0.0f, 1.0f, 0.0f})},
                       {core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, core::mat4{1.0f}}};
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.3f, -1.0f, -0.4f};
    sun.illuminance = 2.0f;
    sun.castsShadows = true;
    delta.lights = {{core::LightId{0}, sun}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.1f}};
    return delta;
}

TEST_CASE("reloading lrender's kernels replaces every program without rejections, and frames stay the same") {
    auto const kernels = render::builtInKernelDirectory();
    if (not kernels) {
        MESSAGE("skipped: built without LRENDER_KERNEL_RELOAD");
        return;
    }
    auto edit = 0;
    test::forEachBackend(test::HarnessOptions{}, [&](test::RenderHarness &harness) {
        auto const copy = editedCopyOf(*kernels, ++edit);
        REQUIRE(copy.programs > 20);
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{48, 32}});
        auto const view = test::viewOf(target, test::cameraAt(core::vec3{0.0f, 2.5f, 5.0f}, core::vec3{0.0f, 0.8f, 0.0f}));
        auto const before = harness.render(test::frameOf(litScene(), view));

        auto const reloaded = lexec::sync_wait(harness.renderer->reloadKernels(copy.directory.string()));

        REQUIRE(reloaded.has_value());
        auto const &report = std::get<0>(*reloaded);
        for (auto const &rejection : report.rejected) {
            CAPTURE(rejection.program);
            FAIL(rejection.reason);
        }
        CHECK(report.reloaded.size() == copy.programs);
        auto const after = harness.render(test::frameOf({}, view));
        REQUIRE(before.views.at(0).image.has_value());
        REQUIRE(after.views.at(0).image.has_value());
        CHECK(test::maxChannelDifference(*before.views[0].image, *after.views[0].image) == 0);
    });
}

} // namespace
