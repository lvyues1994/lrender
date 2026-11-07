// The Vulkan presenter where it shows nothing: frames read back from the images it copied
// them into are compared with an image target's of the same frame. Mesa's lavapipe has
// headless surfaces and so a swapchain; other devices (NVIDIA's crashes on a headless
// surface) get the presenter's offscreen images instead.

#include "PresentTestSupport.h"

#include <lr/present/VulkanPresenter.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kExtent = core::ivec2{48, 32};
constexpr auto kFrames = 12u;

// Keeps the Vulkan drivers loaded until the process ends. Where a presenter's instance is
// the only one (the GL backends' tests), the loader unloads every driver when it goes, and
// one of them leaves memory its globals held, which LeakSanitizer then reports.
void keepVulkanDriversLoaded() {
    static auto const instance = [] {
        auto app = VkApplicationInfo{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_1;
        auto info = VkInstanceCreateInfo{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        auto created = VkInstance{VK_NULL_HANDLE};
        return vkCreateInstance(&info, nullptr, &created) == VK_SUCCESS ? created : VK_NULL_HANDLE;
    }();
    static_cast<void>(instance);
}

// A presenter on the device the renderer's GPU backends use (test::deviceFor); null, with
// the reason in `skipped`, where it has no headless surface.
std::shared_ptr<present::VulkanPresenter> headlessPresenter(bool const shareBuffers, std::string &skipped) {
    keepVulkanDriversLoaded();
    auto config = present::VulkanPresenterConfig{};
    config.deviceNameContains = test::deviceFor("vulkan");
    config.offscreen = config.deviceNameContains != "llvmpipe";
    config.shareBuffers = shareBuffers;
    try {
        return present::makeVulkanPresenter(config);
    } catch (std::runtime_error const &error) {
        skipped = error.what();
        return nullptr;
    }
}

// Each GPU backend's renderer with a presenter, unless one of them cannot start.
template <class Fn>
void forEachGpuBackend(bool const shareBuffers, Fn const &fn) {
    auto const requireGpu = std::getenv("LRENDER_TEST_REQUIRE_GPU") != nullptr;
    for (auto const &backend : tc::availableBackends()) {
        if (backend == "cpu") {
            continue;
        }
        CAPTURE(backend);
        auto harness = std::unique_ptr<RenderHarness>{};
        try {
            harness = std::make_unique<RenderHarness>(backend, test::deviceFor(backend));
        } catch (tc::DeviceError const &error) {
            if (requireGpu) {
                FAIL(error.what());
            }
            MESSAGE("skipped: " << error.what());
            continue;
        }
        auto skipped = std::string{};
        auto const presenter = headlessPresenter(shareBuffers, skipped);
        if (presenter == nullptr) {
            MESSAGE("skipped: " << skipped);
            continue;
        }
        fn(*harness, presenter);
    }
}

bool takesSharedPath(RenderHarness const &harness, present::VulkanPresenter const &presenter) {
    auto const &info = harness.renderer->deviceInfo();
    return info.syncFiles and info.importsMemory and presenter.sharesBuffersWith(info);
}

core::RenderWorldDelta coloredQuad(core::vec3 const &radiance) {
    auto delta = test::presentedScene();
    delta.materials = {{core::MaterialId{0}, test::emissive(radiance)}};
    return delta;
}

// Waits for the capture, which the next frame shown fulfils.
render::ImageRgba8 captured(std::future<render::ImageRgba8> &capture) {
    REQUIRE(capture.wait_for(std::chrono::seconds{30}) == std::future_status::ready);
    return capture.get();
}

// One frame to an image target and the presenter's window target: what each shows.
struct BothTargets {
    render::ImageRgba8 image;
    render::ImageRgba8 presented;
};

BothTargets renderBoth(RenderHarness &harness, std::shared_ptr<present::VulkanPresenter> const &presenter,
                       core::RenderWorldDelta delta, core::ivec2 const extent) {
    auto &targets = harness.renderer->targets();
    auto const image = targets.createImage(render::ImageTargetDesc{extent});
    auto const window = targets.createWindow(render::WindowTargetDesc{extent, presenter});
    auto capture = presenter->captureNext();
    auto const result = harness.render(render::FrameDesc{
        std::move(delta),
        {test::viewOf(image, test::presentedCamera(), core::ViewId{0}),
         test::viewOf(window, test::presentedCamera(), core::ViewId{1})}});
    auto both = BothTargets{*result.views.at(0).image, captured(capture)};
    targets.destroy(image);
    targets.destroy(window);
    return both;
}

TEST_CASE("the Vulkan presenter shows the frame an image target gets, through either path") {
    for (auto const shareBuffers : {true, false}) {
        CAPTURE(shareBuffers);
        forEachGpuBackend(shareBuffers, [shareBuffers](RenderHarness &harness,
                                                       std::shared_ptr<present::VulkanPresenter> const &presenter) {
            auto const shared = takesSharedPath(harness, *presenter);
            if (not shareBuffers) {
                CHECK_FALSE(shared);
            }
            auto const both = renderBoth(harness, presenter, test::presentedScene(), kExtent);

            CHECK(both.presented.extent == kExtent);
            CHECK(both.presented.pixels == both.image.pixels);
            presenter->flush();
            auto const stats = presenter->stats();
            CHECK(stats.shown == 1);
            CHECK(stats.sharedShown == (shared ? 1u : 0u));
            auto const path = presenter->deviceName() + (shared ? ": SharedBuffer path" : ": Readback path");
            MESSAGE(path);
        });
    }
}

TEST_CASE("frames handed to the Vulkan presenter back to back are each shown or dropped, the newest shown") {
    forEachGpuBackend(true, [](RenderHarness &harness, std::shared_ptr<present::VulkanPresenter> const &presenter) {
        auto &targets = harness.renderer->targets();
        auto const window = targets.createWindow(render::WindowTargetDesc{kExtent, presenter});
        auto const view = test::viewOf(window, test::presentedCamera(), core::ViewId{0});
        auto senders = std::vector<render::FrameSender>{};
        for (auto frame = 0u; frame < kFrames; ++frame) {
            auto const red = 0.1f + 0.05f * static_cast<float>(frame);
            senders.push_back(harness.renderer->submit(render::FrameDesc{coloredQuad(core::vec3{red, 0.1f, 0.05f}), {view}}));
        }
        for (auto &sender : senders) {
            REQUIRE(lexec::sync_wait(std::move(sender)).has_value());
        }
        presenter->flush();
        auto const stats = presenter->stats();
        CHECK(stats.shown + stats.dropped == kFrames);
        CHECK(stats.shown >= 1);
        targets.destroy(window);

        auto const last = renderBoth(harness, presenter, coloredQuad(core::vec3{0.9f, 0.1f, 0.05f}), kExtent);
        CHECK(last.presented.pixels == last.image.pixels);
    });
}

TEST_CASE("a window target the Vulkan presenter shows can change its extent") {
    forEachGpuBackend(true, [](RenderHarness &harness, std::shared_ptr<present::VulkanPresenter> const &presenter) {
        for (auto const extent : {kExtent, core::ivec2{20, 36}, kExtent}) {
            CAPTURE(extent.x);
            auto const both = renderBoth(harness, presenter, test::presentedScene(), extent);
            CHECK(both.presented.extent == extent);
            CHECK(both.presented.pixels == both.image.pixels);
        }
    });
}

// Throws while recording its pass, which the tone mapping needs.
struct ThrowingFeature final : render::RenderFeature {
    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto &pass = graph.addPass("throw");
        pass.read(view.sceneColor);
        pass.write(view.sceneColor);
        pass.record([](render::PassContext &) { throw std::runtime_error{"deliberately"}; });
    }
};

TEST_CASE("a frame that fails shows nothing, and the Vulkan presenter shows the next") {
    forEachGpuBackend(true, [](RenderHarness &harness, std::shared_ptr<present::VulkanPresenter> const &presenter) {
        auto &targets = harness.renderer->targets();
        auto const window = targets.createWindow(render::WindowTargetDesc{kExtent, presenter});
        auto view = test::viewOf(window, test::presentedCamera(), core::ViewId{0});
        view.pipeline = harness.renderer->pipelines().create(
            render::PipelineDesc{harness.renderer->pipelines().visibility(), {std::make_shared<ThrowingFeature>()}});
        CHECK_THROWS_AS(harness.render(render::FrameDesc{test::presentedScene(), {view}}), std::runtime_error);
        presenter->flush();
        CHECK(presenter->stats().shown == 0);
        targets.destroy(window);

        auto const next = renderBoth(harness, presenter, {}, kExtent);
        CHECK(next.presented.pixels == next.image.pixels);
        presenter->flush();
        CHECK(presenter->stats().shown == 1);
    });
}

} // namespace
