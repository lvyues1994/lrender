#include "PresentTestSupport.h"

#include <lr/present/SdlPresenter.h>
#include <lr/render/Renderer.h>

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace lr;

// SDL on a video driver that shows nothing: offscreen, else dummy. Neither starts when
// SDL was built without them, and then the tests are skipped.
struct HiddenVideo {
    HiddenVideo() {
        for (auto const *const driver : {"offscreen", "dummy"}) {
            ::setenv("SDL_VIDEODRIVER", driver, 1);
            if (SDL_Init(SDL_INIT_VIDEO) == 0) {
                started = true;
                return;
            }
            failure += std::string{driver} + ": " + SDL_GetError() + "; ";
        }
    }
    HiddenVideo(HiddenVideo const &) = delete;
    HiddenVideo &operator=(HiddenVideo const &) = delete;
    ~HiddenVideo() {
        if (started) {
            SDL_Quit();
        }
    }

    bool started = false;
    std::string failure;
};

struct WindowDeleter {
    void operator()(SDL_Window *const window) const noexcept { SDL_DestroyWindow(window); }
};
struct RendererDeleter {
    void operator()(SDL_Renderer *const renderer) const noexcept { SDL_DestroyRenderer(renderer); }
};

// A hidden window and SDL's software renderer for it.
struct HiddenWindow {
    explicit HiddenWindow(core::ivec2 const extent)
        : window(SDL_CreateWindow("lrender test", 0, 0, extent.x, extent.y, SDL_WINDOW_HIDDEN)),
          renderer(window != nullptr ? SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_SOFTWARE) : nullptr) {}

    // What the renderer shows now, as packed RGBA8.
    std::vector<std::uint32_t> readPixels(core::ivec2 const extent) const {
        auto pixels = std::vector<std::uint32_t>(static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y));
        if (SDL_RenderReadPixels(renderer.get(), nullptr, SDL_PIXELFORMAT_ABGR8888, pixels.data(), extent.x * 4) != 0) {
            pixels.clear();
        }
        return pixels;
    }

    std::unique_ptr<SDL_Window, WindowDeleter> window;
    std::unique_ptr<SDL_Renderer, RendererDeleter> renderer;
};

render::PresentFrame frameOf(std::uint64_t const index, render::ImageRgba8 const &image) {
    return render::PresentFrame{index, core::ViewId{0}, core::RenderTargetId{0}, render::imageView(image)};
}

TEST_CASE("an SDL presenter shows the newest frame presented since it last showed one") {
    auto const video = HiddenVideo{};
    if (not video.started) {
        MESSAGE("skipped: SDL has no hidden video driver: " << video.failure);
        return;
    }
    auto const extent = core::ivec2{4, 2};
    auto const window = HiddenWindow{extent};
    REQUIRE(window.renderer != nullptr);
    auto const presenter = present::makeSdlPresenter(present::SdlPresenterConfig{window.renderer.get()});
    auto const older = render::ImageRgba8{extent, std::vector<std::uint32_t>(8, 0xFF0000FFu)};
    auto const newer = render::ImageRgba8{extent, {0xFF0000FFu, 0xFF00FF00u, 0xFFFF0000u, 0xFFFFFFFFu,
                                                   0xFF000000u, 0xFF808080u, 0xFF00FFFFu, 0xFFFF00FFu}};

    CHECK(not presenter->blitLatest().has_value());
    presenter->present(frameOf(5, older));
    presenter->present(frameOf(6, newer));
    CHECK(presenter->blitLatest() == std::optional<std::uint64_t>{6});
    CHECK(window.readPixels(extent) == newer.pixels);
    CHECK(not presenter->blitLatest().has_value());

    presenter->present(frameOf(7, older));
    CHECK(presenter->blitLatest() == std::optional<std::uint64_t>{7});
    CHECK(window.readPixels(extent) == older.pixels);
}

TEST_CASE("an SDL window target shows what the renderer rendered") {
    auto const video = HiddenVideo{};
    if (not video.started) {
        MESSAGE("skipped: SDL has no hidden video driver: " << video.failure);
        return;
    }
    auto const extent = core::ivec2{24, 16};
    auto const window = HiddenWindow{extent};
    REQUIRE(window.renderer != nullptr);
    auto const presenter = present::makeSdlPresenter(present::SdlPresenterConfig{window.renderer.get()});
    {
        auto harness = test::RenderHarness{};
        auto &targets = harness.renderer->targets();
        auto const camera = test::presentedCamera();
        auto const shown = targets.createWindow(render::WindowTargetDesc{extent, presenter});
        auto const image = targets.createImage(render::ImageTargetDesc{extent});

        auto const result = harness.render(render::FrameDesc{
            test::presentedScene(), {test::viewOf(shown, camera, core::ViewId{0}), test::viewOf(image, camera, core::ViewId{1})}});

        CHECK(presenter->blitLatest() == std::optional<std::uint64_t>{result.frameIndex});
        CHECK(window.readPixels(extent) == result.views.at(1).image->pixels);
    }
}

} // namespace
