#pragma once

#include <lr/render/Presenter.h>

#include <cstdint>
#include <memory>
#include <optional>

struct SDL_Renderer;

namespace lr::present {

struct SdlPresenterConfig {
    SDL_Renderer *renderer = nullptr; // the host's
};

// A window target's presenter for SDL, whose calls belong on the host's main thread.
// present() copies each frame into a triple-buffer mailbox, so neither side waits for
// the other, and the main thread shows the newest frame with blitLatest(). Frames may
// come from one GPU thread only. Its texture goes with it: drop the last reference on
// the main thread, after the renderer and before the SDL_Renderer.
struct SdlPresenter : render::Presenter {
    // On the main thread: uploads the newest frame presented since the last call into a
    // streaming texture, stretches it over the renderer's output and presents. Returns the
    // frame's index, or nullopt (drawing nothing) when no frame arrived. Throws
    // std::runtime_error when SDL fails.
    virtual std::optional<std::uint64_t> blitLatest() = 0;
};

std::shared_ptr<SdlPresenter> makeSdlPresenter(SdlPresenterConfig const &config);

} // namespace lr::present
