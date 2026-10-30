#pragma once

#include <lr/render/View.h>

#include <cstdint>

namespace lr::render {

// A frame of a window target once the device has finished it: the Readback path, where
// the final pass wrote the pixels into memory the host reads in place. (A SharedBuffer
// path, rendering into a platform buffer handed over with its acquire fence right after
// submission, is left for Android.)
struct PresentFrame {
    std::uint64_t frameIndex = 0;
    ViewId view;
    RenderTargetId target;
    ImageRgba8View pixels; // valid only during Presenter::present()
};

// Shows the frames of the window targets it belongs to; lr::present has implementations.
struct Presenter {
    virtual ~Presenter() = default;

    // Called on the GPU thread, in frame order. It must not block: work that is slow or
    // belongs to another thread goes to that thread, with a copy of the pixels.
    virtual void present(PresentFrame const &frame) = 0;
};

} // namespace lr::render
