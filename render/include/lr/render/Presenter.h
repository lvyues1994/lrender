#pragma once

#include <lr/render/View.h>

#include <tc/device.hpp>

#include <cstdint>
#include <stdexcept>

namespace lr::render {

// A frame of a window target once the device has finished it: the Readback path, where
// the final pass wrote the pixels into memory the host reads in place.
struct PresentFrame {
    std::uint64_t frameIndex = 0;
    ViewId view;
    RenderTargetId target;
    ImageRgba8View pixels; // valid only during Presenter::present()
};

// One of a presenter's buffers for the SharedBuffer path, for the frame being recorded.
struct SharedBufferSlot {
    std::uint32_t slot = 0;
    // Changes whenever the slot's memory does (for another extent); the renderer then
    // imports it again through Presenter::exportShared().
    std::uint64_t generation = 0;
    // Signalled once the presenter's last use of what the slot held has finished; the
    // device waits for it before writing. Empty when there is nothing to wait for.
    tc::SyncFence release;
};

// A frame written into a presenter's buffer, handed over right after its submission.
struct SharedBufferFrame {
    std::uint64_t frameIndex = 0;
    ViewId view;
    RenderTargetId target;
    std::uint32_t slot = 0;
    tc::SyncFence acquire; // signalled once the device has written the slot
};

// Shows the frames of the window target it belongs to; lr::present has implementations.
// Every call comes from the GPU thread, in frame order, and must not block: work that is
// slow or belongs to another thread goes to that thread.
struct Presenter {
    virtual ~Presenter() = default;

    // The Readback path: a copy of the pixels must be made here to keep them.
    virtual void present(PresentFrame const &frame) = 0;

    // The SharedBuffer path: the final pass writes the frame's packed RGBA8 pixels (as in
    // ImageRgba8) straight into the presenter's memory on the same GPU, and the frame goes
    // to presentShared() with its fence as soon as it is submitted; nothing is read back.
    // Asked once, when the target is created; the renderer takes this path where the
    // presenter wants it and the device exports sync files and imports memory
    // (tc::DeviceInfo::syncFiles, importsMemory). The calls below come only then.
    virtual bool sharesBuffersWith(tc::DeviceInfo const &device) const {
        static_cast<void>(device);
        return false;
    }
    // A slot of `extent` pixels for the frame being recorded: never one acquired and not
    // yet handed back through presentShared() or discardShared().
    virtual SharedBufferSlot acquireShared(core::ivec2 extent) {
        static_cast<void>(extent);
        throw std::logic_error{"lr::render: this presenter shares no buffers"};
    }
    // The slot's memory, for tc::Device::importBuffer: a new descriptor at every call.
    virtual tc::ExternalBufferMemory exportShared(std::uint32_t slot) {
        static_cast<void>(slot);
        throw std::logic_error{"lr::render: this presenter shares no buffers"};
    }
    virtual void presentShared(SharedBufferFrame frame) { static_cast<void>(frame); }
    // The frame that acquired the slot failed, so it holds nothing to show.
    virtual void discardShared(std::uint32_t slot) { static_cast<void>(slot); }
};

} // namespace lr::render
