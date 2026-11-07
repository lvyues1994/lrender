#pragma once

#include "PreparedFrame.h"

#include <lr/render/Presenter.h>

#include <tc/device.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace lr::render {

// A presenter's slot a frame writes, handed back once the frame is submitted.
struct SharedSlot {
    std::shared_ptr<Presenter> presenter;
    ViewId view;
    RenderTargetId target;
    std::uint32_t slot = 0;
};

// The GPU thread's imports of the window targets' shared buffers (the presenters'
// SharedBuffer path), by target and slot: each slot is imported once for each generation
// and extent, and imports a target has not used for `keepFrames` frames go.
struct SharedOutputs {
    explicit SharedOutputs(std::uint64_t const keepFrames_ = 60) noexcept : keepFrames(keepFrames_) {}

    // One of the presenter's slots for a frame of the view, added to `held` as soon as the
    // presenter gives it out: the buffer, as a tc::BufferResource<tc::uint>, with the
    // device ordered after the presenter's last use of the slot.
    std::shared_ptr<void> acquire(tc::Device &device, PreparedView const &view, std::vector<SharedSlot> &held);
    // Once a frame.
    void endFrame();
    // Drops every import, as when the device is lost.
    void clear() noexcept { imports.clear(); }

private:
    struct Import {
        std::uint64_t generation = 0;
        std::size_t count = 0;
        std::shared_ptr<void> buffer;
        std::uint64_t frame = 0; // when a frame last used it
    };

    std::uint64_t keepFrames;
    std::uint64_t frame = 0;
    std::map<std::pair<RenderTargetId, std::uint32_t>, Import> imports;
};

} // namespace lr::render
