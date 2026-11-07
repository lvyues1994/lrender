#include "SharedOutputs.h"

#include <iterator>

namespace lr::render {

std::shared_ptr<void> SharedOutputs::acquire(tc::Device &device, PreparedView const &view,
                                             std::vector<SharedSlot> &held) {
    auto const &output = *view.sharedOutput;
    auto &presenter = *output.presenter;
    auto slot = presenter.acquireShared(output.extent);
    held.push_back(SharedSlot{output.presenter, view.view, view.target, slot.slot});
    auto const count = static_cast<std::size_t>(output.extent.x) * static_cast<std::size_t>(output.extent.y);
    auto &import = imports[{view.target, slot.slot}];
    if (import.buffer == nullptr or import.generation != slot.generation or import.count != count) {
        auto const buffer = device.importBuffer<tc::uint>(presenter.exportShared(slot.slot), count);
        import = Import{slot.generation, count,
                        std::shared_ptr<tc::BufferResource<tc::uint>>{buffer.shared(), buffer.hostView()}, frame};
    }
    import.frame = frame;
    device.waitFor(std::move(slot.release));
    return import.buffer;
}

void SharedOutputs::endFrame() {
    ++frame;
    for (auto entry = imports.begin(); entry != imports.end();) {
        entry = entry->second.frame + keepFrames < frame ? imports.erase(entry) : std::next(entry);
    }
}

} // namespace lr::render
