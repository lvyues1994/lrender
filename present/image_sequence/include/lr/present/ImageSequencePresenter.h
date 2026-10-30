#pragma once

#include <lr/render/Presenter.h>
#include <lr/runtime/Runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace lr::present {

struct ImageSequenceConfig {
    std::filesystem::path directory; // must exist
    std::string prefix = "frame";
    runtime::Runtime *runtime = nullptr; // the files are written on its IO scheduler
};

struct ImageSequenceProgress {
    std::uint64_t written = 0;
    std::uint64_t failed = 0;
};

// Writes each frame to <directory>/<prefix><frame index>.ppm (see writePpm). present()
// copies the pixels and returns; an IO task writes the file on the blocking pool.
// The last reference waits for the files still being written, so keep one until the
// renderer is gone, off the GPU thread.
struct ImageSequencePresenter : render::Presenter {
    // Waits until every frame presented so far is written or has failed to be.
    virtual ImageSequenceProgress flush() = 0;
};

std::shared_ptr<ImageSequencePresenter> makeImageSequencePresenter(ImageSequenceConfig const &config);

} // namespace lr::present
