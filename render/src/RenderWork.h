#pragma once

#include <lr/core/Image.h>
#include <lr/runtime/Runtime.h>

#include <cstddef>
#include <functional>
#include <future>
#include <memory>

namespace lr::render {

// Runs work of the render thread's that splits into independent items at once.
struct ParallelFor {
    virtual ~ParallelFor() = default;
    // Calls item(0) to item(count - 1), maybe on several threads, and returns when all
    // have returned; rethrows what one of them threw.
    virtual void run(std::size_t count, std::function<void(std::size_t)> const &item) = 0;
};

// One item after another, on the calling thread.
std::unique_ptr<ParallelFor> makeSerialFor();
// On the frame pool, which none of the render thread's waits block.
std::unique_ptr<ParallelFor> makeFramePoolFor(runtime::Runtime &runtime);

// Runs the texture conversions TextureLayout starts (decoding, resampling) off the render
// thread.
struct ImageConversions {
    virtual ~ImageConversions() = default;
    virtual std::future<core::ImageHandle> start(std::function<core::ImageHandle()> convert) = 0;
};

// At start(), on the calling thread.
std::unique_ptr<ImageConversions> makeImmediateConversions();
// On the background pool; destroying it waits for the conversions it started.
std::unique_ptr<ImageConversions> makeBackgroundConversions(runtime::Runtime &runtime);

} // namespace lr::render
