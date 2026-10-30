#pragma once

#include <lr/runtime/Runtime.h>
#include <lr/runtime/SerialThread.h>

#include <lexec/execution.hpp>
#include <tc/device.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace lr::gpu {

// The capability to use the device: it exists only on the GPU thread, as the argument of
// a GpuQueue::record() callback.
struct GpuContext {
    tc::Device *device = nullptr;
};

struct GpuQueueConfig {
    std::string backend = "cpu";    // one of tc::availableBackends()
    std::string deviceNameContains; // optional, e.g. "llvmpipe"
    // The CPU backend runs workgroups on its frame pool.
    runtime::Runtime *runtime = nullptr;
};

// The thread that owns a tc::Device: it creates the device, makes every call on it and
// destroys it. OpenGL devices may only be used on the thread that created them.
struct GpuQueue {
    virtual ~GpuQueue() = default;

    virtual tc::DeviceInfo const &info() const noexcept = 0;
    virtual runtime::SerialScheduler scheduler() noexcept = 0;

    // A sender that runs fn(GpuContext &) on the GPU thread and completes with its result;
    // an exception fn throws completes it with set_error.
    template <class Fn>
    auto record(Fn fn) {
        return lexec::schedule(scheduler())
             | lexec::then([this, fn = std::move(fn)]() mutable { return fn(contextOnGpuThread()); });
    }

protected:
    // Calling it on any other thread is a contract violation.
    virtual GpuContext &contextOnGpuThread() noexcept = 0;
};

// Creates the device on the new GPU thread; throws tc::DeviceError when it cannot.
std::unique_ptr<GpuQueue> makeGpuQueue(GpuQueueConfig const &config);

} // namespace lr::gpu
