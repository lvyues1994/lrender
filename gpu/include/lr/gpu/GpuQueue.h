#pragma once

#include <lr/runtime/Runtime.h>
#include <lr/runtime/SerialThread.h>

#include <lexec/any_sender_of.hpp>
#include <lexec/execution.hpp>
#include <tc/device.hpp>

#include <cstdint>
#include <exception>
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
    // Required: whenComplete() waits on its IO scheduler, and the CPU backend runs
    // workgroups on its frame pool.
    runtime::Runtime *runtime = nullptr;
};

// Completes on the IO scheduler once the device has finished a submission.
using CompletionSender =
    lexec::any_sender_of<lexec::set_value_t(), lexec::set_error_t(std::exception_ptr), lexec::set_stopped_t()>;

// The thread that owns a tc::Device: it creates the device, makes every call on it and
// destroys it. OpenGL devices may only be used on the thread that created them.
struct GpuQueue {
    virtual ~GpuQueue() = default;

    // The first device's; a recreated device comes from the same config.
    virtual tc::DeviceInfo const &info() const noexcept = 0;
    virtual runtime::SerialScheduler scheduler() noexcept = 0;

    // A sender that runs fn(GpuContext &) on the GPU thread and completes with its result;
    // an exception fn throws completes it with set_error.
    template <class Fn>
    auto record(Fn fn) {
        return lexec::schedule(scheduler())
             | lexec::then([this, fn = std::move(fn)]() mutable { return fn(contextOnGpuThread()); });
    }

    // On the GPU thread: sends the work recorded so far to the device without waiting for
    // it, with a sync file where DeviceInfo::syncFiles is set and an empty fence elsewhere.
    virtual tc::FencedSubmission submit(GpuContext &context) = 0;

    // Any thread: a task on the IO scheduler that waits for the device to finish
    // `submitted` (on the poller for a sync file, on the blocking pool for a Submission
    // that is not complete yet) and completes there, so the GPU thread never waits.
    virtual CompletionSender whenComplete(tc::FencedSubmission submitted) = 0;

    // On the GPU thread: destroys the device, which waits for its queued work unless it
    // is lost, and creates another from the same config. Throws tc::DeviceError when it
    // cannot; context.device is then null until a later call succeeds.
    virtual void recreateDevice(GpuContext &context) = 0;

protected:
    // Calling it on any other thread is a contract violation.
    virtual GpuContext &contextOnGpuThread() noexcept = 0;
};

// Creates the device on the new GPU thread; throws tc::DeviceError when it cannot.
std::unique_ptr<GpuQueue> makeGpuQueue(GpuQueueConfig const &config);

} // namespace lr::gpu
