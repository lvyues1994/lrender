#include <lr/gpu/GpuQueue.h>
#include <lr/runtime/Runtime.h>

#include <lexec/execution.hpp>
#include <tc/device.hpp>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>

#include <sys/eventfd.h>
#include <unistd.h>

namespace {

using namespace lr;

std::unique_ptr<runtime::Runtime> smallRuntime() {
    auto config = runtime::RuntimeConfig{};
    config.framePoolThreads = 2;
    return runtime::makeRuntime(config);
}

std::unique_ptr<gpu::GpuQueue> cpuQueue(runtime::Runtime &runtime) {
    auto config = gpu::GpuQueueConfig{};
    config.runtime = &runtime;
    return gpu::makeGpuQueue(config);
}

std::thread::id gpuThreadOf(gpu::GpuQueue &queue) {
    auto const id = lexec::sync_wait(queue.record([](gpu::GpuContext &) { return std::this_thread::get_id(); }));
    return std::get<0>(*id);
}

// Completes when opened, like the work of a submission the device is still running.
struct Gate final : tc::Submission::Completion {
    bool isComplete() const override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return open;
    }

    void wait() const override {
        auto lock = std::unique_lock<std::mutex>{mutex};
        opened.wait(lock, [this] { return open; });
        if (fails) {
            throw tc::DeviceError{"the device was lost while the gate was closed"};
        }
    }

    void release() {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            open = true;
        }
        opened.notify_all();
    }

    bool fails = false;

private:
    mutable std::mutex mutex;
    mutable std::condition_variable opened;
    bool open = false;
};

// Opens the gate, or signals the descriptor, a little later from a thread of its own.
template <class Fn>
std::thread later(Fn fn) {
    return std::thread{[fn] {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        fn();
    }};
}

TEST_CASE("submitting on the CPU backend gives a complete submission and an empty fence") {
    auto runtime = smallRuntime();
    auto queue = cpuQueue(*runtime);

    auto submitted = lexec::sync_wait(queue->record([&queue](gpu::GpuContext &context) { return queue->submit(context); }));

    REQUIRE(submitted.has_value());
    auto &fenced = std::get<0>(*submitted);
    CHECK(fenced.submission.isComplete());
    CHECK(fenced.submission.sequence() == 1);
    CHECK(fenced.fence.empty());
    CHECK(lexec::sync_wait(queue->whenComplete(std::move(fenced))).has_value());
}

TEST_CASE("whenComplete waits on the poller until a sync file is readable, then completes on an IO task") {
    auto runtime = smallRuntime();
    auto queue = cpuQueue(*runtime);
    auto const fd = ::eventfd(0, EFD_CLOEXEC);
    REQUIRE(fd >= 0);
    auto signalling = std::atomic<bool>{false};
    auto wrote = std::atomic<bool>{false};
    auto signaller = later([fd, &signalling, &wrote] {
        auto const one = std::uint64_t{1};
        signalling = true;
        wrote = ::write(fd, &one, sizeof one) == static_cast<ssize_t>(sizeof one);
    });

    // An eventfd stands in for a sync file: both become readable once signalled.
    auto const completedOn = lexec::sync_wait(queue->whenComplete(tc::FencedSubmission{tc::Submission{}, tc::SyncFence{fd}})
                                              | lexec::then([&signalling] {
                                                    return std::make_tuple(signalling.load(), std::this_thread::get_id());
                                                }));
    signaller.join();

    CHECK(wrote);
    REQUIRE(completedOn.has_value());
    auto const [afterSignal, thread] = std::get<0>(*completedOn);
    CHECK(afterSignal);
    auto const onIoTask = thread != std::this_thread::get_id() and thread != gpuThreadOf(*queue);
    CHECK(onIoTask);
}

TEST_CASE("whenComplete waits on the blocking pool for a submission without a fence") {
    auto runtime = smallRuntime();
    auto queue = cpuQueue(*runtime);
    auto const gate = std::make_shared<Gate>();
    auto released = std::atomic<bool>{false};
    auto opener = later([gate, &released] {
        released = true;
        gate->release();
    });

    auto const completed = lexec::sync_wait(
        queue->whenComplete(tc::FencedSubmission{tc::Submission{gate}, tc::SyncFence{}})
        | lexec::then([&released] { return released.load(); }));
    opener.join();

    REQUIRE(completed.has_value());
    CHECK(std::get<0>(*completed));
}

TEST_CASE("a wait that fails completes whenComplete with the device's error") {
    auto runtime = smallRuntime();
    auto queue = cpuQueue(*runtime);
    auto const gate = std::make_shared<Gate>();
    gate->fails = true;
    auto opener = later([gate] { gate->release(); });

    CHECK_THROWS_AS(lexec::sync_wait(queue->whenComplete(tc::FencedSubmission{tc::Submission{gate}, tc::SyncFence{}})),
                    tc::DeviceError);
    opener.join();
}

TEST_CASE("recreating the device gives the context a working device made from the same config") {
    auto runtime = smallRuntime();
    auto queue = cpuQueue(*runtime);

    auto const recreated = lexec::sync_wait(queue->record([&queue](gpu::GpuContext &context) {
        queue->recreateDevice(context);
        auto const name = context.device->info().deviceName;
        return std::make_tuple(name, queue->submit(context).submission.sequence());
    }));

    REQUIRE(recreated.has_value());
    auto const &[name, sequence] = std::get<0>(*recreated);
    CHECK(name == queue->info().deviceName);
    CHECK(sequence == 1); // a new device counts its submissions from 1 again
}

TEST_CASE("a GPU queue needs a runtime") {
    CHECK_THROWS_AS(gpu::makeGpuQueue(gpu::GpuQueueConfig{}), std::invalid_argument);
}

} // namespace
