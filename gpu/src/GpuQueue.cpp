#include <lr/gpu/GpuQueue.h>

#include <stackfull/exec/spawn.hpp>
#include <stackfull/io/Registration.h>
#include <stackfull/runtime/Runtime.h>
#include <tc/cpu/cpu_device.hpp>
#include <tc/devices.hpp>

#include <cassert>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>

namespace lr::gpu {

namespace {

// tc's CPU backend running workgroups on the frame pool. bulk() is called from the GPU
// thread, which is not one of the pool's workers, so blocking it until the pool is done
// cannot deadlock the pool.
struct FramePoolExecutor final : tc::cpu::ParallelExecutor {
    explicit FramePoolExecutor(runtime::Runtime &runtime_)
        : pool(runtime_.framePool()), threads(runtime_.framePoolThreads()) {}

    std::uint32_t concurrency() const override { return threads; }

    void bulk(std::uint32_t const count, tc::cpu::TaskRef const task) override {
        auto const done = lexec::sync_wait(lexec::schedule(pool)
                                           | lexec::bulk(lexec::par, count, [task](std::uint32_t const index) noexcept {
                                                 task(index);
                                             }));
        assert(done.has_value());
        static_cast<void>(done);
    }

private:
    runtime::ComputeScheduler pool;
    std::uint32_t threads;
};

std::unique_ptr<tc::Device> createDevice(GpuQueueConfig const &config) {
    if (config.backend == "cpu") {
        auto cpu = tc::cpu::CpuDeviceConfig{};
        cpu.executor = std::make_shared<FramePoolExecutor>(*config.runtime);
        return tc::cpu::createDevice(cpu);
    }
    return tc::createDevice(tc::DeviceRequest{config.backend, config.deviceNameContains});
}

bool setNonBlocking(int const fd) noexcept {
    auto const flags = ::fcntl(fd, F_GETFL);
    return flags >= 0 and ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

// Parks the calling IO task until the sync file is readable; false when the poller
// cannot watch it.
bool waitUntilSignaled(stackfull::io::Poller &poller, tc::SyncFence const &fence) {
    if (not setNonBlocking(fence.get())) {
        return false;
    }
    auto registration = stackfull::io::Registration{poller, fence.get()};
    if (registration.error()) {
        return false;
    }
    auto seen = registration.readEvents();
    while (not fence.isSignaled()) {
        if (registration.waitReadable(seen)) {
            return false;
        }
        seen = registration.readEvents();
    }
    return true;
}

// In an IO task: returns once the device has finished `submitted`. The submission's own
// completion may be seen a moment after its sync file signals.
void waitForDevice(stackfull::io::Poller &poller, tc::FencedSubmission const &submitted) {
    if (not submitted.fence.empty() and not waitUntilSignaled(poller, submitted.fence)) {
        stackfull::blocking([&submitted] { submitted.fence.wait(); });
    }
    if (not submitted.submission.isComplete()) {
        stackfull::blocking([&submitted] { submitted.submission.wait(); });
    }
}

std::exception_ptr asException(std::exception_ptr error) noexcept { return error; }

std::exception_ptr asException(std::error_code const error) {
    return std::make_exception_ptr(std::system_error{error, "lr::gpu: cannot wait for the device"});
}

struct GpuQueueImpl final : GpuQueue {
    explicit GpuQueueImpl(GpuQueueConfig const &config_) : config(config_) {
        if (config.runtime == nullptr) {
            throw std::invalid_argument{"lr::gpu: a GPU queue needs a runtime to wait for the device on"};
        }
        auto created = lexec::sync_wait(lexec::schedule(thread.scheduler()) | lexec::then([this] {
                                            device = createDevice(config);
                                            token.device = device.get();
                                            return device->info();
                                        }));
        deviceInfo = std::get<0>(std::move(*created));
    }

    ~GpuQueueImpl() override {
        auto const destroyed =
            lexec::sync_wait(lexec::schedule(thread.scheduler()) | lexec::then([this]() noexcept {
                                 token.device = nullptr;
                                 device.reset();
                             }));
        static_cast<void>(destroyed);
    }

    tc::DeviceInfo const &info() const noexcept override { return deviceInfo; }
    runtime::SerialScheduler scheduler() noexcept override { return thread.scheduler(); }

    tc::FencedSubmission submit(GpuContext &context) override {
        assert(thread.isCurrentThread());
        if (deviceInfo.syncFiles) {
            return context.device->submitWithFence();
        }
        return tc::FencedSubmission{context.device->submit(), tc::SyncFence{}};
    }

    CompletionSender whenComplete(tc::FencedSubmission submitted) override {
        auto &poller = config.runtime->poller();
        return stackfull::exec::spawn(config.runtime->io(),
                                      [&poller, submitted = std::move(submitted)] { waitForDevice(poller, submitted); })
             | lexec::let_error([](auto &error) { return lexec::just_error(asException(error)); });
    }

    void recreateDevice(GpuContext &context) override {
        assert(thread.isCurrentThread());
        context.device = nullptr;
        device.reset();
        device = createDevice(config);
        context.device = device.get();
    }

protected:
    GpuContext &contextOnGpuThread() noexcept override {
        assert(thread.isCurrentThread());
        return token;
    }

private:
    GpuQueueConfig config;
    runtime::SerialThread thread;
    // Created, used and destroyed on `thread`.
    std::unique_ptr<tc::Device> device;
    GpuContext token;
    tc::DeviceInfo deviceInfo;
};

} // namespace

std::unique_ptr<GpuQueue> makeGpuQueue(GpuQueueConfig const &config) {
    return std::make_unique<GpuQueueImpl>(config);
}

} // namespace lr::gpu
