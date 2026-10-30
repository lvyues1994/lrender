#include <lr/gpu/GpuQueue.h>

#include <tc/cpu/cpu_device.hpp>
#include <tc/devices.hpp>

#include <cassert>
#include <stdexcept>

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
        if (config.runtime == nullptr) {
            throw std::invalid_argument{"lr::gpu: the CPU backend needs a runtime to run kernels on"};
        }
        auto cpu = tc::cpu::CpuDeviceConfig{};
        cpu.executor = std::make_shared<FramePoolExecutor>(*config.runtime);
        return tc::cpu::createDevice(cpu);
    }
    return tc::createDevice(tc::DeviceRequest{config.backend, config.deviceNameContains});
}

struct GpuQueueImpl final : GpuQueue {
    explicit GpuQueueImpl(GpuQueueConfig const &config) {
        auto created = lexec::sync_wait(lexec::schedule(thread.scheduler()) | lexec::then([this, &config] {
                                            device = createDevice(config);
                                            context.device = device.get();
                                            return device->info();
                                        }));
        deviceInfo = std::get<0>(std::move(*created));
    }

    ~GpuQueueImpl() override {
        auto const destroyed =
            lexec::sync_wait(lexec::schedule(thread.scheduler()) | lexec::then([this]() noexcept {
                                 context.device = nullptr;
                                 device.reset();
                             }));
        static_cast<void>(destroyed);
    }

    tc::DeviceInfo const &info() const noexcept override { return deviceInfo; }
    runtime::SerialScheduler scheduler() noexcept override { return thread.scheduler(); }

protected:
    GpuContext &contextOnGpuThread() noexcept override {
        assert(thread.isCurrentThread());
        return context;
    }

private:
    runtime::SerialThread thread;
    // Created, used and destroyed on `thread`.
    std::unique_ptr<tc::Device> device;
    GpuContext context;
    tc::DeviceInfo deviceInfo;
};

} // namespace

std::unique_ptr<GpuQueue> makeGpuQueue(GpuQueueConfig const &config) {
    return std::make_unique<GpuQueueImpl>(config);
}

} // namespace lr::gpu
