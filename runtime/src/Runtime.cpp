#include <lr/runtime/Runtime.h>

#include <stackfull/sched/Scheduler.h>
#include <stackfull/sched/SchedulerOptions.h>

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace lr::runtime {

namespace {

std::uint32_t defaultFramePoolThreads() {
    auto const hardware = std::thread::hardware_concurrency();
    auto const reserved = 3u; // the host thread, the GPU thread, the background pool
    return hardware > reserved ? hardware - reserved : 1u;
}

RuntimeConfig resolve(RuntimeConfig config) {
    if (config.framePoolThreads == 0) {
        config.framePoolThreads = defaultFramePoolThreads();
    }
    if (config.backgroundPoolThreads == 0 or config.ioWorkers == 0) {
        throw std::invalid_argument{"lr::runtime: the background pool and the IO scheduler need at least one thread"};
    }
    return config;
}

std::unique_ptr<stackfull::sched::Scheduler> makeIoScheduler(RuntimeConfig const &config,
                                                             stackfull::io::Poller &poller) {
    auto options = stackfull::sched::SchedulerOptions{};
    options.workers = config.ioWorkers;
    options.driver = &poller;
    return stackfull::sched::makeScheduler(options);
}

struct RuntimeImpl final : Runtime {
    explicit RuntimeImpl(RuntimeConfig const &config)
        : frameThreads(config.framePoolThreads), frame(config.framePoolThreads),
          background(config.backgroundPoolThreads), ioPoller(stackfull::io::makeDefaultPoller()),
          ioScheduler(makeIoScheduler(config, *ioPoller)) {}

    ComputeScheduler framePool() noexcept override { return frame.get_scheduler(); }
    std::uint32_t framePoolThreads() const noexcept override { return frameThreads; }
    ComputeScheduler backgroundPool() noexcept override { return background.get_scheduler(); }
    IoScheduler io() noexcept override { return IoScheduler{*ioScheduler}; }
    stackfull::io::Poller &poller() noexcept override { return *ioPoller; }

private:
    std::uint32_t frameThreads;
    // Destroyed bottom-up, which is the shutdown order: the IO scheduler, the poller it
    // borrows, then the pools.
    lexec::static_thread_pool frame;
    lexec::static_thread_pool background;
    std::unique_ptr<stackfull::io::Poller> ioPoller;
    std::unique_ptr<stackfull::sched::Scheduler> ioScheduler;
};

} // namespace

std::unique_ptr<Runtime> makeRuntime(RuntimeConfig const &config) {
    return std::make_unique<RuntimeImpl>(resolve(config));
}

} // namespace lr::runtime
