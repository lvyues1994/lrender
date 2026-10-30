#pragma once

#include <lexec/schedulers/static_thread_pool.hpp>
#include <stackfull/exec/scheduler.hpp>
#include <stackfull/io/Poller.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace lr::runtime {

// A lexec static_thread_pool's scheduler: schedule() and a parallel bulk().
using ComputeScheduler = decltype(std::declval<lexec::static_thread_pool &>().get_scheduler());

// The IO scheduler as a lexec scheduler; stackfull::exec::spawn(io, fn) runs fn as a task.
using IoScheduler = stackfull::exec::scheduler;

struct RuntimeConfig {
    // 0: one per hardware thread, less one each for the host thread, the GPU thread and
    // the background pool, and at least 1.
    std::uint32_t framePoolThreads = 0;
    std::uint32_t backgroundPoolThreads = 1;
    std::uint32_t ioWorkers = 2;
};

// The execution resources several modules share: the frame pool for work a frame waits
// on, the background pool for long work (decoding, BLAS builds) that must not hold the
// frame pool up, and the IO scheduler, whose idle workers sleep in poller().
//
// Destroying it stops the IO scheduler first, while the pools its tasks may be waiting
// on are still running, then joins the background pool and the frame pool.
struct Runtime {
    virtual ~Runtime() = default;

    virtual ComputeScheduler framePool() noexcept = 0;
    virtual std::uint32_t framePoolThreads() const noexcept = 0;
    virtual ComputeScheduler backgroundPool() noexcept = 0;
    virtual IoScheduler io() noexcept = 0;
    // Register descriptors here (stackfull::io::Registration) to wait for them in IO tasks.
    virtual stackfull::io::Poller &poller() noexcept = 0;
};

std::unique_ptr<Runtime> makeRuntime(RuntimeConfig const &config);

} // namespace lr::runtime
