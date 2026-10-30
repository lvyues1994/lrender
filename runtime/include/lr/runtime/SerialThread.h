#pragma once

#include <lexec/schedulers/run_loop.hpp>

#include <thread>
#include <utility>

namespace lr::runtime {

using SerialScheduler = decltype(std::declval<lexec::run_loop &>().get_scheduler());

// A thread running a lexec::run_loop: work scheduled on it runs one item at a time, in
// the order it was scheduled. State touched only from this thread needs no locking.
struct SerialThread {
    SerialThread();
    SerialThread(SerialThread const &) = delete;
    SerialThread &operator=(SerialThread const &) = delete;
    // Runs the work already queued, then joins the thread.
    ~SerialThread();

    SerialScheduler scheduler() noexcept { return loop.get_scheduler(); }
    bool isCurrentThread() const noexcept { return std::this_thread::get_id() == thread.get_id(); }

private:
    lexec::run_loop loop;
    std::thread thread;
};

} // namespace lr::runtime
