#pragma once

// A script is a co2 coroutine returning ScriptTask, run by a ScriptHost:
//
//   auto blink(World &world, EntityId const lamp) CO2_BEG(ScriptTask, (world, lamp)) {
//       for (;;) {
//           world.setLight(lamp, bright);
//           CO2_AWAIT(waitFor(std::chrono::milliseconds{500}));
//           world.setLight(lamp, dim);
//           CO2_AWAIT(waitFor(std::chrono::milliseconds{500}));
//       }
//   }
//   CO2_END
//
// It awaits lexec senders, nextFrame(), waitFor() and co2::getStopToken(), and resumes only
// inside ScriptHost::tick() on the thread calling it, so it may use the World freely.

#include <lr/core/Id.h>

#include <co2/coroutine.hpp>
#include <co2/env.hpp>
#include <co2/scheduler.hpp>
#include <co2/stop_token.hpp>
#include <lexec/algorithms/continues_on.hpp>
#include <lexec/coro/co2.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <optional>
#include <type_traits>
#include <utility>

namespace lr::scene {

using ScriptId = core::Id<struct ScriptTag>;

// Simulation time, which ScriptHost::tick() advances: whole nanoseconds, so sums are exact.
using SimDuration = std::chrono::nanoseconds;

struct NextFrame {};

struct WaitFor {
    SimDuration duration{0};
};

// Resumes in the next tick; CO2_AWAIT_SET(elapsed, nextFrame()) also gives the simulation
// time that passed meanwhile.
inline NextFrame nextFrame() noexcept { return NextFrame{}; }
// Resumes in the first tick whose simulation time reaches `duration` from now, giving
// the simulation time that passed.
inline WaitFor waitFor(SimDuration const duration) noexcept { return WaitFor{duration}; }

struct ScriptTask;

namespace detail {

struct ScriptPromise;

struct OperationGate;

// A script waiting for simulation time.
struct TimedWait {
    ScriptPromise *script = nullptr;
    SimDuration deadline{0};
    std::uint64_t sequence = 0; // the host's
    co2::coroutine_handle<> continuation;
};

// The host side of the scripts, which ScriptHost's implementation provides. All but
// operationCompleted() are called on the host thread, inside a tick.
struct ScriptContext {
    virtual ~ScriptContext() = default;

    virtual SimDuration now() const noexcept = 0;
    virtual void awaitTime(TimedWait &wait) = 0;
    // Before the operation starts.
    virtual void awaitOperation(OperationGate &gate) = 0;
    // When it could not start after all.
    virtual void abandonOperation(OperationGate &gate) noexcept = 0;
    // Any thread: the operation completed, `continuation` replays its completion.
    virtual void operationCompleted(OperationGate &gate, co2::coroutine_handle<> continuation) noexcept = 0;
    // At the script's final suspend point.
    virtual void scriptEnded(ScriptPromise &script) noexcept = 0;
};

// What a ScriptHost releases an awaited sender's completion through: continues_on() hops
// to this co2 scheduler, which holds the continuation until a tick hands it to the host's
// co2::ManualExecutor.
struct OperationGate final : co2::Scheduler {
    explicit OperationGate(ScriptContext &context_) noexcept : context(&context_) {}

    void schedule(co2::coroutine_handle<> const completion) noexcept override {
        context->operationCompleted(*this, completion);
    }

    ScriptContext *context;
    // The host's, under its lock.
    std::uint64_t sequence = 0;
    std::uint64_t awaitedTick = 0;
    co2::coroutine_handle<> continuation;
};

// Awaits a sender as continues_on(sender, the gate). co2 moves an awaiter only before it
// suspends, so the gate and lexec's awaiter are made in await_suspend, where they stay.
template <class Sndr>
struct OperationAwaiter {
    using Gated = decltype(lexec::continues_on(std::declval<Sndr>(), std::declval<lexec::coro::scheduler>()));
    using Awaiter = decltype(lexec::coro::as_awaitable(std::declval<Gated>()));

    OperationAwaiter(ScriptContext &context_, Sndr sender_) : context(&context_), sender(std::move(sender_)) {}

    OperationAwaiter(OperationAwaiter &&other) noexcept(std::is_nothrow_move_constructible_v<Sndr>)
        : context(other.context), sender(std::move(other.sender)) {}

    OperationAwaiter &operator=(OperationAwaiter &&) = delete;

    bool await_ready() const noexcept { return false; }

    bool await_suspend(co2::coroutine_handle<ScriptPromise> const script) {
        auto &opened = gate.emplace(*context);
        awaiter.emplace(
            lexec::coro::as_awaitable(lexec::continues_on(std::move(*sender), lexec::coro::scheduler{opened})));
        context->awaitOperation(opened);
        auto suspended = false;
        try {
            suspended = awaiter->await_suspend(script);
        } catch (...) {
            context->abandonOperation(opened);
            throw;
        }
        if (not suspended) {
            context->abandonOperation(opened);
        }
        return suspended;
    }

    decltype(auto) await_resume() { return awaiter->await_resume(); }

private:
    ScriptContext *context;
    std::optional<Sndr> sender;
    std::optional<OperationGate> gate;
    std::optional<Awaiter> awaiter; // last: it refers to the gate
};

struct TimedAwaiter {
    bool await_ready() const noexcept { return false; }
    void await_suspend(co2::coroutine_handle<> continuation);
    // Throws lexec::coro::stopped_error when the script was asked to stop.
    SimDuration await_resume() const;

    TimedWait wait;
    SimDuration delay{0};
    SimDuration awaitedAt{0};
};

struct StopTokenReader {
    bool await_ready() const noexcept { return true; }
    void await_suspend(co2::coroutine_handle<>) const noexcept {}
    co2::stop_token await_resume() const noexcept { return token; }

    co2::stop_token token;
};

template <class>
inline constexpr bool kAlwaysFalse = false;

template <class Awaitable>
struct UnsupportedAwaitable {
    static_assert(kAlwaysFalse<Awaitable>,
                  "lr::scene: a script awaits lexec senders, nextFrame(), waitFor() and co2::getStopToken() only, "
                  "so that it resumes inside ScriptHost::tick()");
};

struct EndOfScript {
    bool await_ready() const noexcept { return false; }
    void await_suspend(co2::coroutine_handle<ScriptPromise> script) const noexcept;
    void await_resume() const noexcept {}
};

struct ScriptPromise {
    ScriptTask get_return_object() noexcept;
    co2::suspend_always initial_suspend() const noexcept { return {}; }
    EndOfScript final_suspend() const noexcept { return {}; }
    void return_void() const noexcept {}
    void unhandled_exception() noexcept { error = std::current_exception(); }
    co2::stop_token get_stop_token() const noexcept { return stopSource.get_token(); }

    TimedAwaiter await_transform(NextFrame) noexcept { return waitingFor(SimDuration{0}); }
    TimedAwaiter await_transform(WaitFor const wait) noexcept { return waitingFor(wait.duration); }
    StopTokenReader await_transform(co2::GetStopToken) const noexcept { return StopTokenReader{get_stop_token()}; }

    template <class Sndr, std::enable_if_t<lexec::is_sender_v<Sndr>, int> = 0>
    OperationAwaiter<std::decay_t<Sndr>> await_transform(Sndr &&sender) {
        return OperationAwaiter<std::decay_t<Sndr>>{*context, std::forward<Sndr>(sender)};
    }

    template <class Awaitable, std::enable_if_t<not lexec::is_sender_v<Awaitable>, int> = 0>
    UnsupportedAwaitable<std::decay_t<Awaitable>> await_transform(Awaitable &&);

    TimedAwaiter waitingFor(SimDuration const delay) noexcept {
        auto awaiter = TimedAwaiter{};
        awaiter.wait.script = this;
        awaiter.delay = delay;
        return awaiter;
    }

    // ScriptHost::start() sets these.
    ScriptContext *context = nullptr;
    ScriptId id;
    co2::stop_source stopSource;
    std::exception_ptr error; // what ended the script, if anything did
};

struct ScriptTaskAccess;

} // namespace detail

// A script that has not been started yet: destroying it destroys the coroutine.
// ScriptHost::start() takes it over.
struct ScriptTask {
    using promise_type = detail::ScriptPromise;

    ScriptTask(ScriptTask &&other) noexcept : handle(std::exchange(other.handle, nullptr)) {}

    ScriptTask &operator=(ScriptTask &&other) noexcept {
        if (this != &other) {
            reset();
            handle = std::exchange(other.handle, nullptr);
        }
        return *this;
    }

    ScriptTask(ScriptTask const &) = delete;
    ScriptTask &operator=(ScriptTask const &) = delete;
    ~ScriptTask() { reset(); }

private:
    friend struct detail::ScriptPromise;
    friend struct detail::ScriptTaskAccess;

    explicit ScriptTask(co2::coroutine_handle<promise_type> const handle_) noexcept : handle(handle_) {}

    void reset() noexcept {
        if (handle) {
            handle.destroy();
            handle = nullptr;
        }
    }

    co2::coroutine_handle<promise_type> handle;
};

namespace detail {

inline ScriptTask ScriptPromise::get_return_object() noexcept {
    return ScriptTask{co2::coroutine_handle<ScriptPromise>::from_promise(*this)};
}

inline void EndOfScript::await_suspend(co2::coroutine_handle<ScriptPromise> const script) const noexcept {
    script.promise().context->scriptEnded(script.promise());
}

struct ScriptTaskAccess {
    static co2::coroutine_handle<ScriptPromise> handleOf(ScriptTask const &task) noexcept { return task.handle; }
    static void release(ScriptTask &task) noexcept { task.handle = nullptr; }
};

} // namespace detail

} // namespace lr::scene
