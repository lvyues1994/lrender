#include "ScriptTestSupport.h"

#include <co2/co2.hpp>
#include <lexec/execution.hpp>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace lr;
using namespace std::chrono_literals;
using scene::nextFrame;
using scene::ScriptTask;
using scene::SimDuration;
using scene::waitFor;
using test::milliseconds;

struct Resumed {
    char script;
    std::uint64_t tick;

    friend bool operator==(Resumed const &a, Resumed const &b) noexcept {
        return a.script == b.script and a.tick == b.tick;
    }
};

using Log = std::vector<Resumed>;

auto countFrames(scene::ScriptHost &host, Log &log, char const name, int const frames)
    CO2_BEG(ScriptTask, (host, log, name, frames), int frame{};) {
    for (frame = 0; frame < frames; ++frame) {
        log.push_back(Resumed{name, host.tickCount()});
        CO2_AWAIT(nextFrame());
    }
}
CO2_END

TEST_CASE("nextFrame resumes a script once per tick, scripts in the order they started") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto log = Log{};
    auto const first = host->start(countFrames(*host, log, 'a', 3));
    auto const second = host->start(countFrames(*host, log, 'b', 2));
    CHECK(log.empty());

    for (auto tick = 0; tick < 3; ++tick) {
        host->tick(milliseconds(16));
    }
    CHECK(log == Log{{'a', 1}, {'b', 1}, {'a', 2}, {'b', 2}, {'a', 3}});
    CHECK(host->isRunning(first));
    CHECK(not host->isRunning(second));

    host->tick(milliseconds(16));
    CHECK(log.size() == 5);
    CHECK(host->isIdle());
}

struct TimedResumption {
    std::uint64_t tick;
    SimDuration elapsed;

    friend bool operator==(TimedResumption const &a, TimedResumption const &b) noexcept {
        return a.tick == b.tick and a.elapsed == b.elapsed;
    }
};

auto waitTwice(scene::ScriptHost &host, std::vector<TimedResumption> &log)
    CO2_BEG(ScriptTask, (host, log), SimDuration elapsed{};) {
    CO2_AWAIT_SET(elapsed, waitFor(milliseconds(25)));
    log.push_back(TimedResumption{host.tickCount(), elapsed});
    CO2_AWAIT_SET(elapsed, waitFor(milliseconds(30)));
    log.push_back(TimedResumption{host.tickCount(), elapsed});
}
CO2_END

TEST_CASE("waitFor resumes in the first tick whose simulation time reaches it") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto log = std::vector<TimedResumption>{};
    host->start(waitTwice(*host, log));

    // Ticks of 10 ms: the script starts at 10 ms and waits until 35 ms, so until the tick
    // at 40 ms, then until 70 ms, which a tick reaches exactly.
    for (auto tick = 0; tick < 8; ++tick) {
        host->tick(milliseconds(10));
    }

    CHECK(log == std::vector<TimedResumption>{{4, milliseconds(30)}, {7, milliseconds(30)}});
    CHECK(host->isIdle());
}

struct ThreadReport {
    std::atomic<bool> completed{false};
    bool resumed = false;
    bool resumedInsideTick = false;
    std::thread::id resumedOn;
    std::thread::id completedOn;
};

// The pool thread's id, from that thread.
auto poolThread(lexec::static_thread_pool &pool, ThreadReport &report) {
    return lexec::schedule(pool.get_scheduler()) | lexec::then([&report] {
               report.completed = true;
               return std::this_thread::get_id();
           });
}

auto awaitPool(lexec::static_thread_pool &pool, bool const &insideTick, ThreadReport &report)
    CO2_BEG(ScriptTask, (pool, insideTick, report)) {
    CO2_AWAIT_SET(report.completedOn, poolThread(pool, report));
    report.resumed = true;
    report.resumedInsideTick = insideTick;
    report.resumedOn = std::this_thread::get_id();
}
CO2_END

TEST_CASE("a sender completing on another thread resumes its script inside a tick, on the host thread") {
    auto pool = lexec::static_thread_pool{1};
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto report = ThreadReport{};
    auto insideTick = false;
    host->start(awaitPool(pool, insideTick, report));
    auto const tick = [&] {
        insideTick = true;
        host->tick(milliseconds(1));
        insideTick = false;
    };

    tick();
    auto const deadline = std::chrono::steady_clock::now() + 5s;
    while (not report.completed and std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    std::this_thread::sleep_for(20ms);
    REQUIRE(report.completed);
    CHECK(not report.resumed);

    while (not report.resumed and std::chrono::steady_clock::now() < deadline) {
        tick();
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(report.resumed);
    CHECK(report.resumedInsideTick);
    auto const completedElsewhere = report.completedOn != std::this_thread::get_id();
    auto const resumedHere = report.resumedOn == std::this_thread::get_id();
    CHECK(completedElsewhere);
    CHECK(resumedHere);
    CHECK(host->isIdle());
}

auto awaitForever(std::atomic<int> &stopped, bool &reachedEnd) CO2_BEG(ScriptTask, (stopped, reachedEnd)) {
    CO2_AWAIT(test::NeverCompletes{&stopped});
    reachedEnd = true;
}
CO2_END

TEST_CASE("a script asked to stop while awaiting a stoppable sender ends at the next tick") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto stopped = std::atomic<int>{0};
    auto reachedEnd = false;
    auto const script = host->start(awaitForever(stopped, reachedEnd));
    host->tick(milliseconds(1));
    host->tick(milliseconds(1));
    REQUIRE(host->isRunning(script));

    host->requestStop(script);
    CHECK(stopped == 1);
    CHECK(host->isRunning(script));

    host->tick(milliseconds(1));
    CHECK(not host->isRunning(script));
    CHECK(host->isIdle());
    CHECK(not reachedEnd);
    CHECK(host->takeFailures().empty());
}

auto loopFrames(int &frames) CO2_BEG(ScriptTask, (frames)) {
    for (;;) {
        ++frames;
        CO2_AWAIT(nextFrame());
    }
}
CO2_END

auto waitAnHour(bool &woke) CO2_BEG(ScriptTask, (woke)) {
    CO2_AWAIT(waitFor(std::chrono::hours{1}));
    woke = true;
}
CO2_END

TEST_CASE("a script asked to stop in nextFrame or waitFor ends at the next tick") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto frames = 0;
    auto woke = false;
    auto const looping = host->start(loopFrames(frames));
    auto const waiting = host->start(waitAnHour(woke));
    host->tick(milliseconds(1));
    host->tick(milliseconds(1));
    REQUIRE(frames == 2);

    host->requestStop(looping);
    host->requestStop(waiting);
    host->tick(milliseconds(1));

    CHECK(frames == 2);
    CHECK(not woke);
    CHECK(not host->isRunning(looping));
    CHECK(not host->isRunning(waiting));
    CHECK(host->isIdle());
    CHECK(host->takeFailures().empty());
}

auto failAfterAFrame() CO2_BEG(ScriptTask, ()) {
    CO2_AWAIT(nextFrame());
    throw std::runtime_error{"the script failed"};
}
CO2_END

auto failingSender() {
    return lexec::just() | lexec::then([] { throw std::logic_error{"the sender failed"}; });
}

auto awaitFailingSender() CO2_BEG(ScriptTask, ()) {
    CO2_AWAIT(failingSender());
}
CO2_END

TEST_CASE("a script ended by an exception is a failure") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto const throwing = host->start(failAfterAFrame());
    auto const failingSender = host->start(awaitFailingSender());
    host->tick(milliseconds(1));
    CHECK(host->takeFailures().empty());

    host->tick(milliseconds(1));
    auto const failures = host->takeFailures();

    // In the order they were awaited: the wait, then the sender.
    REQUIRE(failures.size() == 2);
    CHECK(failures[0].script == throwing);
    CHECK_THROWS_AS(std::rethrow_exception(failures[0].error), std::runtime_error);
    CHECK(failures[1].script == failingSender);
    CHECK_THROWS_AS(std::rethrow_exception(failures[1].error), std::logic_error);
    CHECK(host->takeFailures().empty());
    CHECK(host->isIdle());
}

auto startChild(scene::ScriptHost &host, Log &log) CO2_BEG(ScriptTask, (host, log)) {
    log.push_back(Resumed{'p', host.tickCount()});
    host.start(countFrames(host, log, 'c', 1));
    CO2_AWAIT(nextFrame());
    log.push_back(Resumed{'p', host.tickCount()});
}
CO2_END

TEST_CASE("a script started by a script runs in the same tick") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto log = Log{};
    host->start(startChild(*host, log));

    host->tick(milliseconds(1));
    CHECK(log == Log{{'p', 1}, {'c', 1}});
    host->tick(milliseconds(1));
    CHECK(log == Log{{'p', 1}, {'c', 1}, {'p', 2}});
    CHECK(host->isIdle());
}

struct SlowWork {
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
};

// Takes 50 ms on a pool thread once it has started, stop requested or not.
auto slowly(lexec::static_thread_pool &pool, SlowWork &work) {
    return lexec::schedule(pool.get_scheduler()) | lexec::then([&work] {
               work.started = true;
               std::this_thread::sleep_for(50ms);
               work.finished = true;
           });
}

auto awaitSlowly(lexec::static_thread_pool &pool, SlowWork &work) CO2_BEG(ScriptTask, (pool, work)) {
    CO2_AWAIT(slowly(pool, work));
}
CO2_END

TEST_CASE("destroying the host stops its scripts and waits for the senders they await") {
    auto pool = lexec::static_thread_pool{1};
    auto stopped = std::atomic<int>{0};
    auto reachedEnd = false;
    auto frames = 0;
    auto woke = false;
    auto work = SlowWork{};
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    host->start(awaitForever(stopped, reachedEnd));
    host->start(loopFrames(frames));
    host->start(waitAnHour(woke));
    host->start(awaitSlowly(pool, work));
    host->tick(milliseconds(1));
    // Started and asked to stop before it ever ran: it runs up to its first await.
    host->start(loopFrames(frames));
    // The pool drops work stopped before it starts; this work is past that point.
    auto const deadline = std::chrono::steady_clock::now() + 5s;
    while (not work.started and std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(work.started);

    host.reset();

    CHECK(stopped == 1);
    CHECK(work.finished);
    CHECK(frames == 2);
    CHECK(not woke);
    CHECK(not reachedEnd);
}

TEST_CASE("a script never started is destroyed with its task, and none starts after shutdown") {
    auto frames = 0;
    { auto const unused = loopFrames(frames); }
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    host->shutdown();

    auto const late = host->start(loopFrames(frames));
    host->tick(milliseconds(1));

    CHECK(not host->isRunning(late));
    CHECK(host->isIdle());
    CHECK(frames == 0);
}

TEST_CASE("simulation time cannot go back") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    CHECK_THROWS_AS(host->tick(milliseconds(-1)), std::invalid_argument);
    auto config = scene::ScriptHostConfig{};
    config.release = scene::ReleaseMode::Deterministic;
    config.deterministicDelay = 0;
    CHECK_THROWS_AS(scene::makeScriptHost(config), std::invalid_argument);
}

} // namespace
