#include "ScriptTestSupport.h"

#include <co2/co2.hpp>
#include <lexec/execution.hpp>

#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace {

using namespace lr;
using namespace std::chrono_literals;
using scene::nextFrame;
using scene::ScriptTask;
using test::milliseconds;

struct Event {
    std::uint64_t tick;
    char script;
    int value;

    friend bool operator==(Event const &a, Event const &b) noexcept {
        return a.tick == b.tick and a.script == b.script and a.value == b.value;
    }
};

using Timeline = std::vector<Event>;

constexpr auto kLoads = 6;

// Loads one value after another, each taking its own time, and waits a frame after every
// second one.
auto loadInTurn(scene::ScriptHost &host, lexec::static_thread_pool &pool, Timeline &timeline, char const name,
                std::vector<std::chrono::microseconds> const delays)
    CO2_BEG(ScriptTask, (host, pool, timeline, name, delays), std::size_t index{}; int value{};) {
    for (index = 0; index < delays.size(); ++index) {
        CO2_AWAIT_SET(value, test::delayed(pool.get_scheduler(), delays[index], static_cast<int>(index)));
        timeline.push_back(Event{host.tickCount(), name, value});
        if (index % 2 == 1) {
            CO2_AWAIT(nextFrame());
        }
    }
}
CO2_END

// Three scripts loading in turn, the loads taking between 0 and 4 ms as `seed` draws them.
Timeline runLoads(scene::ScriptHostConfig const &config, std::uint32_t const seed) {
    auto pool = lexec::static_thread_pool{3};
    auto random = std::mt19937{seed};
    auto const delays = [&random] {
        auto drawn = std::vector<std::chrono::microseconds>{};
        for (auto load = 0; load < kLoads; ++load) {
            drawn.emplace_back(std::uniform_int_distribution<int>{0, 4000}(random));
        }
        return drawn;
    };
    auto timeline = Timeline{};
    auto host = scene::makeScriptHost(config);
    host->start(loadInTurn(*host, pool, timeline, 'a', delays()));
    host->start(loadInTurn(*host, pool, timeline, 'b', delays()));
    host->start(loadInTurn(*host, pool, timeline, 'c', delays()));
    for (auto tick = 0; tick < 100 and not host->isIdle(); ++tick) {
        host->tick(milliseconds(1));
    }
    CHECK(host->isIdle());
    return timeline;
}

// Every load resumes `delay` ticks after the tick it was awaited in; the scripts go in
// lockstep, in the order they started.
Timeline expectedLoads(std::uint32_t const delay) {
    auto timeline = Timeline{};
    auto tick = std::uint64_t{1};
    for (auto load = 0; load < kLoads; ++load) {
        tick += delay;
        for (auto const script : {'a', 'b', 'c'}) {
            timeline.push_back(Event{tick, script, load});
        }
        if (load % 2 == 1) {
            tick += 1;
        }
    }
    return timeline;
}

scene::ScriptHostConfig deterministic(std::uint32_t const delay) {
    auto config = scene::ScriptHostConfig{};
    config.release = scene::ReleaseMode::Deterministic;
    config.deterministicDelay = delay;
    return config;
}

TEST_CASE("under deterministic release, how long loads take does not change the timeline") {
    for (auto const delay : {1u, 2u}) {
        CAPTURE(delay);
        auto const expected = expectedLoads(delay);
        for (auto const seed : {1u, 2u, 3u, 4u}) {
            CAPTURE(seed);
            CHECK(runLoads(deterministic(delay), seed) == expected);
        }
    }
}

auto awaitTrigger(scene::ScriptHost &host, test::Trigger const trigger, Timeline &timeline, char const name)
    CO2_BEG(ScriptTask, (host, trigger, timeline, name), int value{};) {
    CO2_AWAIT_SET(value, trigger.wait());
    timeline.push_back(Event{host.tickCount(), name, value});
}
CO2_END

TEST_CASE("as soon as ready, a script resumes in the tick after its sender completes") {
    auto host = scene::makeScriptHost(scene::ScriptHostConfig{});
    auto const first = test::Trigger{};
    auto const second = test::Trigger{};
    auto timeline = Timeline{};
    host->start(awaitTrigger(*host, first, timeline, 'a'));
    host->start(awaitTrigger(*host, second, timeline, 'b'));
    host->tick(milliseconds(1));

    second.fire(2);
    host->tick(milliseconds(1));
    host->tick(milliseconds(1));
    host->tick(milliseconds(1));
    first.fire(1);
    host->tick(milliseconds(1));

    // The later sender completed first, so its script went first.
    CHECK(timeline == Timeline{{2, 'b', 2}, {5, 'a', 1}});
    CHECK(host->isIdle());
}

TEST_CASE("deterministic release waits for a late sender and resumes in the order of the awaits") {
    auto host = scene::makeScriptHost(deterministic(1));
    auto const first = test::Trigger{};
    auto const second = test::Trigger{};
    auto timeline = Timeline{};
    host->start(awaitTrigger(*host, first, timeline, 'a'));
    host->start(awaitTrigger(*host, second, timeline, 'b'));
    host->tick(milliseconds(1));

    second.fire(2);
    auto late = std::thread{[first] {
        std::this_thread::sleep_for(30ms);
        first.fire(1);
    }};
    host->tick(milliseconds(1));
    late.join();

    CHECK(timeline == Timeline{{2, 'a', 1}, {2, 'b', 2}});
    CHECK(host->isIdle());
}

} // namespace
