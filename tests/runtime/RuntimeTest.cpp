#include <lr/runtime/Runtime.h>

#include <lexec/execution.hpp>
#include <stackfull/exec/spawn.hpp>
#include <stackfull/exec/wait.hpp>
#include <stackfull/io/Registration.h>
#include <stackfull/runtime/Runtime.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <tuple>
#include <vector>

#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace {

using lr::runtime::makeRuntime;
using lr::runtime::RuntimeConfig;

RuntimeConfig smallConfig() {
    auto config = RuntimeConfig{};
    config.framePoolThreads = 2;
    return config;
}

// Becomes readable once signalled, like the sync file of a finished GPU submission.
struct EventFd {
    EventFd() : fd(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {}
    EventFd(EventFd const &) = delete;
    EventFd &operator=(EventFd const &) = delete;
    ~EventFd() { ::close(fd); }

    bool signal() const {
        auto const one = std::uint64_t{1};
        return ::write(fd, &one, sizeof one) == static_cast<ssize_t>(sizeof one);
    }

    bool isSignaled() const {
        auto descriptor = pollfd{fd, POLLIN, 0};
        return ::poll(&descriptor, 1, 0) == 1;
    }

    int fd;
};

TEST_CASE("work scheduled on the frame pool runs on a pool thread") {
    auto runtime = makeRuntime(smallConfig());

    auto const result = lexec::sync_wait(lexec::schedule(runtime->framePool())
                                         | lexec::then([] { return std::this_thread::get_id(); }));

    REQUIRE(result.has_value());
    auto const ranElsewhere = std::get<0>(*result) != std::this_thread::get_id();
    CHECK(ranElsewhere);
}

TEST_CASE("bulk on the frame pool visits every index exactly once") {
    auto runtime = makeRuntime(smallConfig());
    auto visits = std::vector<std::atomic<int>>(1000);

    auto const done = lexec::sync_wait(lexec::schedule(runtime->framePool())
                                       | lexec::bulk(lexec::par, visits.size(), [&visits](std::size_t const index) {
                                             visits[index].fetch_add(1, std::memory_order_relaxed);
                                         }));

    REQUIRE(done.has_value());
    auto const wrong = std::count_if(visits.begin(), visits.end(), [](std::atomic<int> const &visit) {
        return visit.load(std::memory_order_relaxed) != 1;
    });
    CHECK(wrong == 0);
}

TEST_CASE("the background pool is a pool of its own") {
    auto runtime = makeRuntime(smallConfig());

    auto const result = lexec::sync_wait(lexec::schedule(runtime->backgroundPool()) | lexec::then([] { return 7; }));

    CHECK(runtime->backgroundPool() != runtime->framePool());
    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 7);
}

TEST_CASE("an IO task waits for a sender on the frame pool and continues with its value") {
    auto runtime = makeRuntime(smallConfig());
    auto const framePool = runtime->framePool();

    auto const result = lexec::sync_wait(stackfull::exec::spawn(runtime->io(), [framePool] {
        return stackfull::exec::await(lexec::schedule(framePool) | lexec::then([] { return 20; })) + 22;
    }));

    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 42);
}

TEST_CASE("an IO task hands blocking calls to the blocking pool") {
    auto runtime = makeRuntime(smallConfig());

    auto const result = lexec::sync_wait(
        stackfull::exec::spawn(runtime->io(), [] { return stackfull::blocking([] { return 7; }); }));

    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 7);
}

TEST_CASE("an IO task waits on the runtime's poller until a descriptor becomes readable") {
    auto runtime = makeRuntime(smallConfig());
    auto const event = EventFd{};
    REQUIRE(event.fd >= 0);
    auto &poller = runtime->poller();

    auto waited = stackfull::exec::spawn(runtime->io(), [&poller, &event] {
        auto registration = stackfull::io::Registration{poller, event.fd};
        if (registration.error()) {
            return false;
        }
        auto seen = registration.readEvents();
        while (not event.isSignaled()) {
            if (registration.waitReadable(seen)) {
                return false;
            }
            seen = registration.readEvents();
        }
        return true;
    });
    auto signalled = std::atomic<bool>{false};
    auto signaller = std::thread{[&event, &signalled] {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        signalled = event.signal();
    }};
    auto const result = lexec::sync_wait(std::move(waited));
    signaller.join();

    CHECK(signalled);
    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result));
}

// An idle IO worker sleeps in the poller; work handed to it from another thread must wake
// it every time, however the hand-off falls against its way to sleep.
TEST_CASE("an idle IO worker wakes for every task handed to it from another thread") {
    constexpr auto kHandOffs = 20000;
    auto config = smallConfig();
    config.ioWorkers = 1;
    auto runtime = makeRuntime(config);
    auto done = std::atomic<int>{0};
    auto driver = std::thread{[&] {
        for (auto handOff = 0; handOff < kHandOffs; ++handOff) {
            static_cast<void>(lexec::sync_wait(lexec::schedule(runtime->io()) | lexec::then([] {})));
            done.store(handOff + 1, std::memory_order_release);
        }
    }};
    auto seen = 0;
    auto progressAt = std::chrono::steady_clock::now();
    while (seen < kHandOffs) {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        auto const now = done.load(std::memory_order_acquire);
        if (now != seen) {
            seen = now;
            progressAt = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - progressAt > std::chrono::seconds{10}) {
            FAIL_CHECK("hand-off " << seen << " never woke the IO worker");
            std::abort(); // the stuck driver thread cannot be joined
        }
    }
    driver.join();
    CHECK(seen == kHandOffs);
}

TEST_CASE("a runtime without threads for the background pool is rejected") {
    auto config = smallConfig();
    config.backgroundPoolThreads = 0;

    CHECK_THROWS_AS(makeRuntime(config), std::invalid_argument);
}

} // namespace
