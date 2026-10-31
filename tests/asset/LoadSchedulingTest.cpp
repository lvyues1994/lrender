#include "AssetTestSupport.h"

#include <lexec/execution.hpp>
#include <lexec/scopes/counting_scope.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <optional>
#include <thread>
#include <vector>

namespace {

using namespace lr;
using core::ColorEncoding;
using test::AssetHarness;
using test::AssetHarnessOptions;
using ImageResult = asset::AssetResult<core::ImageHandle>;

test::Bytes smallPng() { return test::pngOf(core::ivec2{2, 2}, test::solid(core::ivec2{2, 2}, 9, 8, 7)); }

// A source whose reads wait until it is opened.
std::unique_ptr<test::TestSource> closedSource(std::unordered_map<std::string, test::Bytes> files) {
    return std::make_unique<test::TestSource>(std::move(files), false);
}

// Starts `sender` on this thread; its result lands in `result`.
void startInto(asset::LoadSender<core::ImageHandle> sender, std::optional<ImageResult> &result,
               lexec::simple_counting_scope &scope) {
    lexec::spawn(std::move(sender) | lexec::then([&result](ImageResult loaded) noexcept { result = std::move(loaded); }),
                 scope.get_token());
}

// Starts `sender` on a thread of its own with `stop`'s token; true there if it stopped.
std::thread startStoppable(asset::LoadSender<core::ImageHandle> sender, lexec::inplace_stop_source &stop,
                           std::atomic<bool> &stopped) {
    return std::thread{[sender = std::move(sender), &stop, &stopped]() mutable {
        auto const result =
            lexec::sync_wait(lexec::write_env(std::move(sender), lexec::prop{lexec::get_stop_token, stop.get_token()}));
        stopped = not result.has_value();
    }};
}

TEST_CASE("concurrent loads of one image read it once and all get the same handle") {
    constexpr auto kRequests = std::size_t{8};
    auto harness = AssetHarness{closedSource({{"shared.png", smallPng()}})};
    auto &source = harness.testSource();
    auto results = std::vector<std::optional<ImageResult>>(kRequests);
    auto scope = lexec::simple_counting_scope{};

    for (auto &result : results) {
        startInto(harness.library->loadImage("shared.png", ColorEncoding::Srgb), result, scope);
    }
    REQUIRE(source.waitForReads("shared.png", 1));
    source.open();
    static_cast<void>(lexec::sync_wait(scope.join()));

    CHECK(source.readsOf("shared.png") == 1);
    auto const loaded = std::all_of(results.begin(), results.end(), [&results](std::optional<ImageResult> const &result) {
        return result and *result and **result == **results.front();
    });
    CHECK(loaded);
    auto const stats = harness.library->stats();
    CHECK(stats.loads == 1);
    CHECK(stats.sharedRequests == kRequests - 1);
    CHECK(stats.decodes == 1);

    // Nothing is kept once the load is over.
    auto const later = test::loadNow(harness.library->loadImage("shared.png", ColorEncoding::Srgb));
    REQUIRE(later.has_value());
    auto const fresh = *later != **results.front();
    CHECK(fresh);
    CHECK(source.readsOf("shared.png") == 2);
}

TEST_CASE("loads of different images overlap: the IO worker runs another load while one waits to be decoded") {
    auto options = AssetHarnessOptions{};
    options.ioWorkers = 1;
    options.backgroundThreads = 1;
    auto harness = AssetHarness{{{"a.png", smallPng()}, {"b.png", smallPng()}}, options};
    auto &source = harness.testSource();
    auto release = std::promise<void>{};
    auto held = std::atomic<bool>{false};
    auto blocker = std::thread{[&] {
        static_cast<void>(lexec::sync_wait(lexec::schedule(harness.runtimeInstance->backgroundPool())
                                           | lexec::then([&] {
                                                 held = true;
                                                 release.get_future().wait();
                                             })));
    }};
    REQUIRE(test::eventually([&] { return held.load(); }));
    auto a = std::optional<ImageResult>{};
    auto b = std::optional<ImageResult>{};
    auto scope = lexec::simple_counting_scope{};

    startInto(harness.library->loadImage("a.png", ColorEncoding::Srgb), a, scope);
    REQUIRE(source.waitForFinishedReads("a.png", 1));
    // a has read its file and cannot decode it: the pool is held. Only if a parks there
    // can the one IO worker run b's task.
    startInto(harness.library->loadImage("b.png", ColorEncoding::Srgb), b, scope);
    auto const overlapped = source.waitForFinishedReads("b.png", 1);
    auto const decodesMeanwhile = harness.library->stats().decodes;
    release.set_value();
    blocker.join();
    static_cast<void>(lexec::sync_wait(scope.join()));

    CHECK(overlapped);
    CHECK(decodesMeanwhile == 0);
    auto const bothLoaded = a and *a and b and *b;
    CHECK(bothLoaded);
    CHECK(harness.library->stats().decodes == 2);
}

TEST_CASE("a load stopped before decoding ends with set_stopped and never decodes") {
    auto harness = AssetHarness{closedSource({{"a.png", smallPng()}})};
    auto &source = harness.testSource();
    auto stop = lexec::inplace_stop_source{};
    auto stopped = std::atomic<bool>{false};

    auto requester = startStoppable(harness.library->loadImage("a.png", ColorEncoding::Srgb), stop, stopped);
    REQUIRE(source.waitForReads("a.png", 1));
    stop.request_stop();
    // The requester completes at once, while its load is still reading.
    requester.join();
    CHECK(stopped);
    source.open();

    REQUIRE(test::eventually([&] { return harness.library->stats().stoppedLoads == 1; }));
    CHECK(harness.library->stats().decodes == 0);
    auto const again = test::loadNow(harness.library->loadImage("a.png", ColorEncoding::Srgb));
    CHECK(again.has_value());
    CHECK(harness.library->stats().loads == 2);
    CHECK(harness.library->stats().decodes == 1);
}

TEST_CASE("a shared load goes on while any of its requesters still waits") {
    auto harness = AssetHarness{closedSource({{"a.png", smallPng()}})};
    auto &source = harness.testSource();
    auto stop = lexec::inplace_stop_source{};
    auto stopped = std::atomic<bool>{false};
    auto kept = std::optional<ImageResult>{};
    auto scope = lexec::simple_counting_scope{};

    auto leaving = startStoppable(harness.library->loadImage("a.png", ColorEncoding::Srgb), stop, stopped);
    REQUIRE(source.waitForReads("a.png", 1));
    startInto(harness.library->loadImage("a.png", ColorEncoding::Srgb), kept, scope);
    stop.request_stop();
    leaving.join();
    source.open();
    static_cast<void>(lexec::sync_wait(scope.join()));

    CHECK(stopped);
    auto const loaded = kept and *kept;
    CHECK(loaded);
    auto const stats = harness.library->stats();
    CHECK(stats.loads == 1);
    CHECK(stats.sharedRequests == 1);
    CHECK(stats.stoppedLoads == 0);
    CHECK(stats.decodes == 1);
}

TEST_CASE("destroying the library stops its loads, and their requesters complete with set_stopped") {
    auto harness = AssetHarness{closedSource({{"a.png", smallPng()}})};
    auto &source = harness.testSource();
    auto result = std::optional<std::optional<std::tuple<ImageResult>>>{};
    auto requester = std::thread{
        [&result, sender = harness.library->loadImage("a.png", ColorEncoding::Srgb)]() mutable {
            result = lexec::sync_wait(std::move(sender));
        }};
    REQUIRE(source.waitForReads("a.png", 1));

    // The read, held back, ends when the load is asked to stop.
    harness.library.reset();
    requester.join();

    REQUIRE(result.has_value());
    CHECK(not result->has_value());
    CHECK(source.readsOf("a.png") == 1);
}

} // namespace
