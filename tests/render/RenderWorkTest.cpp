#include "RenderWork.h"

#include <lr/runtime/Runtime.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace lr;

std::unique_ptr<runtime::Runtime> smallRuntime() {
    auto config = runtime::RuntimeConfig{};
    config.framePoolThreads = 3;
    return runtime::makeRuntime(config);
}

TEST_CASE("the frame pool runs every item once, on its threads, and rethrows what one throws") {
    auto const runtime = smallRuntime();
    auto const parallel = render::makeFramePoolFor(*runtime);
    auto counts = std::vector<std::atomic<int>>(1000);
    auto mutex = std::mutex{};
    auto threads = std::set<std::thread::id>{};
    parallel->run(counts.size(), [&](std::size_t const index) {
        counts[index] += 1;
        auto const lock = std::lock_guard<std::mutex>{mutex};
        threads.insert(std::this_thread::get_id());
    });
    auto const once = std::all_of(counts.begin(), counts.end(), [](std::atomic<int> const &count) { return count == 1; });
    CHECK(once);
    CHECK(threads.count(std::this_thread::get_id()) == 0);

    auto reached = std::atomic<int>{0};
    CHECK_THROWS_AS(parallel->run(100,
                                  [&](std::size_t const index) {
                                      reached += 1;
                                      if (index == 37) {
                                          throw std::runtime_error{"item 37"};
                                      }
                                  }),
                    std::runtime_error);
    CHECK(reached == 100);
}

TEST_CASE("background conversions run off the calling thread, and their owner waits for those it started") {
    auto const runtime = smallRuntime();
    auto converted = std::atomic<bool>{false};
    auto result = std::future<core::ImageHandle>{};
    auto thread = std::thread::id{};
    {
        auto const conversions = render::makeBackgroundConversions(*runtime);
        result = conversions->start([&] {
            thread = std::this_thread::get_id();
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
            converted = true;
            auto image = core::ImageData{};
            image.extent = core::ivec2{1, 1};
            return std::make_shared<core::ImageData const>(std::move(image));
        });
    }
    CHECK(converted);
    auto const elsewhere = thread != std::this_thread::get_id();
    CHECK(elsewhere);
    CHECK(result.get()->extent == core::ivec2{1, 1});

    auto const conversions = render::makeBackgroundConversions(*runtime);
    auto failed = conversions->start([]() -> core::ImageHandle { throw std::runtime_error{"cannot decode"}; });
    CHECK_THROWS_AS(failed.get(), std::runtime_error);
}

} // namespace
