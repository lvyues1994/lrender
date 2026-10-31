#include "RenderWork.h"

#include <lexec/execution.hpp>
#include <lexec/scopes/counting_scope.hpp>

#include <cassert>
#include <exception>
#include <mutex>
#include <utility>

namespace lr::render {

namespace {

struct SerialFor final : ParallelFor {
    void run(std::size_t const count, std::function<void(std::size_t)> const &item) override {
        for (auto index = std::size_t{0}; index < count; ++index) {
            item(index);
        }
    }
};

struct FramePoolFor final : ParallelFor {
    explicit FramePoolFor(runtime::Runtime &runtime_) noexcept : runtime(runtime_) {}

    void run(std::size_t const count, std::function<void(std::size_t)> const &item) override {
        if (count < 2) {
            SerialFor{}.run(count, item);
            return;
        }
        auto mutex = std::mutex{};
        auto failure = std::exception_ptr{};
        auto const done = lexec::sync_wait(lexec::schedule(runtime.framePool())
                                           | lexec::bulk(lexec::par, count, [&](std::size_t const index) noexcept {
                                                 try {
                                                     item(index);
                                                 } catch (...) {
                                                     auto const lock = std::lock_guard<std::mutex>{mutex};
                                                     failure = failure ? failure : std::current_exception();
                                                 }
                                             }));
        assert(done.has_value());
        static_cast<void>(done);
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

private:
    runtime::Runtime &runtime;
};

struct ImmediateConversions final : ImageConversions {
    std::future<core::ImageHandle> start(std::function<core::ImageHandle()> convert) override {
        auto result = std::promise<core::ImageHandle>{};
        try {
            result.set_value(convert());
        } catch (...) {
            result.set_exception(std::current_exception());
        }
        return result.get_future();
    }
};

struct BackgroundConversions final : ImageConversions {
    explicit BackgroundConversions(runtime::Runtime &runtime_) noexcept : runtime(runtime_) {}

    BackgroundConversions(BackgroundConversions const &) = delete;
    BackgroundConversions &operator=(BackgroundConversions const &) = delete;
    ~BackgroundConversions() override { static_cast<void>(lexec::sync_wait(scope.join())); }

    std::future<core::ImageHandle> start(std::function<core::ImageHandle()> convert) override {
        auto result = std::make_shared<std::promise<core::ImageHandle>>();
        auto future = result->get_future();
        lexec::spawn(lexec::schedule(runtime.backgroundPool())
                         | lexec::then([result, convert = std::move(convert)]() noexcept {
                               try {
                                   result->set_value(convert());
                               } catch (...) {
                                   result->set_exception(std::current_exception());
                               }
                           }),
                     scope.get_token());
        return future;
    }

private:
    runtime::Runtime &runtime;
    lexec::counting_scope scope;
};

} // namespace

std::unique_ptr<ParallelFor> makeSerialFor() { return std::make_unique<SerialFor>(); }

std::unique_ptr<ParallelFor> makeFramePoolFor(runtime::Runtime &runtime) {
    return std::make_unique<FramePoolFor>(runtime);
}

std::unique_ptr<ImageConversions> makeImmediateConversions() { return std::make_unique<ImmediateConversions>(); }

std::unique_ptr<ImageConversions> makeBackgroundConversions(runtime::Runtime &runtime) {
    return std::make_unique<BackgroundConversions>(runtime);
}

} // namespace lr::render
