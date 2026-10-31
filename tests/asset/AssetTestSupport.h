#pragma once

#include "TestFiles.h"

#include <lr/asset/AssetLibrary.h>
#include <lr/asset/AssetSource.h>
#include <lr/runtime/Runtime.h>

#include <lexec/execution.hpp>
#include <lexec/stop_token.hpp>
#include <stackfull/exec/spawn.hpp>
#include <stackfull/runtime/Runtime.h>

#include <doctest/doctest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace lr::test {

using namespace std::chrono_literals;

// Files in memory, with every read counted. Closed, it holds reads back until open() or
// until the reading load is asked to stop: the task waits on stackfull's blocking pool,
// so the IO workers stay free.
struct TestSource final : asset::AssetSource {
    explicit TestSource(std::unordered_map<std::string, Bytes> files_ = {}, bool const opened_ = true)
        : files(std::move(files_)), opened(opened_) {}

    asset::AssetResult<Bytes> read(asset::AssetPath const &path) override {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            ++reads[path];
        }
        changed.notify_all();
        auto const stop = stackfull::exec::get_stop_token();
        auto const wake = lexec::inplace_stop_callback{stop, Wake{this}};
        stackfull::blocking([this, &stop] {
            auto lock = std::unique_lock<std::mutex>{mutex};
            changed.wait(lock, [this, &stop] { return opened or stop.stop_requested(); });
        });
        auto result = asset::AssetResult<Bytes>{};
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            ++finishedReads[path];
            auto const found = files.find(path);
            if (found == files.end()) {
                result = asset::assetError(asset::AssetErrorKind::NotFound, path + ": not in the test source");
            } else {
                result = found->second;
            }
        }
        changed.notify_all();
        return result;
    }

    std::optional<std::string> localFile(asset::AssetPath const &) const override { return std::nullopt; }

    void put(std::string const &path, Bytes bytes) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        files[path] = std::move(bytes);
    }

    void open() {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            opened = true;
        }
        changed.notify_all();
    }

    std::size_t readsOf(std::string const &path) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return reads[path];
    }

    // Whether reading `path` began `count` times within `timeout`.
    bool waitForReads(std::string const &path, std::size_t const count, std::chrono::milliseconds const timeout = 5s) {
        auto lock = std::unique_lock<std::mutex>{mutex};
        return changed.wait_for(lock, timeout, [&] { return reads[path] >= count; });
    }

    // Whether reading `path` returned `count` times within `timeout`.
    bool waitForFinishedReads(std::string const &path, std::size_t const count,
                              std::chrono::milliseconds const timeout = 5s) {
        auto lock = std::unique_lock<std::mutex>{mutex};
        return changed.wait_for(lock, timeout, [&] { return finishedReads[path] >= count; });
    }

private:
    struct Wake {
        void operator()() const noexcept {
            { auto const lock = std::lock_guard<std::mutex>{source->mutex}; }
            source->changed.notify_all();
        }

        TestSource *source;
    };

    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<std::string, Bytes> files;
    std::unordered_map<std::string, std::size_t> reads;
    std::unordered_map<std::string, std::size_t> finishedReads;
    bool opened;
};

struct AssetHarnessOptions {
    std::uint32_t ioWorkers = 2;
    std::uint32_t backgroundThreads = 1;
    std::uint32_t compressedFormats = 0;
    bool preferKtx2 = false;
    bool watchFiles = false;
};

// A library over a test source, or over another source (a file source).
struct AssetHarness {
    explicit AssetHarness(std::unique_ptr<asset::AssetSource> source_, AssetHarnessOptions const &options = {}) {
        auto runtimeConfig = runtime::RuntimeConfig{};
        runtimeConfig.framePoolThreads = 1;
        runtimeConfig.backgroundPoolThreads = options.backgroundThreads;
        runtimeConfig.ioWorkers = options.ioWorkers;
        runtimeInstance = runtime::makeRuntime(runtimeConfig);
        source = std::move(source_);
        auto config = asset::AssetLibraryConfig{};
        config.source = source.get();
        config.runtime = runtimeInstance.get();
        config.compressedFormats = options.compressedFormats;
        config.preferKtx2 = options.preferKtx2;
        config.watchFiles = options.watchFiles;
        library = asset::makeAssetLibrary(config);
    }

    explicit AssetHarness(std::unordered_map<std::string, Bytes> files, AssetHarnessOptions const &options = {})
        : AssetHarness(std::make_unique<TestSource>(std::move(files)), options) {}

    TestSource &testSource() { return static_cast<TestSource &>(*source); }

    std::unique_ptr<runtime::Runtime> runtimeInstance;
    std::unique_ptr<asset::AssetSource> source;
    std::unique_ptr<asset::AssetLibrary> library;
};

// Fails the test if the load stopped.
template <class T>
asset::AssetResult<T> loadNow(asset::LoadSender<T> sender) {
    auto result = lexec::sync_wait(std::move(sender));
    REQUIRE(result.has_value());
    return std::get<0>(std::move(*result));
}

template <class Predicate>
bool eventually(Predicate const &predicate, std::chrono::milliseconds const timeout = 5s) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (not predicate()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(5ms);
    }
    return true;
}

// An empty directory of the tests' own under the build tree.
inline std::filesystem::path freshDirectory(std::string const &name) {
    auto const directory = std::filesystem::path{LR_TEST_OUTPUT_DIR} / name;
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

inline void writeFile(std::filesystem::path const &path, Bytes const &bytes) {
    auto file = std::ofstream{path, std::ios::binary | std::ios::trunc};
    file.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <class T>
bool isError(asset::AssetResult<T> const &result, asset::AssetErrorKind const kind) {
    return not result and result.error().kind == kind;
}

} // namespace lr::test
