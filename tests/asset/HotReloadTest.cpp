#include "AssetTestSupport.h"

#include <lr/asset/FileWatcher.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <vector>

namespace {

using namespace lr;
using asset::FileChange;
using asset::FileChangeKind;
using test::AssetHarness;
using test::AssetHarnessOptions;

test::Bytes pngOf(std::uint8_t const red) {
    return test::pngOf(core::ivec2{2, 2}, test::solid(core::ivec2{2, 2}, red, 0, 0));
}

AssetHarness watchingHarness(std::filesystem::path const &directory) {
    auto source = asset::FileSourceConfig{};
    source.root = directory.string();
    auto options = AssetHarnessOptions{};
    options.watchFiles = true;
    return AssetHarness{asset::makeFileSource(source), options};
}

// Drains `library` until it reports `asset`.
bool reports(asset::AssetLibrary &library, std::string const &asset) {
    auto changes = std::vector<asset::AssetChange>{};
    return test::eventually([&] {
        library.drainChanges(changes);
        return std::any_of(changes.begin(), changes.end(),
                           [&asset](asset::AssetChange const &change) { return change.asset == asset; });
    });
}

TEST_CASE("a file watcher reports files created, modified and removed in a watched directory") {
    auto const directory = test::freshDirectory("file-watcher");
    auto runtimeConfig = runtime::RuntimeConfig{};
    runtimeConfig.framePoolThreads = 1;
    auto const runtime = runtime::makeRuntime(runtimeConfig);
    auto watcherConfig = asset::FileWatcherConfig{};
    watcherConfig.runtime = runtime.get();
    auto const watcher = asset::makeFileWatcher(watcherConfig);
    test::writeFile(directory / "old.txt", test::Bytes{1});
    auto const watching = watcher->watchDirectory(directory.string());
    REQUIRE(not watching);

    test::writeFile(directory / "new.txt", test::Bytes{2});
    test::writeFile(directory / "old.txt", test::Bytes{3});
    auto seen = std::vector<FileChange>{};
    auto const reported = [&seen, &directory](FileChangeKind const kind, char const *const name) {
        auto const path = (directory / name).string();
        return std::any_of(seen.begin(), seen.end(),
                           [&](FileChange const &change) { return change.kind == kind and change.path == path; });
    };
    auto const drain = [&seen, &watcher] {
        auto changes = std::vector<FileChange>{};
        watcher->drainChanges(changes);
        seen.insert(seen.end(), changes.begin(), changes.end());
    };
    auto const written = test::eventually([&] {
        drain();
        return reported(FileChangeKind::Created, "new.txt") and reported(FileChangeKind::Modified, "new.txt")
           and reported(FileChangeKind::Modified, "old.txt");
    });
    CHECK(written);
    CHECK(not reported(FileChangeKind::Created, "old.txt"));

    std::filesystem::remove(directory / "new.txt");
    auto const removed = test::eventually([&] {
        drain();
        return reported(FileChangeKind::Removed, "new.txt");
    });
    CHECK(removed);
}

TEST_CASE("the library reports a changed image file it loaded, and loading it again sees the new texels") {
    auto const directory = test::freshDirectory("hot-reload-image");
    test::writeFile(directory / "tex.png", pngOf(255));
    auto harness = watchingHarness(directory);
    auto const first = test::loadNow(harness.library->loadImage("tex.png", core::ColorEncoding::Srgb));
    REQUIRE(first.has_value());
    auto changes = std::vector<asset::AssetChange>{};
    harness.library->drainChanges(changes);
    CHECK(changes.empty());

    test::writeFile(directory / "tex.png", pngOf(64));

    CHECK(reports(*harness.library, "tex.png"));
    auto const second = test::loadNow(harness.library->loadImage("tex.png", core::ColorEncoding::Srgb));
    REQUIRE(second.has_value());
    auto const replaced = (*second)->levels.at(0) == test::solid(core::ivec2{2, 2}, 64, 0, 0) and *second != *first;
    CHECK(replaced);
}

TEST_CASE("a scene is reported when a buffer it was loaded from changes") {
    auto const directory = test::freshDirectory("hot-reload-scene");
    auto scene = test::TestGltf{};
    scene.meshes = {test::quadPrimitive(1.0f)};
    scene.materials = "[{}]";
    auto const files = test::writeGltf(scene, test::GltfLayout::External, "quad");
    for (auto const &[name, bytes] : files.files) {
        test::writeFile(directory / name, bytes);
    }
    auto harness = watchingHarness(directory);
    auto const loaded = test::loadNow(harness.library->loadScene(files.main));
    REQUIRE(loaded.has_value());

    test::writeFile(directory / "quad.bin", files.files.at("quad.bin"));

    CHECK(reports(*harness.library, files.main));
}

} // namespace
