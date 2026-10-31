#include <lr/asset/AssetLibrary.h>
#include <lr/asset/FileWatcher.h>

#include "Gltf.h"
#include "ImageFiles.h"
#include "Ktx2.h"
#include "Paths.h"

#include <lexec/scopes/counting_scope.hpp>
#include <stackfull/exec/spawn.hpp>
#include <stackfull/exec/sync.hpp>
#include <stackfull/exec/wait.hpp>
#include <stackfull/sync/Completion.h>

#include <atomic>
#include <exception>
#include <filesystem>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lr::asset {

namespace {

struct Counters {
    std::atomic<std::uint64_t> loads{0};
    std::atomic<std::uint64_t> sharedRequests{0};
    std::atomic<std::uint64_t> reads{0};
    std::atomic<std::uint64_t> decodes{0};
    std::atomic<std::uint64_t> stoppedLoads{0};
};

// What load tasks use. The library outlives its tasks: it joins them when destroyed.
struct LoadContext {
    AssetSource *source;
    runtime::ComputeScheduler background;
    ImagePolicy images;
    Counters *counters;
};

template <class T>
tl::unexpected<AssetError> errorOf(AssetResult<T> const &failed) {
    return tl::unexpected<AssetError>{failed.error()};
}

// In a load task, between steps: once every requester has stopped, the task ends with
// set_stopped.
void endIfStopped() {
    if (stackfull::exec::stop_requested()) {
        throw stackfull::exec::stopped_error{};
    }
}

AssetResult<Bytes> readFile(LoadContext const &context, AssetPath const &path) {
    endIfStopped();
    context.counters->reads.fetch_add(1, std::memory_order_relaxed);
    return context.source->read(path);
}

// Runs `decode` on the background pool; the task parks until it is done. Stopped while
// still queued, it never runs.
template <class Decode>
auto onBackgroundPool(LoadContext const &context, Decode &&decode) {
    endIfStopped();
    return stackfull::exec::await(lexec::schedule(context.background) | lexec::then([&context, &decode] {
                                      context.counters->decodes.fetch_add(1, std::memory_order_relaxed);
                                      return decode();
                                  }));
}

struct ImageFile {
    AssetPath path; // what was read: the image asked for, or its KTX2 variant
    Bytes bytes;
};

bool rendererKeeps(Bytes const &ktx2, ImagePolicy const &policy) {
    auto const info = readKtx2Info(ByteView::of(ktx2));
    return info and (not info->format or keepsBlocks(*info, policy.compressedFormats));
}

// `name.<ext>`, or `name.ktx2` where the policy prefers it and the renderer keeps what it
// holds; a variant the renderer would decode is used only without the image itself.
AssetResult<ImageFile> readImageFile(LoadContext const &context, AssetPath const &path) {
    auto variant = std::optional<ImageFile>{};
    if (context.images.preferKtx2 and not hasExtension(path, ".ktx2")) {
        auto const variantPath = withExtension(path, ".ktx2");
        auto bytes = readFile(context, variantPath);
        if (bytes and rendererKeeps(*bytes, context.images)) {
            return ImageFile{variantPath, std::move(*bytes)};
        }
        if (bytes) {
            variant = ImageFile{variantPath, std::move(*bytes)};
        }
    }
    auto bytes = readFile(context, path);
    if (bytes) {
        return ImageFile{path, std::move(*bytes)};
    }
    if (variant) {
        return std::move(*variant);
    }
    return errorOf(bytes);
}

AssetResult<core::ImageHandle> loadImageInTask(LoadContext const &context, AssetPath const &path,
                                               core::ColorEncoding const encoding, std::vector<AssetPath> &files) {
    auto const file = readImageFile(context, path);
    if (not file) {
        return errorOf(file);
    }
    files.push_back(file->path);
    auto image = onBackgroundPool(context, [&] { return decodeImage(ByteView::of(file->bytes), encoding, context.images); });
    if (not image) {
        return assetError(image.error().kind, file->path + ": " + image.error().message);
    }
    return std::make_shared<core::ImageData const>(std::move(*image));
}

// The glTF file parsed, with the buffers (and images, for Everything) it refers to.
AssetResult<std::unique_ptr<GltfDocument>> readGltf(LoadContext const &context, AssetPath const &path,
                                                    GltfParts const parts, std::vector<AssetPath> &files) {
    auto bytes = readFile(context, path);
    if (not bytes) {
        return errorOf(bytes);
    }
    files.push_back(path);
    auto document = onBackgroundPool(context, [&] { return parseGltf(std::move(*bytes), path); });
    if (not document) {
        return document;
    }
    auto const buffers = (*document)->externalBuffers();
    for (auto index = std::size_t{0}; index < buffers.size(); ++index) {
        auto buffer = readFile(context, buffers[index]);
        if (not buffer) {
            return assetError(buffer.error().kind, path + " refers to " + buffer.error().message);
        }
        files.push_back(buffers[index]);
        (*document)->provideBuffer(index, std::move(*buffer));
    }
    if (parts == GltfParts::Geometry) {
        return document;
    }
    auto const images = (*document)->externalImages();
    for (auto index = std::size_t{0}; index < images.size(); ++index) {
        auto image = readImageFile(context, images[index]);
        if (not image) {
            return assetError(image.error().kind, path + " refers to " + image.error().message);
        }
        files.push_back(image->path);
        (*document)->provideImage(index, std::move(image->bytes));
    }
    return document;
}

AssetResult<SceneHandle> loadSceneInTask(LoadContext const &context, AssetPath const &path,
                                         std::vector<AssetPath> &files) {
    auto const document = readGltf(context, path, GltfParts::Everything, files);
    if (not document) {
        return errorOf(document);
    }
    auto scene = onBackgroundPool(context, [&] { return (*document)->buildScene(context.images, GltfParts::Everything); });
    if (not scene) {
        return assetError(scene.error().kind, path + ": " + scene.error().message);
    }
    return std::make_shared<SceneData const>(std::move(*scene));
}

AssetResult<core::MeshHandle> loadMeshInTask(LoadContext const &context, AssetPath const &path,
                                             std::vector<AssetPath> &files) {
    auto const document = readGltf(context, path, GltfParts::Geometry, files);
    if (not document) {
        return errorOf(document);
    }
    auto mesh = onBackgroundPool(context, [&]() -> AssetResult<core::MeshData> {
        auto const scene = (*document)->buildScene(context.images, GltfParts::Geometry);
        if (not scene) {
            return errorOf(scene);
        }
        return mergedMesh(*scene);
    });
    if (not mesh) {
        return assetError(mesh.error().kind, path + ": " + mesh.error().message);
    }
    if (mesh->indices.empty()) {
        return assetError(AssetErrorKind::Unsupported, path + ": the scene has no triangles to make a mesh of");
    }
    return std::make_shared<core::MeshData const>(std::move(*mesh));
}

template <class Handle>
using Delivery = stackfull::sync::Completion<AssetResult<Handle>>;

// One load in flight and its requesters, each with a Completion of its own: a
// Completion has a single consumer.
template <class Handle>
struct SharedLoad {
    // False once the load finished or was abandoned: start another.
    bool join(Delivery<Handle> const &delivery) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        if (finished or abandoned) {
            return false;
        }
        deliveries.push_back(delivery);
        ++waiting;
        return true;
    }

    // A requester stopped; the last one to stop abandons the load.
    void leave() noexcept {
        auto last = false;
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            if (not finished and waiting > 0) {
                --waiting;
                last = waiting == 0;
                abandoned = last;
            }
        }
        if (last) {
            stop.request_stop();
        }
    }

    // Requesters that stopped get theirs too, and nobody reads it.
    std::vector<Delivery<Handle>> finish() {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        finished = true;
        return std::move(deliveries);
    }

    lexec::inplace_stop_source stop;

private:
    std::mutex mutex;
    std::vector<Delivery<Handle>> deliveries;
    std::size_t waiting = 0;
    bool finished = false;
    bool abandoned = false;
};

template <class Handle>
struct LoadRequest {
    Delivery<Handle> delivery;
    std::shared_ptr<SharedLoad<Handle>> load;
    bool startsLoad = false;
};

// The loads of one kind of asset in flight, by key.
template <class Handle>
struct LoadTable {
    LoadRequest<Handle> join(std::string const &key, Counters &counters) {
        auto request = LoadRequest<Handle>{};
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto &slot = loads[key];
        if (slot != nullptr and slot->join(request.delivery)) {
            counters.sharedRequests.fetch_add(1, std::memory_order_relaxed);
            request.load = slot;
            return request;
        }
        slot = std::make_shared<SharedLoad<Handle>>();
        slot->join(request.delivery);
        request.load = slot;
        request.startsLoad = true;
        return request;
    }

    // Once the load's task ended; requests from now on start another load.
    std::vector<Delivery<Handle>> finish(std::string const &key, SharedLoad<Handle> &load) {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            auto const found = loads.find(key);
            if (found != loads.end() and found->second.get() == &load) {
                loads.erase(found);
            }
        }
        return load.finish();
    }

    void stopAll() {
        auto stopping = std::vector<std::shared_ptr<SharedLoad<Handle>>>{};
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            for (auto const &entry : loads) {
                stopping.push_back(entry.second);
            }
        }
        for (auto const &load : stopping) {
            load->stop.request_stop();
        }
    }

private:
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<SharedLoad<Handle>>> loads;
};

std::string messageOf(std::exception_ptr const &error) {
    try {
        std::rethrow_exception(error);
    } catch (std::exception const &caught) {
        return caught.what();
    } catch (...) {
        return "an unknown exception";
    }
}

std::string messageOf(std::error_code const &error) { return "no IO task could start: " + error.message(); }

std::string keyOf(AssetPath const &path, core::ColorEncoding const encoding) {
    return path + (encoding == core::ColorEncoding::Srgb ? "\n(sRGB)" : "\n(linear)");
}

// Which loaded assets each watched file feeds.
struct WatchedFiles {
    void add(std::string const &file, AssetPath const &asset) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        dependents[file].insert(asset);
    }

    void collect(FileChange const &change, std::set<AssetPath> &assets) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        if (change.kind == FileChangeKind::Overflow) {
            for (auto const &entry : dependents) {
                assets.insert(entry.second.begin(), entry.second.end());
            }
            return;
        }
        auto const found = dependents.find(change.path);
        if (change.kind == FileChangeKind::Modified and found != dependents.end()) {
            assets.insert(found->second.begin(), found->second.end());
        }
    }

private:
    std::mutex mutex;
    std::unordered_map<std::string, std::set<AssetPath>> dependents;
};

LoadContext contextOf(AssetLibraryConfig const &config, Counters &counters) {
    if (config.source == nullptr or config.runtime == nullptr) {
        throw std::invalid_argument{"lr::asset: an asset library needs a source and a runtime"};
    }
    return LoadContext{config.source, config.runtime->backgroundPool(),
                       ImagePolicy{config.compressedFormats, config.preferKtx2}, &counters};
}

std::unique_ptr<FileWatcher> watcherOf(AssetLibraryConfig const &config) {
    if (not config.watchFiles) {
        return nullptr;
    }
    auto watcherConfig = FileWatcherConfig{};
    watcherConfig.runtime = config.runtime;
    return makeFileWatcher(watcherConfig);
}

struct AssetLibraryImpl final : AssetLibrary {
    explicit AssetLibraryImpl(AssetLibraryConfig const &config)
        : context(contextOf(config, counters)), io(config.runtime->io()), watcher(watcherOf(config)) {}

    ~AssetLibraryImpl() override {
        meshes.stopAll();
        images.stopAll();
        scenes.stopAll();
        static_cast<void>(lexec::sync_wait(scope.join()));
    }

    LoadSender<core::MeshHandle> loadMesh(AssetPath const &path) override {
        return request(meshes, path, path, [this, path](std::vector<AssetPath> &files) {
            return loadMeshInTask(context, path, files);
        });
    }

    LoadSender<core::ImageHandle> loadImage(AssetPath const &path, core::ColorEncoding const encoding) override {
        return request(images, keyOf(path, encoding), path, [this, path, encoding](std::vector<AssetPath> &files) {
            return loadImageInTask(context, path, encoding, files);
        });
    }

    LoadSender<SceneHandle> loadScene(AssetPath const &path) override {
        return request(scenes, path, path, [this, path](std::vector<AssetPath> &files) {
            return loadSceneInTask(context, path, files);
        });
    }

    void drainChanges(std::vector<AssetChange> &changes) override {
        changes.clear();
        if (watcher == nullptr) {
            return;
        }
        watcher->drainChanges(fileChanges);
        auto assets = std::set<AssetPath>{};
        for (auto const &change : fileChanges) {
            watchedFiles.collect(change, assets);
        }
        for (auto const &asset : assets) {
            changes.push_back(AssetChange{asset});
        }
    }

    AssetLibraryStats stats() const noexcept override {
        auto result = AssetLibraryStats{};
        result.loads = counters.loads.load(std::memory_order_relaxed);
        result.sharedRequests = counters.sharedRequests.load(std::memory_order_relaxed);
        result.reads = counters.reads.load(std::memory_order_relaxed);
        result.decodes = counters.decodes.load(std::memory_order_relaxed);
        result.stoppedLoads = counters.stoppedLoads.load(std::memory_order_relaxed);
        return result;
    }

private:
    // Each start of the sender joins the load of `key` in flight or starts one, then
    // waits for its own delivery.
    template <class Handle, class Load>
    LoadSender<Handle> request(LoadTable<Handle> &table, std::string key, AssetPath asset, Load load) {
        return lexec::let_value(lexec::just(), [this, &table, key = std::move(key), asset = std::move(asset),
                                                load = std::move(load)]() noexcept {
            auto joined = table.join(key, counters);
            if (joined.startsLoad) {
                start(table, key, asset, joined.load, load);
            }
            return stackfull::exec::async_get(joined.delivery)
                 | lexec::let_stopped([shared = std::move(joined.load)]() noexcept {
                       shared->leave();
                       return lexec::just_stopped();
                   })
                 // Only a load that could not finish fails deliveries: stopped with the library.
                 | lexec::let_error([](auto &) noexcept { return lexec::just_stopped(); });
        });
    }

    // The load's task on the IO scheduler, outside the table's lock: a task that cannot
    // start completes at once.
    template <class Handle, class Load>
    void start(LoadTable<Handle> &table, std::string const &key, AssetPath const &asset,
               std::shared_ptr<SharedLoad<Handle>> const &load, Load const &body) {
        counters.loads.fetch_add(1, std::memory_order_relaxed);
        auto task = stackfull::exec::spawn(io, [this, asset, body] {
            auto files = std::vector<AssetPath>{};
            auto result = body(files);
            if (result) {
                watch(asset, files);
            }
            return result;
        });
        auto const deliver = [&table, key, load](AssetResult<Handle> const &result) {
            for (auto const &delivery : table.finish(key, *load)) {
                delivery.set(result);
            }
        };
        lexec::spawn(lexec::write_env(std::move(task), lexec::prop{lexec::get_stop_token, load->stop.get_token()})
                         | lexec::then([deliver](AssetResult<Handle> result) noexcept { deliver(result); })
                         | lexec::upon_error([deliver](auto const &error) noexcept {
                               deliver(assetError(AssetErrorKind::Failed, messageOf(error)));
                           })
                         | lexec::upon_stopped([this, &table, key, load]() noexcept {
                               counters.stoppedLoads.fetch_add(1, std::memory_order_relaxed);
                               for (auto const &delivery : table.finish(key, *load)) {
                                   delivery.fail(std::make_error_code(std::errc::operation_canceled));
                               }
                           }),
                     scope.get_token());
    }

    // In the load's task, before its requesters hear of it, so that a change right after
    // the load completes is reported.
    void watch(AssetPath const &asset, std::vector<AssetPath> const &files) {
        if (watcher == nullptr) {
            return;
        }
        for (auto const &file : files) {
            auto const local = context.source->localFile(file);
            if (not local) {
                continue;
            }
            auto const directory = std::filesystem::path{*local}.parent_path().string();
            if (not watcher->watchDirectory(directory)) {
                watchedFiles.add(*local, asset);
            }
        }
    }

    Counters counters;
    LoadContext context;
    runtime::IoScheduler io;
    LoadTable<core::MeshHandle> meshes;
    LoadTable<core::ImageHandle> images;
    LoadTable<SceneHandle> scenes;
    std::unique_ptr<FileWatcher> watcher;
    WatchedFiles watchedFiles;
    std::vector<FileChange> fileChanges; // host thread
    lexec::simple_counting_scope scope;
};

} // namespace

std::unique_ptr<AssetLibrary> makeAssetLibrary(AssetLibraryConfig const &config) {
    return std::make_unique<AssetLibraryImpl>(config);
}

} // namespace lr::asset
