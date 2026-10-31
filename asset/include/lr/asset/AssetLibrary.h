#pragma once

#include <lr/asset/AssetResult.h>
#include <lr/asset/AssetSource.h>
#include <lr/asset/Scene.h>

#include <lr/core/Image.h>
#include <lr/core/Mesh.h>
#include <lr/runtime/Runtime.h>

#include <lexec/any_sender_of.hpp>
#include <lexec/execution.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lr::asset {

// A failed load is an AssetError value; a stopped requester completes with set_stopped.
template <class T>
using LoadSender = lexec::any_sender_of<lexec::set_value_t(AssetResult<T>), lexec::set_stopped_t()>;

struct AssetLibraryConfig {
    AssetSource *source = nullptr;       // required; outlives the library
    runtime::Runtime *runtime = nullptr; // required; outlives the library
    // The compressed formats (tc::compressedBit) whose blocks the renderer keeps: its
    // device's DeviceLimits::compressedArrayFormats. Other compressed images are decoded
    // to RGBA8 on the background pool.
    std::uint32_t compressedFormats = 0;
    // An image `name.<ext>` loads `name.ktx2` instead where that file exists and holds
    // RGBA8 or blocks the renderer keeps.
    bool preferKtx2 = true;
    // Watch the local files assets were loaded from, for drainChanges().
    bool watchFiles = false;
};

struct AssetChange {
    AssetPath asset; // a file it was loaded from changed: load it again
};

struct AssetLibraryStats {
    std::uint64_t loads = 0;          // load tasks started
    std::uint64_t sharedRequests = 0; // requests that waited for a load in flight instead
    std::uint64_t reads = 0;          // AssetSource::read() calls
    std::uint64_t decodes = 0;        // decoding steps run on the background pool
    std::uint64_t stoppedLoads = 0;   // loads that ended stopped
};

// Loads run as stackfull tasks on the runtime's IO scheduler, one per load: the task
// reads through the source and decodes on the background pool, parked meanwhile, so the
// IO workers run other loads. A load starts when its sender is started. Senders started
// while a load of the same asset is in flight (the same path, and for images the same
// encoding) wait for it and get the same handle; nothing is cached after a load ends.
//
// Stopping: a requester completes with set_stopped as soon as its stop token fires. The
// shared load goes on while any of its requesters waits; once none does, it ends at its
// next step (before the next read or decoding step) with no result, and a later request
// starts a new load.
//
// Destroy it on a plain thread. Loads in flight then stop and their requesters complete
// with set_stopped.
struct AssetLibrary {
    virtual ~AssetLibrary() = default;

    // A glTF file's default scene merged into one mesh in scene space, without materials.
    virtual LoadSender<core::MeshHandle> loadMesh(AssetPath const &path) = 0;
    // A PNG, JPEG or KTX2 file (uncompressed RGBA8 or block-compressed, not
    // supercompressed). `encoding` says how its values relate to linear ones: sRGB for
    // colour, linear for data such as normals, whatever a KTX2 format's name says.
    virtual LoadSender<core::ImageHandle> loadImage(AssetPath const &path, core::ColorEncoding encoding) = 0;
    // A .gltf file (buffers and images in files or data URIs) or a .glb file, with its
    // images: base colour and emissive ones sRGB, the others linear.
    virtual LoadSender<SceneHandle> loadScene(AssetPath const &path) = 0;

    // Host thread, never waits: replaces `changes` with the assets whose files changed
    // since the last call (with watchFiles).
    virtual void drainChanges(std::vector<AssetChange> &changes) = 0;
    virtual AssetLibraryStats stats() const noexcept = 0;
};

// Throws std::invalid_argument without a source or a runtime, and std::system_error
// when it cannot watch files.
std::unique_ptr<AssetLibrary> makeAssetLibrary(AssetLibraryConfig const &config);

} // namespace lr::asset
