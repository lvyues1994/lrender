#pragma once

#include "RetireList.h"
#include "SceneUpdate.h"

#include <lr/render/GpuTypes.h>

#include <tc/device.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr::render {

struct TexturePoolConfig {
    // Bytes of texture data copied to the device per frame; a mip level larger than this
    // goes alone.
    std::size_t streamingBytes = std::size_t{8} << 20;
    std::uint32_t framesInFlight = 2;
};

// The GPU thread's texture pools, laid out by TextureLayout. Uploads go to the device's
// transfer queue (tc::Device::transfers()), next to the frames' dispatches. As a frame
// starts, the levels within the budget are written to a staging buffer from
// Device::createBuffer and copied once the device is done with the previous frame, the
// last one that may still sample the layers they overwrite (and with the pools this frame
// created or grew, which go to the device first). The frame's views do not wait for the
// copies; what the frame records after them does, generating the mip levels of the colour
// textures the copies complete. A texture counts as resident, and materials refer to it,
// from the next frame on. The staging buffers form a ring with one per frame in flight:
// each frame writes the one the frame framesInFlight before it used, which has finished,
// and with it the copies it waited for.
struct TexturePool {
    explicit TexturePool(TexturePoolConfig const &config_) noexcept : config(config_) {}

    // Creates and grows pools, forgets removed textures and queues the uploads. Textures
    // being replaced are not resident until their new data is in.
    void apply(tc::Device &device, SceneUpdate const &update, RetireList &retired);
    // After apply(): sends the queued levels within the budget to the transfer queue, to
    // be copied after `previous`, the submission of the frame before.
    void send(tc::Device &device, tc::Submission const &previous, RetireList &retired);
    // Before the frame is submitted, after its views: waits on the device for the copies,
    // generates the mip levels of the textures they complete and makes them resident.
    void arrive(tc::Device &device);
    // Whether textures became resident or stopped being so since the last call.
    bool takeResidencyChange() noexcept;

    tc::uint residentRef(core::TextureId id) const;
    MaterialTextures textures() const noexcept;
    std::size_t queuedUploads() const noexcept { return queue.size(); }
    // What the pools take on the device, every mip level of every layer.
    std::uint64_t memoryBytes() const noexcept;
    // Drops every device resource, as when the device is lost.
    void clear() noexcept;

private:
    struct Pool {
        std::optional<tc::CompressedFormat> compressed;
        std::uint32_t size = 0; // 0: none
        std::uint32_t layers = 0;
        ColorPoolImage color;
        CompressedPoolImage blocks;
    };
    struct Pending {
        TextureUpload upload;
        std::uint32_t nextLevel = 0;
    };
    // The copies on the transfer queue, and the textures whose last level they carry.
    struct Sent {
        tc::Submission copies;
        std::vector<TextureUpload> completed;
    };

    void prepare(tc::Device &device, RetireList &retired);
    // This send's staging buffer, made or enlarged to hold at least `bytes`.
    tc::BufferResource<tc::uint> &stagingFor(tc::Device &device, std::size_t bytes, RetireList &retired);
    void change(tc::Device &device, PoolChange const &request, RetireList &retired);
    void forget(core::TextureId id);
    void copyLevel(tc::TransferQueue &transfers, tc::BufferResource<tc::uint> const &staging, std::size_t offset,
                   Pending const &pending);
    void generateMips(tc::Device &device, tc::uint ref);

    TexturePoolConfig config;
    bool prepared = false;
    std::array<Pool, texturePoolSlots()> pools;
    ColorPoolImage emptyColor;
    CompressedPoolImage emptyBlocks;
    EnvironmentBrdfImage environmentBrdf;
    std::vector<std::unique_ptr<tc::BufferResource<tc::uint>>> staging; // made at first use

    std::uint64_t streamedFrames = 0;
    std::deque<Pending> queue;
    std::optional<Sent> sent;
    bool poolsChanged = false; // since the last send: the copies follow those commands
    std::unordered_map<core::TextureId, tc::uint> resident;
    bool residencyChanged = false;
    std::uint64_t residency = 0; // never goes back, not even through clear()
};

} // namespace lr::render
