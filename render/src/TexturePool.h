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

// The GPU thread's texture pools, laid out by TextureLayout. Uploads stream in through a
// ring of staging buffers from Device::createBuffer, one per frame in flight: each frame
// writes the next one, which the frame framesInFlight before it used and has finished
// with. A texture counts as resident, and materials refer to it, once its last level is
// in and, in colour pools, its mip levels have been generated from level 0.
struct TexturePool {
    explicit TexturePool(TexturePoolConfig const &config_) noexcept : config(config_) {}

    // Creates and grows pools, forgets removed textures and queues the uploads. Textures
    // being replaced are not resident until their new data is in.
    void apply(tc::Device &device, SceneUpdate const &update, RetireList &retired);
    // Copies queued levels within this frame's budget and generates the mip levels of the
    // colour textures it completes. Once per recorded frame, before the frame samples them.
    void stream(tc::Device &device, RetireList &retired);
    // Whether textures became resident or stopped being so since the last call.
    bool takeResidencyChange() noexcept;

    tc::uint residentRef(core::TextureId id) const;
    MaterialTextures textures() const noexcept;
    std::size_t queuedUploads() const noexcept { return queue.size(); }
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

    void prepare(tc::Device &device, RetireList &retired);
    // This frame's staging buffer, made or enlarged to hold at least `bytes`.
    tc::BufferResource<tc::uint> &stagingFor(tc::Device &device, std::size_t bytes, RetireList &retired);
    void change(tc::Device &device, PoolChange const &request, RetireList &retired);
    void forget(core::TextureId id);
    void copyLevel(tc::Device &device, tc::BufferResource<tc::uint> const &staging, std::size_t offset,
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
    std::unordered_map<core::TextureId, tc::uint> resident;
    bool residencyChanged = false;
    std::uint64_t residency = 0; // never goes back, not even through clear()
};

} // namespace lr::render
