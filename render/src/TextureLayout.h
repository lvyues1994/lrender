#pragma once

#include "SceneUpdate.h"

#include <lr/core/RenderWorldDelta.h>

#include <tc/device.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr::render {

// What the device can hold, from its DeviceLimits.
struct TextureLimits {
    std::uint32_t maxSize = 2048; // a power of two
    std::uint32_t arrayLayers = 256;
    std::uint32_t compressedArrayFormats = 0xFFFFFFFFu; // tc::compressedBit() of each
};

TextureLimits textureLimitsOf(tc::DeviceLimits const &limits, std::uint32_t maxTextureSize);

// Where a texture lives: a layer of the pool in a slot.
struct TextureLocation {
    std::uint32_t slot = 0;
    std::uint32_t layer = 0;
};

// The render thread's plan of the texture pools: which pool each texture goes to and
// which layer it takes. Colour pool slot k holds RGBA8 textures 16 << k texels on a side
// and doubles its layers when full; compressed slots take the first four (format, size)
// pairs that come along and keep the layers they were made with. Images go to a
// compressed pool as they are when the device has arrays of their format, they are
// square with a power-of-two size the device allows and bring their whole mip chain;
// every other image is decoded to RGBA8 and resampled to the next power-of-two square
// (at most maxSize) for a colour pool, whose other mip levels the GPU generates.
struct TextureLayout {
    explicit TextureLayout(TextureLimits const &limits_) noexcept : limits(limits_) {}

    // Frees what `upsert.id` had, then places its image, adding to `update` the pools to
    // create or grow and the upload. A texture no pool has room for stays unplaced.
    void place(core::TextureUpsert const &upsert, SceneUpdate &update);
    void remove(core::TextureId id);
    // A new device: no pools, nothing placed.
    void reset() noexcept;

    std::optional<TextureLocation> locationOf(core::TextureId id) const;
    std::vector<core::TextureId> placed() const;

private:
    struct Pool {
        std::optional<tc::CompressedFormat> compressed;
        std::uint32_t size = 0; // 0: the slot has no pool
        std::uint32_t layers = 0;
        std::uint32_t used = 0; // layers handed out, freed ones included
        std::vector<std::uint32_t> freed;
    };

    std::optional<std::uint32_t> compressedSlot(tc::CompressedFormat format, std::uint32_t size, SceneUpdate &update);
    std::optional<std::uint32_t> colorSlot(std::uint32_t size, SceneUpdate &update);
    std::uint32_t takeLayer(std::uint32_t slot);
    bool fitsCompressedPool(core::ImageData const &image) const noexcept;

    TextureLimits limits;
    std::array<Pool, texturePoolSlots()> pools;
    std::unordered_map<core::TextureId, TextureLocation> locations;
};

} // namespace lr::render
