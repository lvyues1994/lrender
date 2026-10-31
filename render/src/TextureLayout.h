#pragma once

#include "RenderWork.h"
#include "SceneUpdate.h"

#include <lr/core/RenderWorldDelta.h>
#include <lr/render/Renderer.h>

#include <tc/device.hpp>

#include <array>
#include <cstdint>
#include <future>
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
    std::uint64_t placement = 0; // tells this placing of the texture from later ones
};

// The render thread's plan of the texture pools: which pool each texture goes to and
// which layer it takes. Colour pool slot k holds RGBA8 textures 16 << k texels on a side
// and doubles its layers when full; compressed slots take the first four (format, size)
// pairs that come along and keep the layers they were made with. Images go to a
// compressed pool as they are when the device has arrays of their format, they are
// square with a power-of-two size the device allows and bring their whole mip chain;
// every other image is decoded to RGBA8 and resampled to the next power-of-two square
// (at most maxSize) for a colour pool, whose other mip levels the GPU generates. Those
// conversions run on `conversions` while the frame goes on, and their uploads go with a
// later frame's update as `delivery` says; images that go to a pool as they are upload at
// once.
struct TextureLayout {
    // `conversions` must outlive the layout.
    TextureLayout(TextureLimits const &limits_, ImageConversions &conversions_,
                  ConversionDelivery const delivery_ = ConversionDelivery::Deterministic) noexcept
        : limits(limits_), conversions(&conversions_), delivery(delivery_) {}

    // Frees what `upsert.id` had, then places its image, adding to `update` the pools to
    // create or grow and the upload, or starting its conversion. A texture no pool has
    // room for stays unplaced.
    void place(core::TextureUpsert const &upsert, SceneUpdate &update);
    void remove(core::TextureId id);
    // Once a frame, after its place() and remove() calls: adds the uploads of the textures
    // earlier frames started converting that are still placed where they were, of those
    // whose conversion has finished or, Deterministic, of all of them, waiting.
    void deliverConversions(SceneUpdate &update);
    // A new device: no pools, nothing placed, nothing converting.
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

    struct Conversion {
        core::TextureId id;
        std::uint64_t placement = 0;
        tc::uint ref = 0;
        std::future<core::ImageHandle> image;
        std::uint64_t frame = 0; // that started it
    };

    std::optional<std::uint32_t> compressedSlot(tc::CompressedFormat format, std::uint32_t size, SceneUpdate &update);
    std::optional<std::uint32_t> colorSlot(std::uint32_t size, SceneUpdate &update);
    std::uint32_t takeLayer(std::uint32_t slot);
    bool fitsCompressedPool(core::ImageData const &image) const noexcept;
    TextureLocation locate(core::TextureId id, std::uint32_t slot);

    TextureLimits limits;
    ImageConversions *conversions;
    ConversionDelivery delivery;
    std::array<Pool, texturePoolSlots()> pools;
    std::unordered_map<core::TextureId, TextureLocation> locations;
    std::uint64_t placements = 0;
    std::vector<Conversion> converting;
    std::uint64_t frame = 0;
};

} // namespace lr::render
