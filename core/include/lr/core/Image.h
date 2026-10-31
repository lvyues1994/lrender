#pragma once

#include <lr/core/Math.h>

#include <tc/compressed.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace lr::core {

// How stored colour values relate to linear ones: base colour and emissive images are
// usually sRGB-encoded, normal and metallic-roughness images linear.
enum class ColorEncoding : std::uint8_t {
    Srgb,
    Linear,
};

// A 2D image and its mip levels, level 0 first; level k is levelExtent(extent, k) texels.
// Uncompressed levels hold RGBA8 texels row by row from the top, red first; compressed
// levels hold the format's blocks in row-major order, as a KTX2 file stores them.
struct ImageData {
    ivec2 extent{0};
    ColorEncoding encoding = ColorEncoding::Srgb;
    std::optional<tc::CompressedFormat> compressed; // nullopt: RGBA8
    std::vector<std::vector<std::uint8_t>> levels;
};

using ImageHandle = std::shared_ptr<ImageData const>;

ivec2 levelExtent(ivec2 extent, std::uint32_t level) noexcept;
// The levels of a full chain, down to 1x1.
std::uint32_t fullLevelCount(ivec2 extent) noexcept;
// The bytes level `level` of `image` must hold.
std::size_t levelBytes(ImageData const &image, std::uint32_t level) noexcept;

} // namespace lr::core
