#include "ImageFiles.h"

#include <tc/compressed.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lr::asset {

namespace {

bool isPowerOfTwo(int const value) noexcept { return value > 0 and (value & (value - 1)) == 0; }

std::uint8_t quantized(float const value) noexcept {
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

} // namespace

bool keepsBlocks(Ktx2Info const &image, std::uint32_t const compressedFormats) noexcept {
    if (not image.format or (compressedFormats & tc::compressedBit(*image.format)) == 0) {
        return false;
    }
    return image.extent.x == image.extent.y and isPowerOfTwo(image.extent.x)
       and image.levels >= core::fullLevelCount(image.extent);
}

core::ImageData decodedToRgba8(core::ImageData const &image) {
    auto result = core::ImageData{};
    result.extent = image.extent;
    result.encoding = image.encoding;
    for (auto level = std::uint32_t{0}; level < image.levels.size(); ++level) {
        auto const extent = core::levelExtent(image.extent, level);
        auto const count = static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y);
        auto decoded = std::vector<tc::cpu::RGBA32F>(count);
        auto const &blocks = image.levels[level];
        tc::decodeCompressed(*image.compressed, extent, blocks.data(), blocks.size(), decoded.data());
        auto &texels = result.levels.emplace_back(count * 4);
        for (auto index = std::size_t{0}; index < count; ++index) {
            for (auto channel = std::size_t{0}; channel < 4; ++channel) {
                texels[index * 4 + channel] = quantized(decoded[index].channels[channel]);
            }
        }
    }
    return result;
}

AssetResult<core::ImageData> decodeImage(ByteView const bytes, core::ColorEncoding const encoding,
                                         ImagePolicy const &policy) {
    if (not isKtx2(bytes)) {
        return decodePngOrJpeg(bytes, encoding);
    }
    auto image = readKtx2(bytes, encoding);
    if (not image or not image->compressed) {
        return image;
    }
    auto const shape = Ktx2Info{image->compressed, image->extent, static_cast<std::uint32_t>(image->levels.size())};
    if (keepsBlocks(shape, policy.compressedFormats)) {
        return image;
    }
    return decodedToRgba8(*image);
}

} // namespace lr::asset
