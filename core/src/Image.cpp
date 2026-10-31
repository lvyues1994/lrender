#include <lr/core/Image.h>

#include <algorithm>

namespace lr::core {

ivec2 levelExtent(ivec2 const extent, std::uint32_t const level) noexcept {
    return ivec2{std::max(extent.x >> level, 1), std::max(extent.y >> level, 1)};
}

std::uint32_t fullLevelCount(ivec2 const extent) noexcept {
    auto levels = std::uint32_t{1};
    for (auto side = std::max(extent.x, extent.y); side > 1; side >>= 1) {
        ++levels;
    }
    return levels;
}

std::size_t levelBytes(ImageData const &image, std::uint32_t const level) noexcept {
    if (image.compressed) {
        return tc::compressedLayerBytes(*image.compressed, image.extent, level);
    }
    auto const size = levelExtent(image.extent, level);
    return static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y) * 4;
}

} // namespace lr::core
