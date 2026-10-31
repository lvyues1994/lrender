#include "TextureLayout.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace lr::render {

namespace {

constexpr auto kSmallestColorPool = std::uint32_t{16};
constexpr auto kFirstColorLayers = std::uint32_t{4};
// Compressed pools cannot grow, so they start this large (and with at least 4 layers).
constexpr auto kCompressedPoolBytes = std::size_t{32} << 20;
constexpr auto kMaxLayers = std::uint32_t{4096}; // what textureRef() can name

bool isPowerOfTwo(std::uint32_t const value) noexcept { return value != 0 and (value & (value - 1)) == 0; }

std::uint32_t floorPowerOfTwo(std::uint32_t const value) noexcept {
    auto result = std::uint32_t{1};
    while (result * 2 <= value and result < (1u << 30)) {
        result *= 2;
    }
    return result;
}

std::uint32_t ceilPowerOfTwo(std::uint32_t const value) noexcept {
    auto result = std::uint32_t{1};
    while (result < value) {
        result *= 2;
    }
    return result;
}

std::uint32_t log2Of(std::uint32_t const powerOfTwo) noexcept {
    auto result = std::uint32_t{0};
    while ((1u << result) < powerOfTwo) {
        ++result;
    }
    return result;
}

tc::uint wrapCode(core::TextureWrap const wrap) noexcept {
    switch (wrap) {
    case core::TextureWrap::ClampToEdge:
        return wrapClampToEdge();
    case core::TextureWrap::MirroredRepeat:
        return wrapMirroredRepeat();
    case core::TextureWrap::Repeat:
        break;
    }
    return wrapRepeat();
}

float decodeChannel(std::uint8_t const value, bool const srgb) noexcept {
    auto const encoded = static_cast<float>(value) / 255.0f;
    if (not srgb) {
        return encoded;
    }
    return encoded <= 0.04045f ? encoded / 12.92f : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

std::uint8_t encodeChannel(float const value, bool const srgb) noexcept {
    auto const linear = std::clamp(value, 0.0f, 1.0f);
    auto const encoded =
        not srgb ? linear : (linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f);
    return static_cast<std::uint8_t>(std::lround(std::clamp(encoded, 0.0f, 1.0f) * 255.0f));
}

// Texels in linear colour (alpha as it is), rows from the top.
struct FloatImage {
    int width = 0;
    int height = 0;
    std::vector<core::vec4> texels;
};

int address(int const index, int const count, core::TextureWrap const wrap) noexcept {
    switch (wrap) {
    case core::TextureWrap::Repeat: {
        auto const wrapped = index % count;
        return wrapped < 0 ? wrapped + count : wrapped;
    }
    case core::TextureWrap::MirroredRepeat: {
        auto const period = 2 * count;
        auto wrapped = index % period;
        wrapped = wrapped < 0 ? wrapped + period : wrapped;
        return wrapped < count ? wrapped : period - 1 - wrapped;
    }
    case core::TextureWrap::ClampToEdge:
        break;
    }
    return std::clamp(index, 0, count - 1);
}

// Each row resampled to `width` texels: the mean over each new texel's span when it
// shrinks, linear interpolation between texel centres, wrapped as the texture wraps, when
// it grows.
FloatImage resampleRows(FloatImage const &source, int const width, core::TextureWrap const wrap) {
    auto result = FloatImage{width, source.height, std::vector<core::vec4>(static_cast<std::size_t>(width) *
                                                                             static_cast<std::size_t>(source.height))};
    auto const scale = static_cast<double>(source.width) / static_cast<double>(width);
    for (auto y = 0; y < source.height; ++y) {
        auto const *in = source.texels.data() + static_cast<std::ptrdiff_t>(y) * source.width;
        auto *out = result.texels.data() + static_cast<std::ptrdiff_t>(y) * width;
        for (auto x = 0; x < width; ++x) {
            if (scale >= 1.0) {
                auto const start = x * scale;
                auto const end = (x + 1) * scale;
                auto sum = core::vec4{0.0f};
                for (auto i = static_cast<int>(std::floor(start)); i < static_cast<int>(std::ceil(end)); ++i) {
                    auto const overlap = std::min(end, i + 1.0) - std::max(start, static_cast<double>(i));
                    sum += in[std::min(i, source.width - 1)] * static_cast<float>(overlap);
                }
                out[x] = sum / static_cast<float>(scale);
            } else {
                auto const centre = (x + 0.5) * scale - 0.5;
                auto const left = static_cast<int>(std::floor(centre));
                auto const fraction = static_cast<float>(centre - left);
                out[x] = in[address(left, source.width, wrap)] * (1.0f - fraction)
                       + in[address(left + 1, source.width, wrap)] * fraction;
            }
        }
    }
    return result;
}

FloatImage transposed(FloatImage const &source) {
    auto result = FloatImage{source.height, source.width, std::vector<core::vec4>(source.texels.size())};
    for (auto y = 0; y < source.height; ++y) {
        for (auto x = 0; x < source.width; ++x) {
            result.texels[static_cast<std::size_t>(x) * static_cast<std::size_t>(source.height)
                          + static_cast<std::size_t>(y)] =
                source.texels[static_cast<std::size_t>(y) * static_cast<std::size_t>(source.width)
                              + static_cast<std::size_t>(x)];
        }
    }
    return result;
}

// Level 0 of `image` as RGBA8 texels.
std::vector<std::uint8_t> rgba8LevelZero(core::ImageData const &image) {
    if (not image.compressed) {
        return image.levels[0];
    }
    auto const count = static_cast<std::size_t>(image.extent.x) * static_cast<std::size_t>(image.extent.y);
    auto decoded = std::vector<tc::cpu::RGBA32F>(count);
    tc::decodeCompressed(*image.compressed, image.extent, image.levels[0].data(), image.levels[0].size(),
                         decoded.data());
    auto texels = std::vector<std::uint8_t>(count * 4);
    for (auto index = std::size_t{0}; index < count; ++index) {
        for (auto channel = std::size_t{0}; channel < 4; ++channel) {
            texels[index * 4 + channel] = encodeChannel(decoded[index].channels[channel], false);
        }
    }
    return texels;
}

// `image` as an RGBA8 image `size` texels square with one level, resampled in linear colour.
core::ImageHandle colorImageOf(core::ImageHandle const &image, std::uint32_t const size,
                               core::TextureSampler const &sampler) {
    auto const side = static_cast<int>(size);
    if (not image->compressed and image->extent == core::ivec2{side, side}) {
        return image;
    }
    auto const srgb = image->encoding == core::ColorEncoding::Srgb;
    auto const texels = rgba8LevelZero(*image);
    auto source = FloatImage{image->extent.x, image->extent.y, std::vector<core::vec4>(texels.size() / 4)};
    for (auto index = std::size_t{0}; index < source.texels.size(); ++index) {
        source.texels[index] = core::vec4{decodeChannel(texels[index * 4], srgb), decodeChannel(texels[index * 4 + 1], srgb),
                                          decodeChannel(texels[index * 4 + 2], srgb),
                                          decodeChannel(texels[index * 4 + 3], false)};
    }
    auto const resampled =
        transposed(resampleRows(transposed(resampleRows(source, side, sampler.wrapU)), side, sampler.wrapV));
    auto result = core::ImageData{};
    result.extent = core::ivec2{side, side};
    result.encoding = image->encoding;
    auto &level = result.levels.emplace_back(resampled.texels.size() * 4);
    for (auto index = std::size_t{0}; index < resampled.texels.size(); ++index) {
        auto const &texel = resampled.texels[index];
        level[index * 4] = encodeChannel(texel.x, srgb);
        level[index * 4 + 1] = encodeChannel(texel.y, srgb);
        level[index * 4 + 2] = encodeChannel(texel.z, srgb);
        level[index * 4 + 3] = encodeChannel(texel.w, false);
    }
    return std::make_shared<core::ImageData const>(std::move(result));
}

std::size_t compressedChainBytes(tc::CompressedFormat const format, std::uint32_t const size) {
    auto bytes = std::size_t{0};
    auto const extent = core::ivec2{static_cast<int>(size), static_cast<int>(size)};
    for (auto level = std::uint32_t{0}; level < core::fullLevelCount(extent); ++level) {
        bytes += tc::compressedLayerBytes(format, extent, level);
    }
    return bytes;
}

} // namespace

TextureLimits textureLimitsOf(tc::DeviceLimits const &limits, std::uint32_t const maxTextureSize) {
    auto result = TextureLimits{};
    auto const largestPool = kSmallestColorPool << (colorPoolSlots() - 1);
    result.maxSize = std::min({floorPowerOfTwo(std::max(maxTextureSize, kSmallestColorPool)),
                               floorPowerOfTwo(std::max(limits.imageDimension2D, kSmallestColorPool)), largestPool});
    result.arrayLayers = std::clamp(limits.imageArrayLayers, kFirstColorLayers, kMaxLayers);
    result.compressedArrayFormats = limits.compressedArrayFormats;
    return result;
}

bool TextureLayout::fitsCompressedPool(core::ImageData const &image) const noexcept {
    if (not image.compressed or (limits.compressedArrayFormats & tc::compressedBit(*image.compressed)) == 0) {
        return false;
    }
    auto const size = static_cast<std::uint32_t>(image.extent.x);
    return image.extent.x == image.extent.y and isPowerOfTwo(size) and size <= limits.maxSize
       and image.levels.size() >= core::fullLevelCount(image.extent);
}

std::uint32_t TextureLayout::takeLayer(std::uint32_t const slot) {
    auto &pool = pools[slot];
    if (not pool.freed.empty()) {
        auto const layer = pool.freed.back();
        pool.freed.pop_back();
        return layer;
    }
    return pool.used++;
}

std::optional<std::uint32_t> TextureLayout::compressedSlot(tc::CompressedFormat const format, std::uint32_t const size,
                                                           SceneUpdate &update) {
    auto empty = std::optional<std::uint32_t>{};
    for (auto slot = colorPoolSlots(); slot < texturePoolSlots(); ++slot) {
        auto const &pool = pools[slot];
        if (pool.size == 0) {
            empty = empty ? empty : std::optional<std::uint32_t>{slot};
        } else if (pool.compressed == format and pool.size == size
                   and (not pool.freed.empty() or pool.used < pool.layers)) {
            return slot;
        }
    }
    if (not empty) {
        return std::nullopt;
    }
    auto const fitting = static_cast<std::uint32_t>(
        std::min<std::size_t>(kCompressedPoolBytes / compressedChainBytes(format, size), limits.arrayLayers));
    pools[*empty] = Pool{format, size, std::max(fitting, std::min(kFirstColorLayers, limits.arrayLayers)), 0, {}};
    update.pools.push_back(PoolChange{*empty, format, size, pools[*empty].layers});
    return empty;
}

std::optional<std::uint32_t> TextureLayout::colorSlot(std::uint32_t const size, SceneUpdate &update) {
    auto const slot = log2Of(size) - log2Of(kSmallestColorPool);
    auto &pool = pools[slot];
    if (pool.size == 0) {
        pool = Pool{std::nullopt, size, std::min(kFirstColorLayers, limits.arrayLayers), 0, {}};
    } else if (pool.freed.empty() and pool.used == pool.layers) {
        if (pool.layers == limits.arrayLayers) {
            return std::nullopt;
        }
        pool.layers = std::min(pool.layers * 2, limits.arrayLayers);
    } else {
        return slot;
    }
    update.pools.push_back(PoolChange{slot, std::nullopt, size, pool.layers});
    return slot;
}

void TextureLayout::place(core::TextureUpsert const &upsert, SceneUpdate &update) {
    remove(upsert.id);
    auto const &image = *upsert.image;
    auto const srgb = image.encoding == core::ColorEncoding::Srgb;
    auto const &sampler = upsert.sampler;
    auto const refOf = [&](std::uint32_t const slot, std::uint32_t const layer, std::uint32_t const size) {
        return textureRef(slot, layer, log2Of(size), srgb, wrapCode(sampler.wrapU), wrapCode(sampler.wrapV));
    };
    if (fitsCompressedPool(image)) {
        auto const size = static_cast<std::uint32_t>(image.extent.x);
        if (auto const slot = compressedSlot(*image.compressed, size, update)) {
            auto const layer = takeLayer(*slot);
            locations[upsert.id] = TextureLocation{*slot, layer};
            update.textureUploads.push_back(
                TextureUpload{upsert.id, refOf(*slot, layer, size), upsert.image, core::fullLevelCount(image.extent)});
            return;
        }
    }
    auto const longest = static_cast<std::uint32_t>(std::max(image.extent.x, image.extent.y));
    auto const size = std::clamp(ceilPowerOfTwo(longest), kSmallestColorPool, limits.maxSize);
    auto const slot = colorSlot(size, update);
    if (not slot) {
        return;
    }
    auto const layer = takeLayer(*slot);
    locations[upsert.id] = TextureLocation{*slot, layer};
    update.textureUploads.push_back(
        TextureUpload{upsert.id, refOf(*slot, layer, size), colorImageOf(upsert.image, size, sampler), 1});
}

void TextureLayout::remove(core::TextureId const id) {
    auto const found = locations.find(id);
    if (found == locations.end()) {
        return;
    }
    pools[found->second.slot].freed.push_back(found->second.layer);
    locations.erase(found);
}

void TextureLayout::reset() noexcept {
    pools = {};
    locations.clear();
}

std::optional<TextureLocation> TextureLayout::locationOf(core::TextureId const id) const {
    auto const found = locations.find(id);
    if (found == locations.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::vector<core::TextureId> TextureLayout::placed() const {
    auto ids = std::vector<core::TextureId>{};
    ids.reserve(locations.size());
    for (auto const &[id, location] : locations) {
        ids.push_back(id);
    }
    return ids;
}

} // namespace lr::render
