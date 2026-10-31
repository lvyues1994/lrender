#include "Ktx2.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

namespace lr::asset {

namespace {

constexpr auto kIdentifier =
    std::array<std::uint8_t, 12>{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
constexpr auto kHeaderBytes = std::size_t{80};
constexpr auto kLevelEntryBytes = std::size_t{24};
constexpr auto kLargestSide = std::uint32_t{1} << 15;

// Header fields, little-endian, as offsets into the file.
constexpr auto kVkFormat = std::size_t{12};
constexpr auto kPixelWidth = std::size_t{20};
constexpr auto kPixelHeight = std::size_t{24};
constexpr auto kPixelDepth = std::size_t{28};
constexpr auto kLayerCount = std::size_t{32};
constexpr auto kFaceCount = std::size_t{36};
constexpr auto kLevelCount = std::size_t{40};
constexpr auto kSupercompression = std::size_t{44};

constexpr auto kVkR8G8B8A8Unorm = std::uint32_t{37};
constexpr auto kVkR8G8B8A8Srgb = std::uint32_t{43};

struct BlockFormat {
    std::uint32_t vkFormat;
    tc::CompressedFormat format;
};

// The VK_FORMAT_*_BLOCK values of the formats tc decodes, UNORM and SRGB alike.
constexpr auto kBlockFormats = std::array<BlockFormat, 30>{{
    {131, tc::CompressedFormat::Bc1},        {132, tc::CompressedFormat::Bc1},
    {133, tc::CompressedFormat::Bc1},        {134, tc::CompressedFormat::Bc1},
    {137, tc::CompressedFormat::Bc3},        {138, tc::CompressedFormat::Bc3},
    {139, tc::CompressedFormat::Bc4},        {141, tc::CompressedFormat::Bc5},
    {145, tc::CompressedFormat::Bc7},        {146, tc::CompressedFormat::Bc7},
    {147, tc::CompressedFormat::Etc2Rgb8},   {148, tc::CompressedFormat::Etc2Rgb8},
    {149, tc::CompressedFormat::Etc2Rgb8A1}, {150, tc::CompressedFormat::Etc2Rgb8A1},
    {151, tc::CompressedFormat::Etc2Rgba8},  {152, tc::CompressedFormat::Etc2Rgba8},
    {153, tc::CompressedFormat::EacR11},     {155, tc::CompressedFormat::EacRg11},
    {157, tc::CompressedFormat::Astc4x4},    {158, tc::CompressedFormat::Astc4x4},
    {161, tc::CompressedFormat::Astc5x5},    {162, tc::CompressedFormat::Astc5x5},
    {165, tc::CompressedFormat::Astc6x6},    {166, tc::CompressedFormat::Astc6x6},
    {171, tc::CompressedFormat::Astc8x8},    {172, tc::CompressedFormat::Astc8x8},
    {179, tc::CompressedFormat::Astc10x10},  {180, tc::CompressedFormat::Astc10x10},
    {183, tc::CompressedFormat::Astc12x12},  {184, tc::CompressedFormat::Astc12x12},
}};

std::uint64_t littleEndian(ByteView const bytes, std::size_t const offset, std::size_t const width) noexcept {
    auto value = std::uint64_t{0};
    for (auto index = std::size_t{0}; index < width; ++index) {
        value |= std::uint64_t{bytes.data[offset + index]} << (8u * index);
    }
    return value;
}

std::uint32_t u32At(ByteView const bytes, std::size_t const offset) noexcept {
    return static_cast<std::uint32_t>(littleEndian(bytes, offset, 4));
}

std::uint64_t u64At(ByteView const bytes, std::size_t const offset) noexcept { return littleEndian(bytes, offset, 8); }

char const *supercompressionName(std::uint32_t const scheme) noexcept {
    switch (scheme) {
    case 1:
        return "BasisLZ";
    case 2:
        return "Zstandard";
    case 3:
        return "ZLIB";
    default:
        return "an unknown scheme";
    }
}

AssetResult<std::optional<tc::CompressedFormat>> formatOf(std::uint32_t const vkFormat) {
    if (vkFormat == kVkR8G8B8A8Unorm or vkFormat == kVkR8G8B8A8Srgb) {
        return std::optional<tc::CompressedFormat>{};
    }
    auto const found = std::find_if(kBlockFormats.begin(), kBlockFormats.end(),
                                    [vkFormat](BlockFormat const &entry) { return entry.vkFormat == vkFormat; });
    if (found != kBlockFormats.end()) {
        return std::optional<tc::CompressedFormat>{found->format};
    }
    if (vkFormat == 0) {
        return assetError(AssetErrorKind::Unsupported,
                          "KTX2 with Basis Universal (UASTC) data; lrender reads RGBA8 and block formats");
    }
    return assetError(AssetErrorKind::Unsupported, "KTX2 vkFormat " + std::to_string(vkFormat)
                                                       + ", which is neither RGBA8 nor a block format lrender decodes");
}

AssetResult<core::ivec2> extentOf(ByteView const bytes) {
    auto const width = u32At(bytes, kPixelWidth);
    auto const height = u32At(bytes, kPixelHeight);
    if (u32At(bytes, kPixelDepth) != 0 or height == 0) {
        return assetError(AssetErrorKind::Unsupported, "a 1D or 3D KTX2 texture; lrender loads 2D ones");
    }
    if (u32At(bytes, kLayerCount) > 1 or u32At(bytes, kFaceCount) != 1) {
        return assetError(AssetErrorKind::Unsupported, "a KTX2 array or cube map; lrender loads single 2D images");
    }
    if (width == 0 or width > kLargestSide or height > kLargestSide) {
        return assetError(AssetErrorKind::Corrupt, "a KTX2 image of " + std::to_string(width) + "x"
                                                       + std::to_string(height) + " texels");
    }
    return core::ivec2{static_cast<int>(width), static_cast<int>(height)};
}

} // namespace

bool isKtx2(ByteView const bytes) noexcept {
    return bytes.size >= kIdentifier.size() and std::equal(kIdentifier.begin(), kIdentifier.end(), bytes.data);
}

AssetResult<Ktx2Info> readKtx2Info(ByteView const bytes) {
    if (not isKtx2(bytes) or bytes.size < kHeaderBytes) {
        return assetError(AssetErrorKind::Corrupt, "not a KTX2 file, or shorter than its header");
    }
    if (auto const scheme = u32At(bytes, kSupercompression); scheme != 0) {
        return assetError(AssetErrorKind::Unsupported, std::string{"KTX2 supercompressed with "}
                                                           + supercompressionName(scheme)
                                                           + "; lrender reads levels stored as they are");
    }
    auto const format = formatOf(u32At(bytes, kVkFormat));
    if (not format) {
        return tl::make_unexpected(format.error());
    }
    auto const extent = extentOf(bytes);
    if (not extent) {
        return tl::make_unexpected(extent.error());
    }
    auto const levels = std::max(u32At(bytes, kLevelCount), 1u);
    if (levels > core::fullLevelCount(*extent) or bytes.size < kHeaderBytes + levels * kLevelEntryBytes) {
        return assetError(AssetErrorKind::Corrupt, "a KTX2 level index that does not fit the image or the file");
    }
    return Ktx2Info{*format, *extent, levels};
}

AssetResult<core::ImageData> readKtx2(ByteView const bytes, core::ColorEncoding const encoding) {
    auto const info = readKtx2Info(bytes);
    if (not info) {
        return tl::make_unexpected(info.error());
    }
    auto image = core::ImageData{};
    image.extent = info->extent;
    image.encoding = encoding;
    image.compressed = info->format;
    for (auto level = std::uint32_t{0}; level < info->levels; ++level) {
        auto const entry = kHeaderBytes + level * kLevelEntryBytes;
        auto const offset = u64At(bytes, entry);
        auto const length = u64At(bytes, entry + 8);
        auto const expected = core::levelBytes(image, level);
        if (offset > bytes.size or length > bytes.size - offset or length != expected) {
            return assetError(AssetErrorKind::Corrupt, "KTX2 level " + std::to_string(level) + " is "
                                                           + std::to_string(length) + " bytes at "
                                                           + std::to_string(offset) + "; it needs "
                                                           + std::to_string(expected) + " inside the file");
        }
        auto const *const first = bytes.data + offset;
        image.levels.emplace_back(first, first + length);
    }
    return image;
}

} // namespace lr::asset
