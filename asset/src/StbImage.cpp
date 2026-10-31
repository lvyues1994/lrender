#include "ImageFiles.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include <stb_image.h>

#include <climits>
#include <memory>
#include <string>

namespace lr::asset {

namespace {

struct StbFree {
    void operator()(stbi_uc *const pixels) const noexcept { stbi_image_free(pixels); }
};

} // namespace

AssetResult<core::ImageData> decodePngOrJpeg(ByteView const bytes, core::ColorEncoding const encoding) {
    if (bytes.size > static_cast<std::size_t>(INT_MAX)) {
        return assetError(AssetErrorKind::Unsupported, "an image file larger than 2 GiB");
    }
    auto width = 0;
    auto height = 0;
    auto channels = 0;
    auto const pixels = std::unique_ptr<stbi_uc, StbFree>{
        stbi_load_from_memory(bytes.data, static_cast<int>(bytes.size), &width, &height, &channels, STBI_rgb_alpha)};
    if (pixels == nullptr) {
        return assetError(AssetErrorKind::Corrupt, std::string{"not a PNG, JPEG or KTX2 image ("} + stbi_failure_reason()
                                                       + ")");
    }
    auto image = core::ImageData{};
    image.extent = core::ivec2{width, height};
    image.encoding = encoding;
    auto const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
    image.levels.emplace_back(pixels.get(), pixels.get() + count);
    return image;
}

} // namespace lr::asset
