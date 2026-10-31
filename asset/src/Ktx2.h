#pragma once

#include "ByteView.h"

#include <lr/asset/AssetResult.h>
#include <lr/core/Image.h>

#include <tc/compressed.hpp>

#include <cstdint>
#include <optional>

namespace lr::asset {

// What the header and level index of a KTX2 file say.
struct Ktx2Info {
    std::optional<tc::CompressedFormat> format; // nullopt: RGBA8
    core::ivec2 extent{0};
    std::uint32_t levels = 1;
};

bool isKtx2(ByteView bytes) noexcept;

// A 2D image of one layer and face, RGBA8 (VK_FORMAT_R8G8B8A8_*) or a block format tc
// knows, not supercompressed (Unsupported otherwise).
AssetResult<Ktx2Info> readKtx2Info(ByteView bytes);

// Level 0 first, each level's blocks as the file stores them.
AssetResult<core::ImageData> readKtx2(ByteView bytes, core::ColorEncoding encoding);

} // namespace lr::asset
