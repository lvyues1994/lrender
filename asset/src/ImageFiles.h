#pragma once

#include "ByteView.h"
#include "Ktx2.h"

#include <lr/asset/AssetResult.h>
#include <lr/core/Image.h>

#include <cstdint>

namespace lr::asset {

// How images become what the renderer gets (AssetLibraryConfig).
struct ImagePolicy {
    std::uint32_t compressedFormats = 0;
    bool preferKtx2 = true;
};

// Whether the renderer puts the blocks of an image of this shape in a compressed pool
// (render's TextureLayout): a format it has arrays of, square, a power of two, every
// level. It decodes any other compressed image on its render thread.
bool keepsBlocks(Ktx2Info const &image, std::uint32_t compressedFormats) noexcept;

// The levels of a compressed image decoded to RGBA8 (tc::decodeCompressed).
core::ImageData decodedToRgba8(core::ImageData const &image);

// On the background pool. PNG and JPEG become RGBA8, KTX2 keeps its blocks where the
// renderer keeps them and is decoded to RGBA8 elsewhere.
AssetResult<core::ImageData> decodeImage(ByteView bytes, core::ColorEncoding encoding, ImagePolicy const &policy);

// stb_image's PNG and JPEG decoders, to RGBA8.
AssetResult<core::ImageData> decodePngOrJpeg(ByteView bytes, core::ColorEncoding encoding);

} // namespace lr::asset
