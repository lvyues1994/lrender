#include "AssetTestSupport.h"

#include <tc/compressed.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

using namespace lr;
using asset::AssetErrorKind;
using core::ColorEncoding;
using test::AssetHarness;
using test::AssetHarnessOptions;
using test::loadNow;

using Levels = std::vector<std::vector<std::uint8_t>>;

std::vector<std::uint8_t> pattern(core::ivec2 const extent) {
    auto texels = std::vector<std::uint8_t>{};
    for (auto y = 0; y < extent.y; ++y) {
        for (auto x = 0; x < extent.x; ++x) {
            texels.insert(texels.end(), {static_cast<std::uint8_t>(40 * x + 10), static_cast<std::uint8_t>(90 * y + 5),
                                         static_cast<std::uint8_t>(17 * (x + y)), static_cast<std::uint8_t>(200 + x)});
        }
    }
    return texels;
}

// The levels as tc::decodeCompressed decodes them, rounded to RGBA8.
Levels decodedLevels(tc::CompressedFormat const format, core::ivec2 const extent, Levels const &levels) {
    auto result = Levels{};
    for (auto level = std::uint32_t{0}; level < levels.size(); ++level) {
        auto const size = core::levelExtent(extent, level);
        auto decoded = std::vector<tc::cpu::RGBA32F>(static_cast<std::size_t>(size.x * size.y));
        tc::decodeCompressed(format, size, levels[level].data(), levels[level].size(), decoded.data());
        auto &texels = result.emplace_back();
        for (auto const &texel : decoded) {
            for (auto const channel : texel.channels) {
                texels.push_back(static_cast<std::uint8_t>(std::lround(std::clamp(channel, 0.0f, 1.0f) * 255.0f)));
            }
        }
    }
    return result;
}

bool isCompressedAs(core::ImageHandle const &image, tc::CompressedFormat const format) {
    return image->compressed == std::optional<tc::CompressedFormat>{format};
}

test::Ktx2Desc bc1Image() {
    auto desc = test::Ktx2Desc{};
    desc.vkFormat = test::kVkBc1RgbaUnorm;
    desc.extent = core::ivec2{8, 8};
    desc.levels = test::bc1Levels();
    return desc;
}

TEST_CASE("PNG images come back as their RGBA8 texels, in the encoding asked for") {
    auto const extent = core::ivec2{3, 2};
    auto harness = AssetHarness{{{"pattern.png", test::pngOf(extent, pattern(extent))}}};

    auto const colour = loadNow(harness.library->loadImage("pattern.png", ColorEncoding::Srgb));
    auto const data = loadNow(harness.library->loadImage("pattern.png", ColorEncoding::Linear));

    REQUIRE(colour.has_value());
    REQUIRE(data.has_value());
    auto const &image = **colour;
    auto const sized = image.extent == extent;
    CHECK(sized);
    CHECK(not image.compressed.has_value());
    CHECK(image.encoding == ColorEncoding::Srgb);
    REQUIRE(image.levels.size() == 1);
    auto const texels = image.levels[0] == pattern(extent);
    CHECK(texels);
    CHECK((*data)->encoding == ColorEncoding::Linear);
    auto const sameTexels = (*data)->levels == image.levels;
    CHECK(sameTexels);
}

TEST_CASE("JPEG images decode to RGBA8 close to what was written") {
    auto const extent = core::ivec2{16, 16};
    auto harness = AssetHarness{{{"solid.jpg", test::jpegOf(extent, test::solid(extent, 200, 100, 50), 100)}}};

    auto const loaded = loadNow(harness.library->loadImage("solid.jpg", ColorEncoding::Srgb));

    REQUIRE(loaded.has_value());
    auto const &texels = (*loaded)->levels.at(0);
    REQUIRE(texels.size() == 16 * 16 * 4);
    auto worst = 0;
    for (auto index = std::size_t{0}; index < texels.size(); index += 4) {
        worst = std::max({worst, std::abs(texels[index] - 200), std::abs(texels[index + 1] - 100),
                          std::abs(texels[index + 2] - 50), std::abs(texels[index + 3] - 255)});
    }
    CHECK(worst <= 3);
}

TEST_CASE("an uncompressed KTX2 file gives its levels, the largest first") {
    auto const levels = Levels{test::solid(core::ivec2{4, 4}, 255, 0, 0), test::solid(core::ivec2{2, 2}, 0, 255, 0),
                               test::solid(core::ivec2{1, 1}, 0, 0, 255)};
    auto desc = test::Ktx2Desc{};
    desc.extent = core::ivec2{4, 4};
    desc.levels = levels;
    auto harness = AssetHarness{{{"levels.ktx2", test::ktx2Of(desc)}}};

    auto const loaded = loadNow(harness.library->loadImage("levels.ktx2", ColorEncoding::Linear));

    REQUIRE(loaded.has_value());
    CHECK(not (*loaded)->compressed.has_value());
    auto const sized = (*loaded)->extent == core::ivec2{4, 4};
    CHECK(sized);
    auto const inOrder = (*loaded)->levels == levels;
    CHECK(inOrder);
}

TEST_CASE("compressed KTX2 levels pass through where the renderer keeps the format, in level order") {
    auto etc2 = test::Ktx2Desc{};
    etc2.vkFormat = test::kVkEtc2Rgb8Unorm;
    etc2.extent = core::ivec2{16, 16};
    etc2.levels = test::etc2Levels();
    for (auto const &[desc, format] :
         {std::pair{bc1Image(), tc::CompressedFormat::Bc1}, std::pair{etc2, tc::CompressedFormat::Etc2Rgb8}}) {
        auto const name = std::string{tc::compressedFormatInfo(format).name};
        CAPTURE(name);
        auto options = AssetHarnessOptions{};
        options.compressedFormats = tc::compressedBit(format);
        auto harness = AssetHarness{{{"blocks.ktx2", test::ktx2Of(desc)}}, options};

        auto const loaded = loadNow(harness.library->loadImage("blocks.ktx2", ColorEncoding::Srgb));

        REQUIRE(loaded.has_value());
        CHECK(isCompressedAs(*loaded, format));
        auto const sized = (*loaded)->extent == desc.extent;
        CHECK(sized);
        auto const inOrder = (*loaded)->levels == desc.levels;
        CHECK(inOrder);
    }
}

TEST_CASE("a compressed format the renderer has no arrays of is decoded to RGBA8 as tc decodes it") {
    auto const desc = bc1Image();
    auto options = AssetHarnessOptions{};
    options.compressedFormats = tc::compressedBit(tc::CompressedFormat::Etc2Rgb8);
    auto harness = AssetHarness{{{"blocks.ktx2", test::ktx2Of(desc)}}, options};

    auto const loaded = loadNow(harness.library->loadImage("blocks.ktx2", ColorEncoding::Srgb));

    REQUIRE(loaded.has_value());
    CHECK(not (*loaded)->compressed.has_value());
    CHECK((*loaded)->encoding == ColorEncoding::Srgb);
    auto const expected = decodedLevels(tc::CompressedFormat::Bc1, desc.extent, desc.levels);
    auto const decoded = (*loaded)->levels == expected;
    CHECK(decoded);
    CHECK(expected[0][0] == 255); // level 0 is red
    CHECK(expected[1][1] == 255); // level 1 green
}

TEST_CASE("an image with a KTX2 variant loads the variant only where the renderer keeps its blocks") {
    auto const png = test::pngOf(core::ivec2{2, 2}, test::solid(core::ivec2{2, 2}, 10, 20, 30));
    auto const ktx2 = test::ktx2Of(bc1Image());
    auto const files =
        std::unordered_map<std::string, test::Bytes>{{"textures/wood.png", png}, {"textures/wood.ktx2", ktx2}};
    auto options = AssetHarnessOptions{};
    options.preferKtx2 = true;

    SUBCASE("kept: the blocks") {
        options.compressedFormats = tc::compressedBit(tc::CompressedFormat::Bc1);
        auto harness = AssetHarness{files, options};
        auto const loaded = loadNow(harness.library->loadImage("textures/wood.png", ColorEncoding::Srgb));
        REQUIRE(loaded.has_value());
        CHECK(isCompressedAs(*loaded, tc::CompressedFormat::Bc1));
    }
    SUBCASE("not kept: the PNG, not the decoded blocks") {
        auto harness = AssetHarness{files, options};
        auto const loaded = loadNow(harness.library->loadImage("textures/wood.png", ColorEncoding::Srgb));
        REQUIRE(loaded.has_value());
        auto const fromPng = (*loaded)->levels.at(0) == test::solid(core::ivec2{2, 2}, 10, 20, 30);
        CHECK(fromPng);
    }
    SUBCASE("without the PNG: the decoded blocks") {
        auto harness = AssetHarness{{{"textures/wood.ktx2", ktx2}}, options};
        auto const loaded = loadNow(harness.library->loadImage("textures/wood.png", ColorEncoding::Srgb));
        REQUIRE(loaded.has_value());
        CHECK(not (*loaded)->compressed.has_value());
        CHECK((*loaded)->extent.x == 8);
    }
    SUBCASE("variants turned off: the PNG") {
        options.preferKtx2 = false;
        options.compressedFormats = tc::compressedBit(tc::CompressedFormat::Bc1);
        auto harness = AssetHarness{files, options};
        auto const loaded = loadNow(harness.library->loadImage("textures/wood.png", ColorEncoding::Srgb));
        REQUIRE(loaded.has_value());
        CHECK((*loaded)->extent.x == 2);
        CHECK(harness.testSource().readsOf("textures/wood.ktx2") == 0);
    }
}

TEST_CASE("missing, corrupt and unsupported images come back as AssetError values") {
    auto valid = test::Ktx2Desc{};
    valid.extent = core::ivec2{2, 2};
    valid.levels = {test::solid(core::ivec2{2, 2}, 1, 2, 3)};
    auto truncated = test::ktx2Of(valid);
    truncated.resize(truncated.size() - 4);
    auto zstandard = valid;
    zstandard.supercompression = 2;
    auto basis = valid;
    basis.vkFormat = 0;
    auto harness = AssetHarness{{{"garbage.png", test::Bytes{1, 2, 3, 4, 5, 6, 7, 8}},
                                 {"truncated.ktx2", truncated},
                                 {"zstd.ktx2", test::ktx2Of(zstandard)},
                                 {"basis.ktx2", test::ktx2Of(basis)}}};
    auto &library = *harness.library;

    auto const missing = loadNow(library.loadImage("missing.png", ColorEncoding::Srgb));
    auto const garbage = loadNow(library.loadImage("garbage.png", ColorEncoding::Srgb));
    auto const cut = loadNow(library.loadImage("truncated.ktx2", ColorEncoding::Srgb));
    auto const supercompressed = loadNow(library.loadImage("zstd.ktx2", ColorEncoding::Srgb));
    auto const universal = loadNow(library.loadImage("basis.ktx2", ColorEncoding::Srgb));

    CHECK(test::isError(missing, AssetErrorKind::NotFound));
    CHECK(test::isError(garbage, AssetErrorKind::Corrupt));
    CHECK(test::isError(cut, AssetErrorKind::Corrupt));
    CHECK(test::isError(supercompressed, AssetErrorKind::Unsupported));
    CHECK(test::isError(universal, AssetErrorKind::Unsupported));
    REQUIRE(not supercompressed);
    CAPTURE(supercompressed.error().message);
    CHECK(supercompressed.error().message.find("Zstandard") != std::string::npos);
}

} // namespace
