#include "RenderTestSupport.h"
#include "TextureLayout.h"

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>
#include <lr/render/kernels/Lighting.h>

#include <tc/compressed.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

core::MeshId const kQuad{0};
core::MaterialId const kQuadMaterial{0};
core::TextureId const kTexture{1};

core::RenderWorldDelta quadWith(core::MaterialDesc const &material) {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{kQuad, test::share(core::makeQuad(core::vec2{1.0f}))}};
    delta.materials = {{kQuadMaterial, material}};
    delta.instances = {{core::InstanceId{0}, kQuad, kQuadMaterial, core::mat4{1.0f}}};
    return delta;
}

// A view `size` pixels square of the quad filling its middle half, read back in HDR: the
// quad's pixels, rows from the top, where pixel (i, j) sees texture coordinates
// ((i + 0.5) / n, (j + 0.5) / n) for n = size / 2.
struct QuadView {
    QuadView(RenderHarness &harness_, int const size_) : harness(&harness_), size(size_) {
        auto &pipelines = harness->renderer->pipelines();
        auto const target = harness->renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{size, size}});
        view = test::viewOf(target, test::halfFillingCamera());
        view.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {probe}});
    }

    // The frame after `delta`'s, by which the textures it uploads have arrived.
    std::vector<core::vec4> renderUploaded(core::RenderWorldDelta delta) {
        render(std::move(delta));
        return render({});
    }

    // A frame later still, by which textures the renderer decodes or resamples have arrived.
    std::vector<core::vec4> renderConverted(core::RenderWorldDelta delta) {
        renderUploaded(std::move(delta));
        return render({});
    }

    std::vector<core::vec4> render(core::RenderWorldDelta delta) {
        last = harness->render(test::frameOf(std::move(delta), view));
        auto const quarter = size / 4;
        auto pixels = std::vector<core::vec4>{};
        for (auto y = quarter; y < size - quarter; ++y) {
            for (auto x = quarter; x < size - quarter; ++x) {
                pixels.push_back((*probe->pixels)[static_cast<std::size_t>(y * size + x)]);
            }
        }
        return pixels;
    }

    RenderHarness *harness;
    int size;
    std::shared_ptr<test::HdrProbe> probe = std::make_shared<test::HdrProbe>();
    render::ViewDesc view;
    render::FrameResult last;
};

float largestDifference(std::vector<core::vec4> const &a, std::vector<core::vec4> const &b) {
    auto largest = 0.0f;
    for (auto index = std::size_t{0}; index < a.size(); ++index) {
        for (auto c = 0; c < 3; ++c) {
            largest = std::max(largest, std::abs(a[index][c] - b[index][c]));
        }
    }
    return largest;
}

core::vec4 linearOf(tc::ivec4 const &texel, bool const srgb) {
    auto const channel = [srgb](int const value) {
        auto const encoded = static_cast<float>(value) / 255.0f;
        return srgb ? test::srgbToLinear(encoded) : encoded;
    };
    return core::vec4{channel(texel.x), channel(texel.y), channel(texel.z), static_cast<float>(texel.w) / 255.0f};
}

tc::ivec4 patternTexel(int const x, int const y) { return tc::ivec4{x * 8, y * 8, (x + y) * 4, 255}; }

tc::ivec4 fillerTexel(int const x, int const y) { return tc::ivec4{x, y, 0, 255}; }

// With a budget of one 32x32 RGBA8 level per frame, the texture behind three others of
// that size is copied during the fourth frame and shows from the fifth. Until then the
// quad shows the factor.
TEST_CASE("a texture streams in over several frames, shows its factor until then, and the same texels on every "
          "backend") {
    constexpr auto kSize = 64;
    constexpr auto kArrivalFrame = 4;
    auto options = test::HarnessOptions{};
    options.textureStreamingBytes = 32 * 32 * 4;
    auto expected = std::vector<core::vec4>{};
    for (auto y = 0; y < kSize / 2; ++y) {
        for (auto x = 0; x < kSize / 2; ++x) {
            expected.push_back(linearOf(patternTexel(x, y), true));
        }
    }
    auto reference = std::vector<core::vec4>{};
    test::forEachBackend(options, [&](RenderHarness &harness) {
        auto view = QuadView{harness, kSize};
        auto delta = quadWith(test::emissiveTexture(kTexture));
        auto const extent = core::ivec2{kSize / 2, kSize / 2};
        for (auto const filler : {10u, 11u, 12u}) {
            delta.textures.push_back(
                {core::TextureId{filler}, test::rgba8Image(extent, core::ColorEncoding::Srgb, fillerTexel), {}});
        }
        delta.textures.push_back({kTexture, test::rgba8Image(extent, core::ColorEncoding::Srgb, patternTexel), {}});

        auto pixels = view.render(std::move(delta));
        for (auto frame = 1; frame <= kArrivalFrame; ++frame) {
            CAPTURE(frame);
            auto const factorOnly = std::all_of(pixels.begin(), pixels.end(), [](core::vec4 const &pixel) {
                return pixel.x == 1.0f and pixel.y == 1.0f and pixel.z == 1.0f;
            });
            CHECK(factorOnly);
            pixels = view.render({});
        }
        auto const textured = largestDifference(pixels, expected);
        CAPTURE(textured);
        CHECK(textured < 2e-3f);
        if (reference.empty()) {
            reference = pixels;
        }
        CHECK(largestDifference(pixels, reference) < 2e-3f);
    });
}

// Level 1 of a checkerboard of single black and white texels is grey: linear 0.5, sRGB
// 188, where averaging the encoded values would give linear 0.21.
TEST_CASE("mip levels are averaged in linear colour") {
    constexpr auto kSize = 64;
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        auto view = QuadView{harness, kSize};
        auto delta = quadWith(test::emissiveTexture(kTexture));
        delta.textures = {{kTexture, test::rgba8Image(core::ivec2{kSize, kSize}, core::ColorEncoding::Srgb,
                                                      [](int const x, int const y) {
                                                          auto const value = (x + y) % 2 == 0 ? 255 : 0;
                                                          return tc::ivec4{value, value, value, 255};
                                                      }),
                           {}}};

        auto const pixels = view.renderUploaded(std::move(delta));

        auto const grey = std::all_of(pixels.begin(), pixels.end(), [](core::vec4 const &pixel) {
            return std::abs(pixel.x - 0.5f) < 0.01f and std::abs(pixel.y - 0.5f) < 0.01f
               and std::abs(pixel.z - 0.5f) < 0.01f;
        });
        CHECK(grey);
    });
}

std::array<std::uint8_t, 8> bc1Block(std::uint16_t const colour0, std::uint16_t const colour1, std::uint32_t const indices) {
    return {static_cast<std::uint8_t>(colour0 & 0xFFu),  static_cast<std::uint8_t>(colour0 >> 8u),
            static_cast<std::uint8_t>(colour1 & 0xFFu),  static_cast<std::uint8_t>(colour1 >> 8u),
            static_cast<std::uint8_t>(indices & 0xFFu),  static_cast<std::uint8_t>((indices >> 8u) & 0xFFu),
            static_cast<std::uint8_t>((indices >> 16u) & 0xFFu), static_cast<std::uint8_t>(indices >> 24u)};
}

void appendBlock(std::vector<std::uint8_t> &level, std::array<std::uint8_t, 8> const &block) {
    level.insert(level.end(), block.begin(), block.end());
}

// An 8x8 BC1 image whose levels are solid red, green, blue and white: the colour seen
// tells the level of detail, and between levels how far trilinear filtering went.
core::ImageHandle levelColouredImage() {
    auto image = core::ImageData{};
    image.extent = core::ivec2{8, 8};
    image.encoding = core::ColorEncoding::Linear;
    image.compressed = tc::CompressedFormat::Bc1;
    auto const colours = std::array<std::uint16_t, 4>{0xF800u, 0x07E0u, 0x001Fu, 0xFFFFu};
    for (auto level = 0u; level < 4u; ++level) {
        auto &blocks = image.levels.emplace_back();
        auto const count = level == 0 ? 4 : 1;
        for (auto block = 0; block < count; ++block) {
            appendBlock(blocks, bc1Block(colours[level], colours[level], 0u));
        }
    }
    return test::share(std::move(image));
}

// The level of detail a colour of levelColouredImage() was sampled at.
float levelOf(core::vec4 const &colour) {
    if (colour.z < 0.01f) {
        return colour.y;
    }
    if (colour.x < 0.01f) {
        return 1.0f + colour.z;
    }
    return 2.0f + colour.x;
}

core::MaterialDesc levelColoured(float const repeats) {
    auto material = test::emissiveTexture(kTexture);
    material.uvTransform.scale = core::vec2{repeats};
    return material;
}

// Facing the camera, a pixel spans 1/32 of the quad: with the texture repeated r times,
// 8 r / 32 texels, so the level of detail is log2(r / 4).
TEST_CASE("the level of detail is the log of the texels a pixel spans, and grows along a receding plane") {
    constexpr auto kSize = 64;
    test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
        auto view = QuadView{harness, kSize};
        auto delta = quadWith(levelColoured(8.0f));
        delta.textures = {{kTexture, levelColouredImage(), {}}};
        view.render(std::exchange(delta, core::RenderWorldDelta{})); // the texture arrives in the next frame
        for (auto const &pair : {std::pair{8.0f, 1.0f}, std::pair{8.0f * std::sqrt(2.0f), 1.5f},
                                 std::pair{16.0f, 2.0f}, std::pair{4.0f * std::pow(2.0f, 2.25f), 2.25f}}) {
            auto const repeats = pair.first;
            auto const level = pair.second;
            CAPTURE(level);
            auto change = std::move(delta);
            change.materials = {{kQuadMaterial, levelColoured(repeats)}};
            auto const pixels = view.render(std::move(change));
            auto worst = 0.0f;
            for (auto const &pixel : pixels) {
                worst = std::max(worst, std::abs(levelOf(pixel) - level));
            }
            CAPTURE(worst);
            CHECK(worst < 0.03f);
            delta = core::RenderWorldDelta{};
        }

        // A plane seen from 1 above it: rows nearer the horizon see farther and coarser.
        auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
        auto const probe = std::make_shared<test::HdrProbe>();
        auto &pipelines = harness.renderer->pipelines();
        auto plane = test::viewOf(target, test::cameraAt(core::vec3{0.0f, 1.0f, 0.0f}, core::vec3{0.0f, 0.8f, -1.0f}));
        plane.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {probe}});
        auto ground = core::RenderWorldDelta{};
        ground.meshes = {{core::MeshId{1}, test::share(core::makePlane(core::vec2{400.0f}))}};
        ground.materials = {{core::MaterialId{1}, levelColoured(3200.0f)}};
        ground.instances = {{core::InstanceId{1}, core::MeshId{1}, core::MaterialId{1}, core::mat4{1.0f}}};
        ground.removed.instances = {core::InstanceId{0}};
        harness.render(test::frameOf(std::move(ground), plane));
        auto levels = std::vector<float>{};
        for (auto y = kSize - 1; y >= 0; --y) {
            auto const &pixel = (*probe->pixels)[static_cast<std::size_t>(y * kSize + kSize / 2)];
            if (pixel.w > 0.0f and (pixel.x + pixel.y + pixel.z) > 0.5f) {
                levels.push_back(levelOf(pixel));
            }
        }
        REQUIRE(levels.size() > 20);
        auto const rising = std::is_sorted(levels.begin(), levels.end(), [](float const a, float const b) {
            return a < b - 0.02f;
        });
        CAPTURE(levels.front());
        CAPTURE(levels.back());
        CHECK(rising);
        CHECK(levels.front() < 0.5f);
        CHECK(levels.back() > 2.5f);
    });
}

// ETC1 (and so ETC2) individual mode: two 4-bit base colours, a modifier table for each
// half and a 2-bit index per texel; how it decodes does not matter here, only that every
// backend decodes it alike.
std::array<std::uint8_t, 8> etcBlock(std::uint32_t const seed) {
    auto const high = (seed * 0x9E3779B1u) & 0xFFFFFFFCu; // individual mode, no flip
    auto const low = seed * 0x85EBCA6Bu + 0x7F4A7C15u;
    return {static_cast<std::uint8_t>(high >> 24u), static_cast<std::uint8_t>(high >> 16u),
            static_cast<std::uint8_t>(high >> 8u),  static_cast<std::uint8_t>(high),
            static_cast<std::uint8_t>(low >> 24u),  static_cast<std::uint8_t>(low >> 16u),
            static_cast<std::uint8_t>(low >> 8u),   static_cast<std::uint8_t>(low)};
}

// A 16x16 image with a full chain of hand-made blocks, all different.
core::ImageHandle handMadeImage(tc::CompressedFormat const format) {
    auto image = core::ImageData{};
    image.extent = core::ivec2{16, 16};
    image.encoding = core::ColorEncoding::Linear;
    image.compressed = format;
    auto seed = 1u;
    for (auto level = 0u; level < core::fullLevelCount(image.extent); ++level) {
        auto &blocks = image.levels.emplace_back();
        auto const count = core::levelBytes(image, level) / 8;
        for (auto block = std::size_t{0}; block < count; ++block, ++seed) {
            if (format == tc::CompressedFormat::Bc1) {
                // Four-colour mode (colour0 > colour1), every index in turn.
                auto const colour0 = static_cast<std::uint16_t>(0x8000u | (seed * 2654435761u >> 17u));
                auto const colour1 = static_cast<std::uint16_t>(colour0 - 1u - (seed * 40503u) % 0x7000u);
                appendBlock(blocks, bc1Block(colour0, colour1, seed * 0x9E3779B9u));
            } else {
                appendBlock(blocks, etcBlock(seed));
            }
        }
    }
    return test::share(std::move(image));
}

// Pixels see texel centres at level 0, so they show the decoded texels. GPUs interpolate
// BC1's colours with rounding of their own (tc measured up to 6/255 on NVIDIA); ETC2
// decodes alike everywhere. lavapipe has no ETC2, so Vulkan decodes it on the host into
// an RGBA8 pool.
TEST_CASE("hand-made BC1 and ETC2 textures sample to their decoded texels on every backend") {
    constexpr auto kSize = 32;
    for (auto const format : {tc::CompressedFormat::Bc1, tc::CompressedFormat::Etc2Rgb8}) {
        auto const formatName = std::string{tc::compressedFormatInfo(format).name};
        CAPTURE(formatName);
        auto const image = handMadeImage(format);
        auto decoded = std::vector<tc::cpu::RGBA32F>(16 * 16);
        tc::decodeCompressed(format, image->extent, image->levels[0].data(), image->levels[0].size(), decoded.data());
        auto expected = std::vector<core::vec4>{};
        for (auto const &texel : decoded) {
            expected.push_back(core::vec4{texel.channels[0], texel.channels[1], texel.channels[2], texel.channels[3]});
        }
        test::forEachBackend(test::HarnessOptions{}, [&](RenderHarness &harness) {
            auto view = QuadView{harness, kSize};
            auto delta = quadWith(test::emissiveTexture(kTexture));
            delta.textures = {{kTexture, image, {}}};

            auto const difference = largestDifference(view.renderConverted(std::move(delta)), expected);

            CAPTURE(difference);
            auto const tolerance = harness.renderer->deviceInfo().backend == "cpu" ? 1e-3f
                                 : format == tc::CompressedFormat::Bc1         ? 8.0f / 255.0f
                                                                                : 1.5f / 255.0f;
            CHECK(difference < tolerance);
        });
    }
}

TEST_CASE("a material whose texture is removed shows its factor alone") {
    auto harness = RenderHarness{};
    auto view = QuadView{harness, 16};
    auto delta = quadWith(test::emissiveTexture(kTexture, core::vec3{0.5f}));
    delta.textures = {{kTexture, test::rgba8Image(core::ivec2{16, 16}, core::ColorEncoding::Linear,
                                                  [](int, int) { return tc::ivec4{255, 0, 0, 255}; }),
                       {}}};
    CHECK(view.renderUploaded(std::move(delta)).front().y == 0.0f);

    auto removal = core::RenderWorldDelta{};
    removal.removed.textures = {kTexture};
    auto const pixels = view.render(std::move(removal));
    CHECK(pixels.front().x == 0.5f);
    CHECK(pixels.front().y == 0.5f);
}

TEST_CASE("a lost device lists the textures it held, which render again once upserted") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &harness) {
        auto view = QuadView{harness, 16};
        auto &pipelines = harness.renderer->pipelines();
        auto const lossy = pipelines.create(
            render::PipelineDesc{view.view.pipeline, {std::make_shared<test::DeviceLossInjector>(3)}});
        view.view.pipeline = lossy;
        auto const red = test::rgba8Image(core::ivec2{16, 16}, core::ColorEncoding::Linear,
                                          [](int, int) { return tc::ivec4{255, 0, 0, 255}; });
        auto delta = quadWith(test::emissiveTexture(kTexture));
        delta.textures = {{kTexture, red, {}}, {core::TextureId{2}, red, {}}};

        CHECK(view.renderUploaded(std::move(delta)).front().y == 0.0f);
        CHECK_THROWS_AS(view.render({}), tc::DeviceError);
        auto const factorOnly = view.render({});
        CHECK(view.last.lostTextures == std::vector<core::TextureId>{kTexture, core::TextureId{2}});
        CHECK(factorOnly.front().y == 1.0f);
        auto again = core::RenderWorldDelta{};
        again.textures = {{kTexture, red, {}}};
        CHECK(view.renderUploaded(std::move(again)).front().y == 0.0f);
        CHECK(view.last.lostTextures.empty());
    });
}

// A quad facing +z lit from above and in front; its normal map tilts the normal towards
// +Y in tangent space, up in the image, which is up on the quad whichever way u runs.
TEST_CASE("normal maps tilt the normal along the triangle's texture axes, alike in both pipelines") {
    constexpr auto kSize = 32;
    auto const tilted = core::vec3{0.0f, 0.6f, 0.8f};
    auto const encoded = (tilted * 0.5f + core::vec3{0.5f}) * 255.0f;
    auto const normals = test::rgba8Image(core::ivec2{16, 16}, core::ColorEncoding::Linear, [&encoded](int, int) {
        return tc::ivec4{static_cast<int>(std::lround(encoded.x)), static_cast<int>(std::lround(encoded.y)),
                           static_cast<int>(std::lround(encoded.z)), 255};
    });
    auto sun = core::DirectionalLight{};
    sun.direction = tc::normalize(core::vec3{0.0f, -1.0f, -1.0f});
    auto const toLight = -sun.direction;
    auto const expected = [&toLight](core::vec3 const &normal) {
        return render::kernels::reflectedLight(normal, core::vec3{0.0f, 0.0f, 1.0f}, toLight, core::vec3{1.0f},
                                               core::vec3{0.5f}, 0.0f, 1.0f)
            .x;
    };
    for (auto const mirrored : {false, true}) {
        CAPTURE(mirrored);
        auto material = core::MaterialDesc{};
        material.baseColor = core::vec4{0.5f, 0.5f, 0.5f, 1.0f};
        material.roughness = 1.0f;
        material.normalTexture = kTexture;
        material.uvTransform.scale = core::vec2{mirrored ? -1.0f : 1.0f, 1.0f};
        auto delta = quadWith(material);
        delta.textures = {{kTexture, normals, {}}};
        delta.lights = {{core::LightId{0}, sun}};
        auto harness = RenderHarness{};
        auto view = QuadView{harness, kSize};

        auto const centre =
            view.renderUploaded(std::move(delta))[static_cast<std::size_t>(kSize / 4 * kSize / 2 + kSize / 4)];

        auto const bumped = expected(tc::normalize(tilted));
        CAPTURE(centre.x);
        CAPTURE(bumped);
        CHECK(std::abs(centre.x - bumped) < 0.01f * bumped);
        CHECK(bumped > expected(core::vec3{0.0f, 0.0f, 1.0f}) * 1.3f);
        auto traced = view.view;
        traced.pipeline = harness.renderer->pipelines().pathTraceReference();
        auto const image = *harness.render(test::frameOf({}, traced)).views[0].image;
        CHECK(test::nearPixel(test::pixelAt(image, kSize / 2, kSize / 2), test::rgba8(core::vec3{centre.x})));
    }
}

render::TextureLimits withArraysOf(std::uint32_t const formats, std::uint32_t const maxSize = 2048) {
    auto limits = render::TextureLimits{};
    limits.maxSize = maxSize;
    limits.compressedArrayFormats = formats;
    return limits;
}

core::TextureUpsert solid(std::uint32_t const id, int const size) {
    return core::TextureUpsert{core::TextureId{id},
                               test::rgba8Image(core::ivec2{size, size}, core::ColorEncoding::Srgb,
                                                [](int, int) { return tc::ivec4{1, 2, 3, 4}; }),
                               {}};
}

TEST_CASE("a removed texture's layer goes to the next texture of its size, and full colour pools double") {
    auto const conversions = render::makeImmediateConversions();
    auto layout = render::TextureLayout{render::TextureLimits{}, *conversions};
    auto update = render::SceneUpdate{};
    for (auto id = 0u; id < 4u; ++id) {
        layout.place(solid(id, 32), update);
    }
    REQUIRE(update.pools.size() == 1);
    CHECK(update.pools[0].slot == 1); // 32 texels: the second colour pool
    CHECK(update.pools[0].layers == 4);
    CHECK(layout.locationOf(core::TextureId{3})->layer == 3);

    layout.remove(core::TextureId{1});
    layout.place(solid(4, 32), update);
    CHECK(layout.locationOf(core::TextureId{4})->layer == 1);
    CHECK(update.pools.size() == 1);
    CHECK(update.textureUploads.size() == 5);

    layout.place(solid(5, 32), update);
    REQUIRE(update.pools.size() == 2);
    CHECK(update.pools[1].layers == 8);
    CHECK(layout.locationOf(core::TextureId{5})->layer == 4);
    CHECK(not layout.locationOf(core::TextureId{1}).has_value());
}

// Decoded images upload with the frame after the one that placed them.
TEST_CASE("compressed images keep their blocks where the device has arrays of them, and are decoded elsewhere") {
    auto const bc1 = handMadeImage(tc::CompressedFormat::Bc1);
    auto const etc2 = handMadeImage(tc::CompressedFormat::Etc2Rgb8);
    auto const conversions = render::makeImmediateConversions();
    auto layout = render::TextureLayout{withArraysOf(tc::compressedBit(tc::CompressedFormat::Bc1)), *conversions};
    auto update = render::SceneUpdate{};

    layout.place(core::TextureUpsert{core::TextureId{0}, bc1, {}}, update);
    layout.place(core::TextureUpsert{core::TextureId{1}, etc2, {}}, update);
    layout.deliverConversions(update);
    auto next = render::SceneUpdate{};
    layout.deliverConversions(next);

    REQUIRE(update.textureUploads.size() == 1);
    auto const &blocks = update.textureUploads[0];
    CHECK(render::textureSlot(blocks.ref) >= render::colorPoolSlots());
    auto const unchanged = blocks.image == bc1;
    CHECK(unchanged);
    CHECK(blocks.levels == 5);
    REQUIRE(next.textureUploads.size() == 1);
    auto const &decoded = next.textureUploads[0];
    CHECK(render::textureSlot(decoded.ref) == 0);
    CHECK(not decoded.image->compressed.has_value());
    CHECK(decoded.levels == 1);
    REQUIRE(update.pools.size() == 2);
    CHECK(update.pools[0].compressed == tc::CompressedFormat::Bc1);
    CHECK(update.pools[0].size == 16);
}

TEST_CASE("images are resampled to their pool's square, larger ones scaled down in linear colour") {
    auto const conversions = render::makeImmediateConversions();
    auto layout = render::TextureLayout{withArraysOf(0, 16), *conversions};
    auto placing = render::SceneUpdate{};
    auto update = render::SceneUpdate{};
    auto const checker = test::rgba8Image(core::ivec2{32, 32}, core::ColorEncoding::Srgb, [](int const x, int const y) {
        auto const value = (x + y) % 2 == 0 ? 255 : 0;
        return tc::ivec4{value, value, value, 255};
    });
    auto const narrow = test::rgba8Image(core::ivec2{20, 12}, core::ColorEncoding::Linear,
                                         [](int, int) { return tc::ivec4{10, 20, 30, 40}; });

    layout.place(core::TextureUpsert{core::TextureId{0}, checker, {}}, placing);
    layout.place(core::TextureUpsert{core::TextureId{1}, narrow, {}}, placing);
    layout.deliverConversions(placing);
    layout.deliverConversions(update);

    CHECK(placing.textureUploads.empty());
    REQUIRE(update.textureUploads.size() == 2);
    auto const &scaled = *update.textureUploads[0].image;
    CHECK(scaled.extent == core::ivec2{16, 16});
    auto const grey = std::all_of(scaled.levels[0].begin(), scaled.levels[0].end(),
                                  [](std::uint8_t const value) { return value == 187 or value == 188 or value == 255; });
    CHECK(grey);
    auto const &stretched = *update.textureUploads[1].image;
    CHECK(stretched.extent == core::ivec2{16, 16});
    CHECK(stretched.levels[0][0] == 10);
    CHECK(stretched.levels[0][3] == 40);
}

// Conversions that finish when the test says so.
struct HeldConversions final : render::ImageConversions {
    std::future<core::ImageHandle> start(std::function<core::ImageHandle()> convert) override {
        held.push_back(Held{std::move(convert), std::promise<core::ImageHandle>{}});
        return held.back().result.get_future();
    }

    void finishAll() {
        for (auto &conversion : held) {
            conversion.result.set_value(conversion.convert());
        }
        held.clear();
    }

    struct Held {
        std::function<core::ImageHandle()> convert;
        std::promise<core::ImageHandle> result;
    };
    std::vector<Held> held;
};

TEST_CASE("as soon as ready, a texture uploads with the first frame after its conversion finished") {
    auto conversions = HeldConversions{};
    auto layout = render::TextureLayout{render::TextureLimits{}, conversions, render::ConversionDelivery::AsSoonAsReady};
    auto const narrow = test::rgba8Image(core::ivec2{20, 12}, core::ColorEncoding::Linear,
                                         [](int, int) { return tc::ivec4{10, 0, 0, 255}; });
    auto frames = std::vector<render::SceneUpdate>(4);
    layout.place(core::TextureUpsert{core::TextureId{0}, narrow, {}}, frames[0]);
    layout.deliverConversions(frames[0]);
    layout.deliverConversions(frames[1]);
    conversions.finishAll();
    layout.deliverConversions(frames[2]);
    layout.deliverConversions(frames[3]);

    CHECK(frames[0].textureUploads.empty());
    CHECK(frames[1].textureUploads.empty()); // the conversion still runs; the frame does not wait
    REQUIRE(frames[2].textureUploads.size() == 1);
    CHECK(frames[2].textureUploads[0].image->extent == core::ivec2{32, 32});
    CHECK(frames[3].textureUploads.empty());
}

TEST_CASE("a texture removed or placed again while it converts uploads what it is by then") {
    auto const conversions = render::makeImmediateConversions();
    auto layout = render::TextureLayout{render::TextureLimits{}, *conversions};
    auto const narrow = [](int const red) {
        return test::rgba8Image(core::ivec2{20, 12}, core::ColorEncoding::Linear,
                                [red](int, int) { return tc::ivec4{red, 0, 0, 255}; });
    };
    auto first = render::SceneUpdate{};
    layout.place(core::TextureUpsert{core::TextureId{0}, narrow(10), {}}, first);
    layout.place(core::TextureUpsert{core::TextureId{1}, narrow(20), {}}, first);
    layout.deliverConversions(first);

    auto second = render::SceneUpdate{};
    layout.remove(core::TextureId{0});
    layout.place(core::TextureUpsert{core::TextureId{1}, narrow(30), {}}, second);
    layout.deliverConversions(second);
    auto third = render::SceneUpdate{};
    layout.deliverConversions(third);

    CHECK(first.textureUploads.empty());
    CHECK(second.textureUploads.empty());
    REQUIRE(third.textureUploads.size() == 1);
    CHECK(third.textureUploads[0].id == core::TextureId{1});
    CHECK(third.textureUploads[0].image->levels[0][0] == 30);
}

} // namespace
