#include "AssetTestSupport.h"
#include "render/RenderTestSupport.h"

#include <lr/asset/SceneDelta.h>

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace lr;

constexpr auto kSize = 32;     // pixels across the view
constexpr auto kTexels = 16;   // across the texture: the quad covers 16 pixels, one per texel

// A quad of half size 1 seen by the file's camera 4 away, which fills the middle half of
// a square view (test::halfFillingCamera): pixel (8 + i, 8 + j) sees the centre of texel
// (i, j) of a 16-texel texture.
test::TestGltf quadScene(std::string const &material) {
    auto fov = std::array<char, 32>{};
    std::snprintf(fov.data(), fov.size(), "%.9g", 2.0 * std::atan(0.5));
    auto scene = test::TestGltf{};
    scene.meshes = {test::quadPrimitive(1.0f)};
    scene.nodes = R"([{"mesh":0},{"camera":0,"translation":[0,0,4]}])";
    scene.scenes = R"([{"nodes":[0,1]}])";
    scene.cameras = R"([{"type":"perspective","perspective":{"yfov":)" + std::string{fov.data()}
                  + R"(,"znear":0.05,"zfar":100}}])";
    scene.materials = "[" + material + "]";
    return scene;
}

tc::ivec4 patternTexel(int const x, int const y) { return tc::ivec4{x * 16, y * 16, 128, 255}; }

test::Bytes patternPng() {
    auto texels = std::vector<std::uint8_t>{};
    for (auto y = 0; y < kTexels; ++y) {
        for (auto x = 0; x < kTexels; ++x) {
            auto const texel = patternTexel(x, y);
            texels.insert(texels.end(), {static_cast<std::uint8_t>(texel.x), static_cast<std::uint8_t>(texel.y),
                                         static_cast<std::uint8_t>(texel.z), static_cast<std::uint8_t>(texel.w)});
        }
    }
    return test::pngOf(core::ivec2{kTexels, kTexels}, texels);
}

asset::SceneHandle loadedScene(test::TestGltf const &gltf) {
    auto const files = test::writeGltf(gltf, test::GltfLayout::Binary, "scene");
    auto harness = test::AssetHarness{files.files};
    auto const loaded = test::loadNow(harness.library->loadScene(files.main));
    REQUIRE(loaded.has_value());
    REQUIRE((*loaded)->cameras.size() == 1);
    return *loaded;
}

// The scene through the renderer, on the second frame: textures stream in on the first.
render::ImageRgba8 rendered(test::RenderHarness &harness, asset::SceneData const &scene) {
    auto ids = asset::makeSequentialIds();
    auto delta = asset::sceneDelta(scene, *ids).delta;
    auto const target = harness.renderer->targets().createImage(render::ImageTargetDesc{core::ivec2{kSize, kSize}});
    auto const view = test::viewOf(target, scene.cameras[0].camera);
    harness.render(test::frameOf(std::move(delta), view));
    auto frame = harness.render(test::frameOf({}, view));
    REQUIRE(frame.views.at(0).image.has_value());
    return *frame.views[0].image;
}

// Inside the quad, the expected pixel; outside, the black sky.
template <class Expected>
std::uint32_t largestError(render::ImageRgba8 const &image, Expected const &expected) {
    auto largest = std::uint32_t{0};
    for (auto y = 0; y < kSize; ++y) {
        for (auto x = 0; x < kSize; ++x) {
            auto const inside = x >= kSize / 4 and x < 3 * kSize / 4 and y >= kSize / 4 and y < 3 * kSize / 4;
            auto const want = inside ? expected(x - kSize / 4, y - kSize / 4) : test::rgba8(core::vec3{0.0f});
            auto const pixel = test::pixelAt(image, x, y);
            for (auto channel = 0u; channel < 3u; ++channel) {
                largest = std::max(largest, test::channelDifference(pixel, want, channel));
            }
        }
    }
    return largest;
}

// Each backend against the expected pixels, and the GPU backends against the CPU.
template <class Expected>
void checkOnEveryBackend(asset::SceneData const &scene, Expected const &expected) {
    auto images = std::map<std::string, render::ImageRgba8>{};
    test::forEachBackend(test::HarnessOptions{}, [&](test::RenderHarness &harness) {
        auto const image = rendered(harness, scene);
        CHECK(largestError(image, expected) <= 1);
        images.emplace(harness.renderer->deviceInfo().backend, image);
    });
    REQUIRE(images.count("cpu") == 1);
    for (auto const &entry : images) {
        auto const &backend = entry.first;
        CAPTURE(backend);
        CHECK(test::maxChannelDifference(entry.second, images.at("cpu")) <= 1);
    }
}

TEST_CASE("a glTF quad that glows renders its emission through the file's camera") {
    auto const scene = loadedScene(quadScene(
        R"({"pbrMetallicRoughness":{"baseColorFactor":[0,0,0,1]},"emissiveFactor":[0.25,0.5,0.75]})"));

    checkOnEveryBackend(*scene, [](int, int) { return test::rgba8(core::vec3{0.25f, 0.5f, 0.75f}); });
}

TEST_CASE("a glTF quad with an emissive PNG texture shows each texel at the pixel that sees its centre") {
    auto gltf = quadScene(
        R"({"pbrMetallicRoughness":{"baseColorFactor":[0,0,0,1]},"emissiveFactor":[1,1,1],"emissiveTexture":{"index":0}})");
    gltf.images = {patternPng()};
    gltf.textures = R"([{"source":0}])";
    auto const scene = loadedScene(gltf);

    checkOnEveryBackend(*scene, [](int const x, int const y) {
        auto const texel = patternTexel(x, y);
        auto const linear = [](int const value) { return test::srgbToLinear(static_cast<float>(value) / 255.0f); };
        return test::rgba8(core::vec3{linear(texel.x), linear(texel.y), linear(texel.z)});
    });
}

} // namespace
