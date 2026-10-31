#pragma once

// Asset files the tests make for themselves: images, KTX2 containers and glTF scenes.

#include <lr/asset/AssetSource.h>
#include <lr/core/Math.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr::test {

using asset::Bytes;

// RGBA8 texels, rows from the top.
Bytes pngOf(core::ivec2 extent, std::vector<std::uint8_t> const &rgba);
Bytes jpegOf(core::ivec2 extent, std::vector<std::uint8_t> const &rgba, int quality);
// `extent` texels all of one colour.
std::vector<std::uint8_t> solid(core::ivec2 extent, std::uint8_t r, std::uint8_t g, std::uint8_t b,
                                std::uint8_t a = 255);

struct Ktx2Desc {
    std::uint32_t vkFormat = 37; // VK_FORMAT_R8G8B8A8_UNORM
    core::ivec2 extent{1};
    std::vector<std::vector<std::uint8_t>> levels; // level 0 first
    std::uint32_t supercompression = 0;
};

inline constexpr std::uint32_t kVkBc1RgbaUnorm = 133;
inline constexpr std::uint32_t kVkEtc2Rgb8Unorm = 147;

// The levels stored smallest first, as KTX2 writers do; the level index lists level 0 first.
Bytes ktx2Of(Ktx2Desc const &desc);

// An 8x8 BC1 image of four levels, solid red, green, blue and white.
std::vector<std::vector<std::uint8_t>> bc1Levels();
// A 16x16 ETC2 RGB image of five levels, every block different.
std::vector<std::vector<std::uint8_t>> etc2Levels();

struct TestPrimitive {
    std::vector<core::vec3> positions;
    std::vector<core::vec3> normals;    // or none
    std::vector<core::vec2> uvs;        // or none
    std::vector<std::uint16_t> indices; // or none
    int material = -1;
};

// A glTF document: meshes of one primitive each and their buffer are written here, the
// rest is JSON given as it is ("[...]" or "{...}", empty to leave the key out).
struct TestGltf {
    std::vector<TestPrimitive> meshes;
    std::vector<Bytes> images; // PNG files, referred to as images 0, 1, ...
    std::string extensionsUsed;
    std::string extensions;
    std::string scenes = "[{\"nodes\":[0]}]";
    std::string nodes = "[{\"mesh\":0}]";
    std::string materials;
    std::string textures;
    std::string samplers;
    std::string cameras;
};

enum class GltfLayout : std::uint8_t {
    Embedded, // .gltf with data URIs
    External, // .gltf with a .bin and .png files beside it
    Binary,   // .glb with the images in its buffer
};

struct GltfFiles {
    std::string main; // the file to load
    std::unordered_map<std::string, Bytes> files;
};

// The files of `gltf`, named after `stem` (which may contain a directory).
GltfFiles writeGltf(TestGltf const &gltf, GltfLayout layout, std::string const &stem);

// A quad in the xy plane facing +z, half size `half`, its texture coordinates as
// core::makeQuad's, drawn with material 0.
TestPrimitive quadPrimitive(float half);

} // namespace lr::test
