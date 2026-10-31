#include "TestFiles.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>

namespace lr::test {

namespace {

void appendTo(void *const context, void *const data, int const size) {
    auto &bytes = *static_cast<Bytes *>(context);
    auto const *const first = static_cast<std::uint8_t const *>(data);
    bytes.insert(bytes.end(), first, first + size);
}

void appendLittleEndian(Bytes &bytes, std::uint64_t const value, std::size_t const width) {
    for (auto index = std::size_t{0}; index < width; ++index) {
        bytes.push_back(static_cast<std::uint8_t>((value >> (8u * index)) & 0xFFu));
    }
}

void putLittleEndian(Bytes &bytes, std::size_t const offset, std::uint64_t const value, std::size_t const width) {
    for (auto index = std::size_t{0}; index < width; ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>((value >> (8u * index)) & 0xFFu);
    }
}

std::array<std::uint8_t, 8> bc1Block(std::uint16_t const colour) {
    auto const low = static_cast<std::uint8_t>(colour & 0xFFu);
    auto const high = static_cast<std::uint8_t>(colour >> 8u);
    return {low, high, low, high, 0, 0, 0, 0};
}

// ETC1 (so ETC2) individual mode, no flip, made-up colours and indices.
std::array<std::uint8_t, 8> etcBlock(std::uint32_t const seed) {
    auto const high = (seed * 0x9E3779B1u) & 0xFFFFFFFCu;
    auto const low = seed * 0x85EBCA6Bu + 0x7F4A7C15u;
    return {static_cast<std::uint8_t>(high >> 24u), static_cast<std::uint8_t>(high >> 16u),
            static_cast<std::uint8_t>(high >> 8u),  static_cast<std::uint8_t>(high),
            static_cast<std::uint8_t>(low >> 24u),  static_cast<std::uint8_t>(low >> 16u),
            static_cast<std::uint8_t>(low >> 8u),   static_cast<std::uint8_t>(low)};
}

std::string base64(Bytes const &bytes) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    auto text = std::string{};
    for (auto index = std::size_t{0}; index < bytes.size(); index += 3) {
        auto const left = bytes.size() - index;
        auto const word = std::uint32_t{bytes[index]} << 16u | (left > 1 ? std::uint32_t{bytes[index + 1]} << 8u : 0u)
                        | (left > 2 ? std::uint32_t{bytes[index + 2]} : 0u);
        text += kAlphabet[(word >> 18u) & 63u];
        text += kAlphabet[(word >> 12u) & 63u];
        text += left > 1 ? kAlphabet[(word >> 6u) & 63u] : '=';
        text += left > 2 ? kAlphabet[word & 63u] : '=';
    }
    return text;
}

std::string number(float const value) {
    auto text = std::array<char, 32>{};
    std::snprintf(text.data(), text.size(), "%.9g", static_cast<double>(value));
    return text.data();
}

std::string listOf(std::vector<std::string> const &items) {
    auto text = std::string{"["};
    for (auto index = std::size_t{0}; index < items.size(); ++index) {
        text += (index == 0 ? "" : ",") + items[index];
    }
    return text + "]";
}

std::string baseName(std::string const &stem) {
    auto const slash = stem.rfind('/');
    return slash == std::string::npos ? stem : stem.substr(slash + 1);
}

// The buffer, its views and accessors, as they are written.
struct BufferWriter {
    std::size_t addView(void const *const data, std::size_t const size) {
        while (bytes.size() % 4 != 0) {
            bytes.push_back(0);
        }
        auto const offset = bytes.size();
        auto const *const first = static_cast<std::uint8_t const *>(data);
        bytes.insert(bytes.end(), first, first + size);
        views.push_back("{\"buffer\":0,\"byteOffset\":" + std::to_string(offset)
                        + ",\"byteLength\":" + std::to_string(size) + "}");
        return views.size() - 1;
    }

    // `count` elements of `components` values each.
    template <class T>
    std::size_t addAccessor(std::vector<T> const &values, std::size_t const components, int const componentType,
                            char const *const type, std::string const &extra = {}) {
        auto const view = addView(values.data(), values.size() * sizeof(T));
        accessors.push_back("{\"bufferView\":" + std::to_string(view) + ",\"componentType\":"
                            + std::to_string(componentType) + ",\"count\":"
                            + std::to_string(values.size() / components) + ",\"type\":\"" + type + "\"" + extra + "}");
        return accessors.size() - 1;
    }

    Bytes bytes;
    std::vector<std::string> views;
    std::vector<std::string> accessors;
};

std::string boundsOf(std::vector<core::vec3> const &positions) {
    auto low = core::vec3{std::numeric_limits<float>::max()};
    auto high = core::vec3{-std::numeric_limits<float>::max()};
    for (auto const &position : positions) {
        low = tc::min(low, position);
        high = tc::max(high, position);
    }
    return ",\"min\":[" + number(low.x) + "," + number(low.y) + "," + number(low.z) + "],\"max\":[" + number(high.x)
         + "," + number(high.y) + "," + number(high.z) + "]";
}

std::vector<float> floatsOf(std::vector<core::vec3> const &vectors) {
    auto floats = std::vector<float>{};
    for (auto const &vector : vectors) {
        floats.insert(floats.end(), {vector.x, vector.y, vector.z});
    }
    return floats;
}

std::vector<float> floatsOf(std::vector<core::vec2> const &vectors) {
    auto floats = std::vector<float>{};
    for (auto const &vector : vectors) {
        floats.insert(floats.end(), {vector.x, vector.y});
    }
    return floats;
}

std::string meshOf(BufferWriter &buffer, TestPrimitive const &primitive) {
    constexpr auto kFloat = 5126;
    constexpr auto kUnsignedShort = 5123;
    auto attributes = "\"POSITION\":" + std::to_string(buffer.addAccessor(floatsOf(primitive.positions), 3, kFloat,
                                                                           "VEC3", boundsOf(primitive.positions)));
    if (not primitive.normals.empty()) {
        attributes +=
            ",\"NORMAL\":" + std::to_string(buffer.addAccessor(floatsOf(primitive.normals), 3, kFloat, "VEC3"));
    }
    if (not primitive.uvs.empty()) {
        attributes += ",\"TEXCOORD_0\":" + std::to_string(buffer.addAccessor(floatsOf(primitive.uvs), 2, kFloat, "VEC2"));
    }
    auto text = "{\"primitives\":[{\"attributes\":{" + attributes + "}";
    if (not primitive.indices.empty()) {
        text += ",\"indices\":" + std::to_string(buffer.addAccessor(primitive.indices, 1, kUnsignedShort, "SCALAR"));
    }
    if (primitive.material >= 0) {
        text += ",\"material\":" + std::to_string(primitive.material);
    }
    return text + "}]}";
}

Bytes glbOf(std::string json, Bytes binary) {
    while (json.size() % 4 != 0) {
        json += ' ';
    }
    while (binary.size() % 4 != 0) {
        binary.push_back(0);
    }
    auto glb = Bytes{};
    appendLittleEndian(glb, 0x46546C67u, 4);
    appendLittleEndian(glb, 2, 4);
    appendLittleEndian(glb, 12 + 8 + json.size() + 8 + binary.size(), 4);
    appendLittleEndian(glb, json.size(), 4);
    appendLittleEndian(glb, 0x4E4F534Au, 4);
    glb.insert(glb.end(), json.begin(), json.end());
    appendLittleEndian(glb, binary.size(), 4);
    appendLittleEndian(glb, 0x004E4942u, 4);
    glb.insert(glb.end(), binary.begin(), binary.end());
    return glb;
}

void addKey(std::string &json, char const *const key, std::string const &value) {
    if (not value.empty()) {
        json += ",\"" + std::string{key} + "\":" + value;
    }
}

} // namespace

Bytes pngOf(core::ivec2 const extent, std::vector<std::uint8_t> const &rgba) {
    auto bytes = Bytes{};
    stbi_write_png_to_func(appendTo, &bytes, extent.x, extent.y, 4, rgba.data(), extent.x * 4);
    return bytes;
}

Bytes jpegOf(core::ivec2 const extent, std::vector<std::uint8_t> const &rgba, int const quality) {
    auto bytes = Bytes{};
    stbi_write_jpg_to_func(appendTo, &bytes, extent.x, extent.y, 4, rgba.data(), quality);
    return bytes;
}

std::vector<std::uint8_t> solid(core::ivec2 const extent, std::uint8_t const r, std::uint8_t const g,
                                std::uint8_t const b, std::uint8_t const a) {
    auto texels = std::vector<std::uint8_t>{};
    for (auto index = 0; index < extent.x * extent.y; ++index) {
        texels.insert(texels.end(), {r, g, b, a});
    }
    return texels;
}

Bytes ktx2Of(Ktx2Desc const &desc) {
    constexpr auto kHeaderBytes = std::size_t{80};
    auto const levels = desc.levels.size();
    auto file = Bytes{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    appendLittleEndian(file, desc.vkFormat, 4);
    appendLittleEndian(file, 1, 4); // typeSize
    appendLittleEndian(file, static_cast<std::uint64_t>(desc.extent.x), 4);
    appendLittleEndian(file, static_cast<std::uint64_t>(desc.extent.y), 4);
    appendLittleEndian(file, 0, 4); // depth
    appendLittleEndian(file, 0, 4); // layers
    appendLittleEndian(file, 1, 4); // faces
    appendLittleEndian(file, levels, 4);
    appendLittleEndian(file, desc.supercompression, 4);
    file.resize(kHeaderBytes + levels * 24); // no data format descriptor, key-values or global data
    for (auto level = levels; level-- > 0;) {
        while (file.size() % 16 != 0) {
            file.push_back(0);
        }
        auto const entry = kHeaderBytes + level * 24;
        putLittleEndian(file, entry, file.size(), 8);
        putLittleEndian(file, entry + 8, desc.levels[level].size(), 8);
        putLittleEndian(file, entry + 16, desc.levels[level].size(), 8);
        file.insert(file.end(), desc.levels[level].begin(), desc.levels[level].end());
    }
    return file;
}

std::vector<std::vector<std::uint8_t>> bc1Levels() {
    auto levels = std::vector<std::vector<std::uint8_t>>{};
    auto const colours = std::array<std::uint16_t, 4>{0xF800u, 0x07E0u, 0x001Fu, 0xFFFFu};
    for (auto level = 0u; level < 4u; ++level) {
        auto &blocks = levels.emplace_back();
        for (auto block = 0; block < (level == 0 ? 4 : 1); ++block) {
            auto const bytes = bc1Block(colours[level]);
            blocks.insert(blocks.end(), bytes.begin(), bytes.end());
        }
    }
    return levels;
}

std::vector<std::vector<std::uint8_t>> etc2Levels() {
    auto levels = std::vector<std::vector<std::uint8_t>>{};
    auto seed = 1u;
    for (auto const blocks : {16, 4, 1, 1, 1}) {
        auto &level = levels.emplace_back();
        for (auto block = 0; block < blocks; ++block, ++seed) {
            auto const bytes = etcBlock(seed);
            level.insert(level.end(), bytes.begin(), bytes.end());
        }
    }
    return levels;
}

GltfFiles writeGltf(TestGltf const &gltf, GltfLayout const layout, std::string const &stem) {
    auto result = GltfFiles{};
    auto buffer = BufferWriter{};
    auto meshes = std::vector<std::string>{};
    for (auto const &primitive : gltf.meshes) {
        meshes.push_back(meshOf(buffer, primitive));
    }
    auto images = std::vector<std::string>{};
    for (auto index = std::size_t{0}; index < gltf.images.size(); ++index) {
        auto const &png = gltf.images[index];
        if (layout == GltfLayout::Binary) {
            auto const view = buffer.addView(png.data(), png.size());
            images.push_back("{\"bufferView\":" + std::to_string(view) + ",\"mimeType\":\"image/png\"}");
        } else if (layout == GltfLayout::Embedded) {
            images.push_back("{\"uri\":\"data:image/png;base64," + base64(png) + "\"}");
        } else {
            auto const name = baseName(stem) + "_image" + std::to_string(index) + ".png";
            images.push_back("{\"uri\":\"" + name + "\"}");
            result.files[stem + "_image" + std::to_string(index) + ".png"] = png;
        }
    }
    auto bufferJson = "{\"byteLength\":" + std::to_string(buffer.bytes.size());
    if (layout == GltfLayout::Embedded) {
        bufferJson += ",\"uri\":\"data:application/octet-stream;base64," + base64(buffer.bytes) + "\"";
    } else if (layout == GltfLayout::External) {
        bufferJson += ",\"uri\":\"" + baseName(stem) + ".bin\"";
        result.files[stem + ".bin"] = buffer.bytes;
    }
    auto json = std::string{"{\"asset\":{\"version\":\"2.0\"},\"scene\":0"};
    addKey(json, "extensionsUsed", gltf.extensionsUsed);
    addKey(json, "extensions", gltf.extensions);
    addKey(json, "scenes", gltf.scenes);
    addKey(json, "nodes", gltf.nodes);
    addKey(json, "meshes", listOf(meshes));
    addKey(json, "accessors", listOf(buffer.accessors));
    addKey(json, "bufferViews", listOf(buffer.views));
    addKey(json, "buffers", "[" + bufferJson + "}]");
    addKey(json, "materials", gltf.materials);
    addKey(json, "textures", gltf.textures);
    addKey(json, "samplers", gltf.samplers);
    addKey(json, "images", images.empty() ? std::string{} : listOf(images));
    addKey(json, "cameras", gltf.cameras);
    json += "}";
    if (layout == GltfLayout::Binary) {
        result.main = stem + ".glb";
        result.files[result.main] = glbOf(std::move(json), std::move(buffer.bytes));
    } else {
        result.main = stem + ".gltf";
        result.files[result.main] = Bytes(json.begin(), json.end());
    }
    return result;
}

TestPrimitive quadPrimitive(float const half) {
    auto quad = TestPrimitive{};
    quad.positions = {core::vec3{-half, -half, 0.0f}, core::vec3{half, -half, 0.0f}, core::vec3{half, half, 0.0f},
                      core::vec3{-half, half, 0.0f}};
    quad.normals = std::vector<core::vec3>(4, core::vec3{0.0f, 0.0f, 1.0f});
    quad.uvs = {core::vec2{0.0f, 1.0f}, core::vec2{1.0f, 1.0f}, core::vec2{1.0f, 0.0f}, core::vec2{0.0f, 0.0f}};
    quad.indices = {0, 1, 2, 0, 2, 3};
    quad.material = 0;
    return quad;
}

} // namespace lr::test
