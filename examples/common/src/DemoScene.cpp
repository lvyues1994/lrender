#include <lr/examples/DemoScene.h>

#include <lr/core/Primitives.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

namespace lr::examples {

namespace {

core::MeshHandle share(core::MeshData mesh) { return std::make_shared<core::MeshData const>(std::move(mesh)); }

core::MaterialDesc material(core::vec3 const &color, float const metallic, float const roughness) {
    auto desc = core::MaterialDesc{};
    desc.baseColor = core::vec4{color, 1.0f};
    desc.metallic = metallic;
    desc.roughness = roughness;
    return desc;
}

core::mat4 at(core::vec3 const &position) {
    auto transform = core::Transform{};
    transform.translation = position;
    return core::matrixOf(transform);
}

std::uint8_t byteOf(float const value) {
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

// An RGBA8 image `size` texels square whose texel (x, y) is texel(x, y), components in [0, 1].
template <class Texel>
core::ImageHandle paint(int const size, core::ColorEncoding const encoding, Texel const &texel) {
    auto image = core::ImageData{};
    image.extent = core::ivec2{size, size};
    image.encoding = encoding;
    auto &level = image.levels.emplace_back(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 4);
    for (auto y = 0; y < size; ++y) {
        for (auto x = 0; x < size; ++x) {
            auto const value = texel(x, y);
            auto const index =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)) * 4;
            level[index] = byteOf(value.x);
            level[index + 1] = byteOf(value.y);
            level[index + 2] = byteOf(value.z);
            level[index + 3] = byteOf(value.w);
        }
    }
    return std::make_shared<core::ImageData const>(std::move(image));
}

core::ImageHandle checkerImage() {
    return paint(256, core::ColorEncoding::Srgb, [](int const x, int const y) {
        auto const light = ((x / 32) + (y / 32)) % 2 == 0;
        return light ? core::vec4{0.82f, 0.80f, 0.74f, 1.0f} : core::vec4{0.30f, 0.33f, 0.38f, 1.0f};
    });
}

// Planks with dark seams and a frame, for every face of the box.
float crateHeight(int const x, int const y) {
    auto const frame = x < 16 or x >= 112 or y < 16 or y >= 112;
    auto const seam = not frame and (y - 16) % 24 < 2;
    return frame ? 1.0f : (seam ? 0.0f : 0.6f);
}

core::ImageHandle crateImage() {
    return paint(128, core::ColorEncoding::Srgb, [](int const x, int const y) {
        auto const grain = 0.06f * std::sin(static_cast<float>(x) * 0.35f + 3.0f * std::sin(static_cast<float>(y) * 0.05f));
        auto const height = crateHeight(x, y);
        auto const wood = core::vec3{0.62f, 0.42f, 0.22f} * (0.75f + grain + 0.25f * height);
        return core::vec4{height == 0.0f ? wood * 0.35f : wood, 1.0f};
    });
}

// The crate's relief as a tangent-space normal map (+Y up in the image).
core::ImageHandle crateNormals() {
    return paint(128, core::ColorEncoding::Linear, [](int const x, int const y) {
        auto const slopeX = crateHeight(std::min(x + 1, 127), y) - crateHeight(std::max(x - 1, 0), y);
        auto const slopeY = crateHeight(x, std::max(y - 1, 0)) - crateHeight(x, std::min(y + 1, 127));
        auto const normal = tc::normalize(core::vec3{-slopeX, -slopeY, 1.0f});
        return core::vec4{normal * 0.5f + core::vec3{0.5f}, 1.0f};
    });
}

// A lattice of round holes, cut out by alpha.
core::ImageHandle latticeImage() {
    return paint(128, core::ColorEncoding::Srgb, [](int const x, int const y) {
        auto const u = static_cast<float>(x % 32) - 15.5f;
        auto const v = static_cast<float>(y % 32) - 15.5f;
        auto const hole = u * u + v * v < 11.0f * 11.0f;
        return core::vec4{0.25f, 0.55f, 0.3f, hole ? 0.0f : 1.0f};
    });
}

} // namespace

DemoScene makeDemoScene() {
    auto scene = DemoScene{};
    auto &delta = scene.delta;
    auto const plane = core::MeshId{0};
    auto const box = core::MeshId{1};
    auto const sphere = core::MeshId{2};
    auto const quad = core::MeshId{3};
    delta.meshes = {{plane, share(core::makePlane(core::vec2{6.0f, 6.0f}))},
                    {box, share(core::makeBox(core::vec3{0.5f, 0.5f, 0.5f}))},
                    {sphere, share(core::makeSphere(0.7f, 48, 24))},
                    {quad, share(core::makeQuad(core::vec2{0.8f, 0.8f}))}};

    auto const checker = core::TextureId{0};
    auto const crate = core::TextureId{1};
    auto const crateRelief = core::TextureId{2};
    auto const lattice = core::TextureId{3};
    delta.textures = {{checker, checkerImage(), core::TextureSampler{}},
                      {crate, crateImage(), core::TextureSampler{}},
                      {crateRelief, crateNormals(), core::TextureSampler{}},
                      {lattice, latticeImage(), core::TextureSampler{}}};

    auto const ground = core::MaterialId{0};
    auto const wood = core::MaterialId{1};
    auto const gold = core::MaterialId{2};
    auto const glow = core::MaterialId{3};
    auto const glass = core::MaterialId{4};
    auto const leaves = core::MaterialId{5};
    auto groundMaterial = material(core::vec3{1.0f}, 0.0f, 0.9f);
    groundMaterial.baseColorTexture = checker;
    groundMaterial.uvTransform.scale = core::vec2{6.0f};
    auto woodMaterial = material(core::vec3{1.0f}, 0.0f, 0.6f);
    woodMaterial.baseColorTexture = crate;
    woodMaterial.normalTexture = crateRelief;
    auto emissive = material(core::vec3{0.0f}, 0.0f, 1.0f);
    emissive.emissive = core::vec3{0.4f, 0.8f, 2.0f};
    auto glassMaterial = material(core::vec3{0.6f, 0.75f, 0.85f}, 0.0f, 0.12f);
    glassMaterial.baseColor.w = 0.35f;
    glassMaterial.alphaMode = core::AlphaMode::Blend;
    auto leavesMaterial = material(core::vec3{1.0f}, 0.0f, 0.8f);
    leavesMaterial.baseColorTexture = lattice;
    leavesMaterial.alphaMode = core::AlphaMode::Mask;
    leavesMaterial.doubleSided = true;
    delta.materials = {{ground, groundMaterial},
                       {wood, woodMaterial},
                       {gold, material(core::vec3{1.0f, 0.78f, 0.34f}, 1.0f, 0.3f)},
                       {glow, emissive},
                       {glass, glassMaterial},
                       {leaves, leavesMaterial}};

    auto turned = core::Transform{};
    turned.translation = core::vec3{-1.3f, 0.5f, 0.2f};
    turned.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, 0.6f);
    auto standing = core::Transform{};
    standing.translation = core::vec3{-0.6f, 0.8f, -1.6f};
    standing.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, 0.35f);
    delta.instances = {{core::InstanceId{0}, plane, ground, core::mat4{1.0f}},
                       {core::InstanceId{1}, box, wood, core::matrixOf(turned)},
                       {core::InstanceId{2}, sphere, gold, at(core::vec3{1.2f, 0.7f, -0.2f})},
                       {core::InstanceId{3}, sphere, glow, at(core::vec3{0.1f, 0.35f, 1.4f}) *
                                                               core::matrixOf(core::Transform{core::vec3{0.0f}, {},
                                                                                              core::vec3{0.35f}})},
                       {core::InstanceId{4}, sphere, glass, at(core::vec3{0.05f, 0.45f, 0.25f}) *
                                                                core::matrixOf(core::Transform{core::vec3{0.0f}, {},
                                                                                               core::vec3{0.643f}})},
                       {core::InstanceId{5}, quad, leaves, core::matrixOf(standing)}};

    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.color = core::vec3{1.0f, 0.95f, 0.85f};
    sun.illuminance = 3.0f;
    sun.castsShadows = true;
    auto lamp = core::PointLight{};
    lamp.position = core::vec3{0.6f, 1.6f, 1.8f};
    lamp.color = core::vec3{1.0f, 0.6f, 0.3f};
    lamp.intensity = 4.0f;
    lamp.range = 8.0f;
    delta.lights = {{core::LightId{0}, sun}, {core::LightId{1}, lamp}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.12f, 0.16f, 0.24f}};

    scene.camera.position = core::vec3{0.0f, 2.2f, 5.5f};
    scene.camera.target = core::vec3{0.0f, 0.5f, 0.0f};
    return scene;
}

} // namespace lr::examples
