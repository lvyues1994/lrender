#include <lr/examples/ScaleScene.h>

#include <lr/core/Primitives.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace lr::examples {

namespace {

constexpr auto kSpacing = 2.0f;
constexpr auto kMaterials = 16u;

// splitmix64: well-spread bits from consecutive inputs.
std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

// Draws from one stream per (seed, item, purpose).
struct Draws {
    std::uint64_t state;

    Draws(std::uint32_t const seed, std::uint64_t const item, std::uint64_t const purpose) noexcept
        : state(mix(mix(mix(seed) ^ item) ^ purpose)) {}

    std::uint32_t next() noexcept {
        state = mix(state);
        return static_cast<std::uint32_t>(state >> 32);
    }
    float unit() noexcept { return static_cast<float>(next() >> 8) / static_cast<float>(1u << 24); }
    float between(float const low, float const high) noexcept { return low + (high - low) * unit(); }
};

struct Shape {
    core::MeshData mesh;
    float rest = 0.0f; // the height of its centre when standing on the ground at scale 1
};

std::array<Shape, 4> shapes(std::uint32_t const segments) {
    auto const rings = std::max(segments / 2u, 2u);
    return {Shape{core::makeBox(core::vec3{0.5f}), 0.5f},
            Shape{core::makeBox(core::vec3{0.3f, 1.5f, 0.3f}), 1.5f},
            Shape{core::makeSphere(0.6f, segments, rings), 0.6f},
            Shape{core::makeSphere(0.35f, std::max(segments / 2u, 3u), std::max(rings / 2u, 2u)), 0.35f}};
}

core::vec3 colorOf(float const hue) {
    auto const channel = [hue](float const offset) {
        return 0.5f + 0.4f * std::cos(6.2831853f * (hue + offset));
    };
    return core::vec3{channel(0.0f), channel(0.33f), channel(0.67f)};
}

void addMaterials(core::RenderWorldDelta &delta, std::uint32_t const seed) {
    auto ground = core::MaterialDesc{};
    ground.baseColor = core::vec4{0.55f, 0.55f, 0.5f, 1.0f};
    ground.roughness = 0.9f;
    delta.materials.push_back(core::MaterialUpsert{core::MaterialId{0}, ground});
    for (auto index = 0u; index < kMaterials; ++index) {
        auto draws = Draws{seed, index, 1};
        auto material = core::MaterialDesc{};
        material.baseColor = core::vec4{colorOf(draws.unit()), 1.0f};
        material.metallic = index % 4u == 0u ? 1.0f : 0.0f;
        material.roughness = draws.between(0.2f, 0.9f);
        delta.materials.push_back(core::MaterialUpsert{core::MaterialId{index + 1u}, material});
    }
}

core::mat4 placed(core::vec3 const &position, float const turn, float const scale) {
    auto transform = core::Transform{};
    transform.translation = position;
    transform.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, turn);
    transform.scale = core::vec3{scale};
    return core::matrixOf(transform);
}

} // namespace

ScaleScene makeScaleScene(ScaleSceneDesc const &desc) {
    auto scene = ScaleScene{};
    auto &delta = scene.delta;
    auto const side = static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(desc.instances))));
    auto const half = 0.5f * kSpacing * static_cast<float>(side);

    auto meshes = shapes(desc.sphereSegments);
    auto triangles = std::array<std::uint64_t, 4>{};
    auto rests = std::array<float, 4>{};
    for (auto index = 0u; index < meshes.size(); ++index) {
        triangles[index] = meshes[index].mesh.indices.size() / 3u;
        rests[index] = meshes[index].rest;
        delta.meshes.push_back(core::MeshUpsert{
            core::MeshId{index + 1u}, std::make_shared<core::MeshData const>(std::move(meshes[index].mesh))});
    }
    delta.meshes.push_back(core::MeshUpsert{
        core::MeshId{0}, std::make_shared<core::MeshData const>(core::makePlane(core::vec2{half + kSpacing}))});
    addMaterials(delta, desc.seed);

    delta.instances.reserve(std::size_t{desc.instances} + 1u);
    delta.instances.push_back(core::InstanceUpsert{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}});
    for (auto index = 0u; index < desc.instances; ++index) {
        auto draws = Draws{desc.seed, index, 2};
        auto const shape = draws.next() % 4u;
        auto const material = draws.next() % kMaterials;
        auto const scale = draws.between(0.6f, 1.4f);
        auto const turn = draws.between(0.0f, 6.2831853f);
        auto const jitter = core::vec2{draws.between(-0.2f, 0.2f), draws.between(-0.2f, 0.2f)};
        auto const cell = core::vec2{static_cast<float>(index % side), static_cast<float>(index / side)};
        auto const ground = (cell + core::vec2{0.5f} + jitter) * kSpacing - core::vec2{half};
        auto const position = core::vec3{ground.x, rests[shape] * scale, ground.y};
        delta.instances.push_back(core::InstanceUpsert{core::InstanceId{index + 1u}, core::MeshId{shape + 1u},
                                                       core::MaterialId{material + 1u},
                                                       placed(position, turn, scale)});
        scene.triangles += triangles[shape];
    }

    for (auto index = 0u; index < desc.lights; ++index) {
        auto draws = Draws{desc.seed, index, 3};
        auto lamp = core::PointLight{};
        lamp.position = core::vec3{draws.between(-half, half), draws.between(1.0f, 3.0f), draws.between(-half, half)};
        lamp.color = colorOf(draws.unit());
        lamp.intensity = 4.0f;
        lamp.range = 6.0f;
        delta.lights.push_back(core::LightUpsert{core::LightId{index + 1u}, lamp});
    }
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.3f, -1.0f, -0.4f};
    sun.color = core::vec3{1.0f, 0.95f, 0.85f};
    sun.illuminance = 2.0f;
    sun.castsShadows = true;
    delta.lights.push_back(core::LightUpsert{core::LightId{0}, sun});
    delta.environment = core::EnvironmentDesc{core::vec3{0.12f, 0.16f, 0.24f}};

    scene.camera.position = core::vec3{-half + 2.0f, 3.0f, half - 2.0f};
    scene.camera.target = core::vec3{0.0f, 0.5f, 0.0f};
    scene.camera.nearPlane = 0.1f;
    scene.camera.farPlane = 4.0f * (half + kSpacing);
    return scene;
}

} // namespace lr::examples
