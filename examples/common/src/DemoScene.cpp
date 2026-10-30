#include <lr/examples/DemoScene.h>

#include <lr/core/Primitives.h>

#include <array>
#include <cstdint>
#include <fstream>
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

} // namespace

DemoScene makeDemoScene() {
    auto scene = DemoScene{};
    auto &delta = scene.delta;
    auto const plane = core::MeshId{0};
    auto const box = core::MeshId{1};
    auto const sphere = core::MeshId{2};
    delta.meshes = {{plane, share(core::makePlane(core::vec2{6.0f, 6.0f}))},
                    {box, share(core::makeBox(core::vec3{0.5f, 0.5f, 0.5f}))},
                    {sphere, share(core::makeSphere(0.7f, 48, 24))}};

    auto const ground = core::MaterialId{0};
    auto const red = core::MaterialId{1};
    auto const gold = core::MaterialId{2};
    auto const glow = core::MaterialId{3};
    auto emissive = material(core::vec3{0.0f}, 0.0f, 1.0f);
    emissive.emissive = core::vec3{0.4f, 0.8f, 2.0f};
    delta.materials = {{ground, material(core::vec3{0.6f}, 0.0f, 0.9f)},
                       {red, material(core::vec3{0.8f, 0.1f, 0.08f}, 0.0f, 0.5f)},
                       {gold, material(core::vec3{1.0f, 0.78f, 0.34f}, 1.0f, 0.3f)},
                       {glow, emissive}};

    auto turned = core::Transform{};
    turned.translation = core::vec3{-1.3f, 0.5f, 0.2f};
    turned.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, 0.6f);
    delta.instances = {{core::InstanceId{0}, plane, ground, core::mat4{1.0f}},
                       {core::InstanceId{1}, box, red, core::matrixOf(turned)},
                       {core::InstanceId{2}, sphere, gold, at(core::vec3{1.2f, 0.7f, -0.2f})},
                       {core::InstanceId{3}, sphere, glow, at(core::vec3{0.1f, 0.35f, 1.4f}) *
                                                               core::matrixOf(core::Transform{core::vec3{0.0f}, {},
                                                                                              core::vec3{0.35f}})}};

    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.color = core::vec3{1.0f, 0.95f, 0.85f};
    sun.illuminance = 3.0f;
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

bool writePpm(render::ImageRgba8 const &image, std::string const &path) {
    auto file = std::ofstream{path, std::ios::binary};
    if (not file) {
        return false;
    }
    file << "P6\n" << image.extent.x << " " << image.extent.y << "\n255\n";
    for (auto const pixel : image.pixels) {
        auto const rgb = std::array<char, 3>{static_cast<char>(pixel & 0xFFu), static_cast<char>((pixel >> 8u) & 0xFFu),
                                             static_cast<char>((pixel >> 16u) & 0xFFu)};
        file.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    }
    return static_cast<bool>(file);
}

} // namespace lr::examples
