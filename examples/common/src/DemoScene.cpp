#include <lr/examples/DemoScene.h>

#include <lr/core/Primitives.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

namespace lr::examples {

namespace {

core::MeshHandle share(core::MeshData mesh) { return std::make_shared<core::MeshData const>(std::move(mesh)); }

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

core::MaterialDesc material(core::vec3 const &color, float const metallic, float const roughness) {
    auto desc = core::MaterialDesc{};
    desc.baseColor = core::vec4{color, 1.0f};
    desc.metallic = metallic;
    desc.roughness = roughness;
    return desc;
}

core::Transform at(core::vec3 const &position) {
    auto transform = core::Transform{};
    transform.translation = position;
    return transform;
}

core::Transform scaledAt(core::vec3 const &position, float const scale) {
    auto transform = at(position);
    transform.scale = core::vec3{scale};
    return transform;
}

core::Transform turnedAt(core::vec3 const &position, float const radians) {
    auto transform = at(position);
    transform.rotation = core::Rotation::axisAngle(core::vec3{0.0f, 1.0f, 0.0f}, radians);
    return transform;
}

void addDrawn(scene::World &world, core::Transform const &local, std::optional<scene::EntityId> const parent,
              scene::Renderable const &renderable) {
    world.setRenderable(world.createEntity(local, parent), renderable);
}

struct DemoMaterials {
    core::MaterialId ground;
    core::MaterialId wood;
    core::MaterialId gold;
    core::MaterialId glow;
    core::MaterialId glass;
    core::MaterialId leaves;
};

// The materials, which hold the textures made for them.
DemoMaterials addMaterials(scene::WorldResources &resources) {
    auto const checker = resources.addTexture(checkerImage(), core::TextureSampler{});
    auto const crate = resources.addTexture(crateImage(), core::TextureSampler{});
    auto const crateRelief = resources.addTexture(crateNormals(), core::TextureSampler{});
    auto const lattice = resources.addTexture(latticeImage(), core::TextureSampler{});
    auto ground = material(core::vec3{1.0f}, 0.0f, 0.9f);
    ground.baseColorTexture = checker;
    ground.uvTransform.scale = core::vec2{6.0f};
    auto wood = material(core::vec3{1.0f}, 0.0f, 0.6f);
    wood.baseColorTexture = crate;
    wood.normalTexture = crateRelief;
    auto glow = material(core::vec3{0.0f}, 0.0f, 1.0f);
    glow.emissive = core::vec3{0.4f, 0.8f, 2.0f};
    auto glass = material(core::vec3{0.6f, 0.75f, 0.85f}, 0.0f, 0.12f);
    glass.baseColor.w = 0.35f;
    glass.alphaMode = core::AlphaMode::Blend;
    auto leaves = material(core::vec3{1.0f}, 0.0f, 0.8f);
    leaves.baseColorTexture = lattice;
    leaves.alphaMode = core::AlphaMode::Mask;
    leaves.doubleSided = true;
    auto const materials = DemoMaterials{resources.addMaterial(ground),
                                         resources.addMaterial(wood),
                                         resources.addMaterial(material(core::vec3{1.0f, 0.78f, 0.34f}, 1.0f, 0.3f)),
                                         resources.addMaterial(glow),
                                         resources.addMaterial(glass),
                                         resources.addMaterial(leaves)};
    for (auto const texture : {checker, crate, crateRelief, lattice}) {
        resources.releaseTexture(texture);
    }
    return materials;
}

void releaseMaterials(scene::WorldResources &resources, DemoMaterials const &materials) {
    for (auto const id : {materials.ground, materials.wood, materials.gold, materials.glow, materials.glass,
                          materials.leaves}) {
        resources.releaseMaterial(id);
    }
}

void addLights(scene::World &world, scene::EntityId const lantern) {
    auto sun = core::DirectionalLight{};
    sun.direction = core::vec3{-0.4f, -1.0f, -0.5f};
    sun.color = core::vec3{1.0f, 0.95f, 0.85f};
    sun.illuminance = 3.0f;
    sun.castsShadows = true;
    world.setLight(world.createEntity(core::Transform::identity(), std::nullopt), sun);
    auto lamp = core::PointLight{};
    lamp.color = core::vec3{1.0f, 0.6f, 0.3f};
    lamp.intensity = 4.0f;
    lamp.range = 8.0f;
    world.setLight(world.createEntity(at(core::vec3{0.5f, 1.25f, 0.4f}), lantern), lamp);
    world.setEnvironment(core::EnvironmentDesc{core::vec3{0.12f, 0.16f, 0.24f}});
}

} // namespace

DemoEntities addDemoScene(scene::World &world) {
    auto &resources = world.resources();
    auto const plane = resources.addMesh(share(core::makePlane(core::vec2{6.0f, 6.0f})));
    auto const box = resources.addMesh(share(core::makeBox(core::vec3{0.5f, 0.5f, 0.5f})));
    auto const sphere = resources.addMesh(share(core::makeSphere(0.7f, 48, 24)));
    auto const quad = resources.addMesh(share(core::makeQuad(core::vec2{0.8f, 0.8f})));
    auto const materials = addMaterials(resources);

    addDrawn(world, core::Transform::identity(), std::nullopt, scene::Renderable{plane, materials.ground});
    addDrawn(world, turnedAt(core::vec3{-1.3f, 0.5f, 0.2f}, 0.6f), std::nullopt, scene::Renderable{box, materials.wood});
    addDrawn(world, at(core::vec3{1.2f, 0.7f, -0.2f}), std::nullopt, scene::Renderable{sphere, materials.gold});
    auto const lantern = world.createEntity(at(core::vec3{0.1f, 0.35f, 1.4f}), std::nullopt);
    addDrawn(world, scaledAt(core::vec3{0.0f}, 0.35f), lantern, scene::Renderable{sphere, materials.glow});
    addDrawn(world, scaledAt(core::vec3{0.05f, 0.45f, 0.25f}, 0.643f), std::nullopt,
             scene::Renderable{sphere, materials.glass});
    addDrawn(world, turnedAt(core::vec3{-0.6f, 0.8f, -1.6f}, 0.35f), std::nullopt,
             scene::Renderable{quad, materials.leaves});
    addLights(world, lantern);
    for (auto const mesh : {plane, box, sphere, quad}) {
        resources.releaseMesh(mesh);
    }
    releaseMaterials(resources, materials);

    auto view = core::CameraDesc{};
    view.position = core::vec3{0.0f, 2.2f, 5.5f};
    view.target = core::vec3{0.0f, 0.5f, 0.0f};
    auto placed = at(view.position);
    placed.rotation = core::Rotation::lookingAlong(view.target - view.position, view.up);
    auto const camera = world.createEntity(placed, std::nullopt);
    auto lens = scene::CameraLens{};
    lens.verticalFov = view.verticalFov;
    lens.nearPlane = view.nearPlane;
    lens.farPlane = view.farPlane;
    world.setCamera(camera, lens);
    return DemoEntities{camera, view, lantern};
}

DemoScene makeDemoScene() {
    auto const world = scene::makeWorld(scene::WorldConfig{});
    auto const entities = addDemoScene(*world);
    return DemoScene{world->extractDelta(), entities.view};
}

} // namespace lr::examples
