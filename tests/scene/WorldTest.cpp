#include <lr/core/Primitives.h>
#include <lr/runtime/Runtime.h>
#include <lr/scene/World.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

namespace {

using namespace lr;
using core::vec3;
using scene::EntityId;

core::MeshHandle quadMesh() { return std::make_shared<core::MeshData const>(core::makeQuad(core::vec2{1.0f})); }

core::ImageHandle pixelImage() {
    auto image = core::ImageData{};
    image.extent = core::ivec2{1};
    image.levels.push_back({255, 255, 255, 255});
    return std::make_shared<core::ImageData const>(std::move(image));
}

core::Transform at(vec3 const &position) {
    auto transform = core::Transform{};
    transform.translation = position;
    return transform;
}

bool sameBits(core::mat4 const &a, core::mat4 const &b) { return std::memcmp(&a, &b, sizeof(core::mat4)) == 0; }

bool near(vec3 const &a, vec3 const &b, float const tolerance = 1e-5f) {
    return std::abs(a.x - b.x) <= tolerance and std::abs(a.y - b.y) <= tolerance and std::abs(a.z - b.z) <= tolerance;
}

vec3 translationOf(core::mat4 const &matrix) { return matrix[3].xyz(); }

bool isEmpty(core::RenderWorldDelta const &delta) {
    auto const &removed = delta.removed;
    return delta.meshes.empty() and delta.textures.empty() and delta.materials.empty() and delta.instances.empty()
       and delta.lights.empty() and not delta.environment and removed.meshes.empty() and removed.textures.empty()
       and removed.materials.empty() and removed.instances.empty() and removed.lights.empty();
}

std::unique_ptr<scene::World> plainWorld() { return scene::makeWorld(scene::WorldConfig{}); }

// An entity drawn with a quad and a material of its own, which only the entity holds.
EntityId quadEntity(scene::World &world, core::Transform const &local, std::optional<EntityId> const parent) {
    auto &resources = world.resources();
    auto const mesh = resources.addMesh(quadMesh());
    auto const material = resources.addMaterial(core::MaterialDesc{});
    auto const entity = world.createEntity(local, parent);
    world.setRenderable(entity, scene::Renderable{mesh, material});
    resources.releaseMesh(mesh);
    resources.releaseMaterial(material);
    return entity;
}

TEST_CASE("an entity is in the next delta when it appears, moves or goes, and in no other") {
    auto world = plainWorld();
    auto const entity = quadEntity(*world, at(vec3{1.0f, 2.0f, 3.0f}), std::nullopt);
    auto const renderable = *world->renderableOf(entity);

    auto const added = world->extractDelta();
    REQUIRE(added.instances.size() == 1);
    CHECK(added.instances[0].mesh == renderable.mesh);
    CHECK(added.instances[0].material == renderable.material);
    CHECK(sameBits(added.instances[0].objectToWorld, core::matrixOf(at(vec3{1.0f, 2.0f, 3.0f}))));
    REQUIRE(added.meshes.size() == 1);
    CHECK(added.meshes[0].id == renderable.mesh);
    REQUIRE(added.materials.size() == 1);
    CHECK(added.materials[0].id == renderable.material);
    auto const instance = added.instances[0].id;

    CHECK(isEmpty(world->extractDelta()));
    world->setTransform(entity, at(vec3{1.0f, 2.0f, 3.0f}));
    CHECK(isEmpty(world->extractDelta()));

    world->setTransform(entity, at(vec3{4.0f, 5.0f, 6.0f}));
    world->setTransform(entity, at(vec3{7.0f, 8.0f, 9.0f}));
    auto const moved = world->extractDelta();
    REQUIRE(moved.instances.size() == 1);
    CHECK(moved.instances[0].id == instance);
    CHECK(sameBits(moved.instances[0].objectToWorld, core::matrixOf(at(vec3{7.0f, 8.0f, 9.0f}))));
    CHECK(moved.meshes.empty());
    CHECK(moved.materials.empty());

    world->destroyEntity(entity);
    auto const removed = world->extractDelta();
    CHECK(removed.instances.empty());
    CHECK(removed.removed.instances == std::vector<core::InstanceId>{instance});
    CHECK(removed.removed.meshes == std::vector<core::MeshId>{renderable.mesh});
    CHECK(removed.removed.materials == std::vector<core::MaterialId>{renderable.material});
    CHECK(world->entityCount() == 0);
    CHECK(isEmpty(world->extractDelta()));
}

TEST_CASE("an entity created and destroyed between two deltas leaves no trace") {
    auto world = plainWorld();
    world->extractDelta();

    world->destroyEntity(quadEntity(*world, at(vec3{1.0f}), std::nullopt));

    CHECK(isEmpty(world->extractDelta()));
}

TEST_CASE("a child follows its parent, keeps its local transform when reparented, and goes with its parent") {
    auto world = plainWorld();
    auto parentTransform = at(vec3{10.0f, 0.0f, 0.0f});
    parentTransform.rotation = core::Rotation::axisAngle(vec3{0.0f, 0.0f, 1.0f}, core::kPi * 0.5f);
    auto const parent = world->createEntity(parentTransform, std::nullopt);
    auto const child = quadEntity(*world, at(vec3{1.0f, 0.0f, 0.0f}), parent);
    auto const other = world->createEntity(at(vec3{0.0f, 5.0f, 0.0f}), std::nullopt);

    // Before any delta, computed on the way; then the same bits from the delta.
    auto const expected = world->worldMatrix(child);
    CHECK(near(translationOf(expected), vec3{10.0f, 1.0f, 0.0f}));
    auto const added = world->extractDelta();
    REQUIRE(added.instances.size() == 1);
    CHECK(sameBits(added.instances[0].objectToWorld, expected));

    world->setTransform(parent, at(vec3{20.0f, 0.0f, 0.0f}));
    auto const followed = world->extractDelta();
    REQUIRE(followed.instances.size() == 1);
    CHECK(near(translationOf(followed.instances[0].objectToWorld), vec3{21.0f, 0.0f, 0.0f}));

    world->setParent(child, other);
    CHECK(world->parentOf(child) == other);
    auto const reparented = world->extractDelta();
    REQUIRE(reparented.instances.size() == 1);
    CHECK(near(translationOf(reparented.instances[0].objectToWorld), vec3{1.0f, 5.0f, 0.0f}));
    CHECK(near(world->transformOf(child).translation, vec3{1.0f, 0.0f, 0.0f}));

    world->setTransform(parent, at(vec3{30.0f, 0.0f, 0.0f}));
    CHECK(isEmpty(world->extractDelta()));

    world->destroyEntity(other);
    CHECK(not world->isAlive(child));
    CHECK(world->isAlive(parent));
    auto const removed = world->extractDelta();
    CHECK(removed.removed.instances == std::vector<core::InstanceId>{added.instances[0].id});
    CHECK(world->entityCount() == 1);
}

TEST_CASE("an entity cannot become its own descendant") {
    auto world = plainWorld();
    auto const root = world->createEntity(core::Transform::identity(), std::nullopt);
    auto const child = world->createEntity(core::Transform::identity(), root);
    auto const grandchild = world->createEntity(core::Transform::identity(), child);

    CHECK_THROWS_AS(world->setParent(root, grandchild), std::invalid_argument);
    CHECK_THROWS_AS(world->setParent(child, child), std::invalid_argument);
    world->setParent(grandchild, std::nullopt);
    CHECK(not world->parentOf(grandchild));
}

TEST_CASE("the ids of destroyed entities are rejected, also once their slot is reused") {
    auto world = plainWorld();
    auto const destroyed = world->createEntity(core::Transform::identity(), std::nullopt);
    world->destroyEntity(destroyed);
    auto const reused = world->createEntity(core::Transform::identity(), std::nullopt);

    CHECK(reused.index == destroyed.index);
    CHECK(reused != destroyed);
    CHECK(not world->isAlive(destroyed));
    CHECK(world->isAlive(reused));
    CHECK_THROWS_AS(world->setTransform(destroyed, core::Transform::identity()), std::invalid_argument);
    CHECK_THROWS_AS(world->worldMatrix(destroyed), std::invalid_argument);
    CHECK_THROWS_AS(world->parentOf(destroyed), std::invalid_argument);
    CHECK_THROWS_AS(world->createEntity(core::Transform::identity(), destroyed), std::invalid_argument);
    CHECK_THROWS_AS(world->destroyEntity(destroyed), std::invalid_argument);
    CHECK_THROWS_AS(world->setLight(destroyed, core::PointLight{}), std::invalid_argument);
    CHECK_THROWS_AS(world->cameraOf(reused), std::invalid_argument);
    CHECK(not world->isAlive(EntityId{}));
    world->setTransform(reused, at(vec3{1.0f}));
}

TEST_CASE("a resource goes to the render world once, however many use it, and leaves with its last reference") {
    auto world = plainWorld();
    auto const mesh = quadMesh();
    auto spawned = std::vector<EntityId>{};
    for (auto index = 0; index < 3; ++index) {
        spawned.push_back(world->spawnInstance(mesh, at(vec3{static_cast<float>(index)})));
    }

    auto const added = world->extractDelta();
    CHECK(added.instances.size() == 3);
    REQUIRE(added.meshes.size() == 1);
    auto const sameMesh = added.meshes[0].mesh == mesh;
    CHECK(sameMesh);
    REQUIRE(added.materials.size() == 1);
    CHECK(added.materials[0].id == world->resources().defaultMaterial());

    world->destroyEntity(spawned[0]);
    world->destroyEntity(spawned[1]);
    auto const fewer = world->extractDelta();
    CHECK(fewer.removed.instances.size() == 2);
    CHECK(fewer.removed.meshes.empty());

    world->destroyEntity(spawned[2]);
    auto const none = world->extractDelta();
    CHECK(none.removed.meshes == std::vector<core::MeshId>{added.meshes[0].id});
    CHECK(none.removed.materials.empty()); // the world holds its default material
    CHECK(world->resources().counts().meshes == 0);
}

TEST_CASE("a material holds its textures, and a texture is one per image and sampler") {
    auto world = plainWorld();
    auto &resources = world->resources();
    auto const image = pixelImage();
    auto const texture = resources.addTexture(image, core::TextureSampler{});
    CHECK(resources.addTexture(image, core::TextureSampler{}) == texture);
    auto clamped = core::TextureSampler{};
    clamped.wrapU = core::TextureWrap::ClampToEdge;
    auto const other = resources.addTexture(image, clamped);
    CHECK(other != texture);
    auto material = core::MaterialDesc{};
    material.baseColorTexture = texture;
    auto const textured = resources.addMaterial(material);
    resources.releaseTexture(texture);
    resources.releaseTexture(texture);
    resources.releaseTexture(other);

    auto const added = world->extractDelta();
    REQUIRE(added.textures.size() == 1);
    CHECK(added.textures[0].id == texture);
    CHECK(added.materials.size() == 1);

    resources.resubmitTextures({texture, core::TextureId{99}});
    auto const resubmitted = world->extractDelta();
    REQUIRE(resubmitted.textures.size() == 1);
    CHECK(resubmitted.textures[0].id == texture);

    resources.releaseMaterial(textured);
    auto const removed = world->extractDelta();
    CHECK(removed.removed.materials == std::vector<core::MaterialId>{textured});
    CHECK(removed.removed.textures == std::vector<core::TextureId>{texture});
    CHECK(resources.counts().textures == 0);

    auto unknown = core::MaterialDesc{};
    unknown.normalTexture = core::TextureId{42};
    CHECK_THROWS_AS(resources.addMaterial(unknown), std::invalid_argument);
    CHECK_THROWS_AS(resources.releaseMesh(core::MeshId{0}), std::invalid_argument);
}

TEST_CASE("replacing a mesh upserts it again under its id") {
    auto world = plainWorld();
    auto const entity = world->spawnInstance(quadMesh(), core::Transform::identity());
    auto const mesh = world->renderableOf(entity)->mesh;
    world->extractDelta();

    auto const reloaded = quadMesh();
    world->resources().replaceMesh(mesh, reloaded);
    auto const delta = world->extractDelta();

    REQUIRE(delta.meshes.size() == 1);
    CHECK(delta.meshes[0].id == mesh);
    auto const replaced = delta.meshes[0].mesh == reloaded;
    CHECK(replaced);
    CHECK(delta.instances.empty());
    CHECK(world->resources().addMesh(reloaded) == mesh);
}

TEST_CASE("an id freed between two deltas is not given out again before the delta that removes it") {
    auto world = plainWorld();
    auto const mesh = quadMesh();
    auto const first = world->spawnInstance(mesh, core::Transform::identity());
    auto const firstInstance = world->extractDelta().instances.at(0).id;

    world->destroyEntity(first);
    world->spawnInstance(mesh, core::Transform::identity());
    auto const swapped = world->extractDelta();
    REQUIRE(swapped.instances.size() == 1);
    CHECK(swapped.instances[0].id != firstInstance);
    CHECK(swapped.removed.instances == std::vector<core::InstanceId>{firstInstance});

    world->spawnInstance(mesh, core::Transform::identity());
    CHECK(world->extractDelta().instances.at(0).id == firstInstance);
}

TEST_CASE("a light moves with its entity, a camera looks down its entity's -z") {
    auto world = plainWorld();
    auto const lamp = world->createEntity(at(vec3{0.0f, 2.0f, 0.0f}), std::nullopt);
    auto point = core::PointLight{};
    point.position = vec3{1.0f, 0.0f, 0.0f};
    world->setLight(lamp, point);
    auto turned = core::Transform{};
    turned.rotation = core::Rotation::axisAngle(vec3{0.0f, 0.0f, 1.0f}, core::kPi * 0.5f);
    auto const sun = world->createEntity(turned, std::nullopt);
    auto directional = core::DirectionalLight{};
    directional.direction = vec3{1.0f, 0.0f, 0.0f};
    world->setLight(sun, directional);
    world->setEnvironment(core::EnvironmentDesc{vec3{0.5f}});

    auto const added = world->extractDelta();
    REQUIRE(added.lights.size() == 2);
    CHECK(near(std::get<core::PointLight>(added.lights[0].light).position, vec3{1.0f, 2.0f, 0.0f}));
    CHECK(near(std::get<core::DirectionalLight>(added.lights[1].light).direction, vec3{0.0f, 1.0f, 0.0f}));
    CHECK(added.environment.has_value());

    world->setTransform(lamp, at(vec3{0.0f, 3.0f, 0.0f}));
    auto const moved = world->extractDelta();
    REQUIRE(moved.lights.size() == 1);
    CHECK(moved.lights[0].id == added.lights[0].id);
    CHECK(near(std::get<core::PointLight>(moved.lights[0].light).position, vec3{1.0f, 3.0f, 0.0f}));
    CHECK(not moved.environment);

    auto const eye = vec3{0.0f, 2.0f, 5.0f};
    auto const forward = tc::normalize(vec3{0.0f, -2.0f, -5.0f});
    auto view = at(eye);
    view.rotation = core::Rotation::lookingAlong(forward, vec3{0.0f, 1.0f, 0.0f});
    auto const camera = world->createEntity(view, std::nullopt);
    auto lens = scene::CameraLens{};
    lens.verticalFov = 0.5f;
    world->setCamera(camera, lens);

    auto const desc = world->cameraOf(camera);
    CHECK(near(desc.position, eye));
    CHECK(near(desc.target - desc.position, forward));
    CHECK(tc::dot(desc.up, forward) == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(desc.up.y > 0.9f);
    CHECK(desc.verticalFov == 0.5f);
}

struct RandomHierarchy {
    std::vector<EntityId> entities;
};

core::Transform randomTransform(std::mt19937 &random) {
    auto uniform = std::uniform_real_distribution<float>{-1.0f, 1.0f};
    auto transform = core::Transform{};
    transform.translation = vec3{uniform(random), uniform(random), uniform(random)} * 10.0f;
    transform.rotation = core::Rotation::axisAngle(vec3{uniform(random), uniform(random), 1.0f}, uniform(random) * 3.0f);
    transform.scale = vec3{1.25f + uniform(random) * 0.75f};
    return transform;
}

// Every entity drawn, each one's parent made before it, so that the same draws give the
// same world, and reparenting to an earlier entity never makes a cycle.
RandomHierarchy buildRandomHierarchy(scene::World &world, std::uint32_t const seed, int const count) {
    auto random = std::mt19937{seed};
    auto hierarchy = RandomHierarchy{};
    auto const mesh = quadMesh();
    for (auto index = 0; index < count; ++index) {
        auto parent = std::optional<EntityId>{};
        if (index > 0 and random() % 32 != 0) {
            parent = hierarchy.entities[random() % hierarchy.entities.size()];
        }
        auto const entity = world.spawnInstance(mesh, randomTransform(random));
        if (parent) {
            world.setParent(entity, *parent);
        }
        hierarchy.entities.push_back(entity);
    }
    return hierarchy;
}

void changeRandomly(scene::World &world, RandomHierarchy const &hierarchy, std::uint32_t const seed) {
    auto random = std::mt19937{seed};
    auto const pick = [&] { return random() % hierarchy.entities.size(); };
    for (auto change = 0; change < 2000; ++change) {
        auto const index = pick();
        auto const entity = hierarchy.entities[index];
        if (not world.isAlive(entity)) {
            continue;
        }
        auto const kind = random() % 16;
        if (kind == 0 and index > 0) {
            auto const parent = hierarchy.entities[random() % index];
            if (world.isAlive(parent)) {
                world.setParent(entity, parent);
            }
        } else if (kind == 1) {
            world.destroyEntity(entity);
        } else {
            world.setTransform(entity, randomTransform(random));
        }
    }
}

void checkSameInstances(core::RenderWorldDelta const &a, core::RenderWorldDelta const &b) {
    REQUIRE(a.instances.size() == b.instances.size());
    auto mismatches = 0;
    for (auto index = std::size_t{0}; index < a.instances.size(); ++index) {
        if (a.instances[index].id != b.instances[index].id
            or not sameBits(a.instances[index].objectToWorld, b.instances[index].objectToWorld)) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
    CHECK(a.removed.instances == b.removed.instances);
}

TEST_CASE("propagating transforms on the frame pool gives the bits it gives on the host thread") {
    auto runtimeConfig = runtime::RuntimeConfig{};
    runtimeConfig.framePoolThreads = 4;
    auto const runtime = runtime::makeRuntime(runtimeConfig);
    auto parallelConfig = scene::WorldConfig{};
    parallelConfig.runtime = runtime.get();
    parallelConfig.parallelTransforms = 1;
    auto parallel = scene::makeWorld(parallelConfig);
    auto sequential = plainWorld();
    constexpr auto kEntities = 20000;
    auto const hierarchy = buildRandomHierarchy(*parallel, 7, kEntities);
    buildRandomHierarchy(*sequential, 7, kEntities);

    auto const parallelAdded = parallel->extractDelta();
    auto const sequentialAdded = sequential->extractDelta();
    CHECK(parallelAdded.instances.size() == kEntities);
    checkSameInstances(parallelAdded, sequentialAdded);

    changeRandomly(*parallel, hierarchy, 11);
    changeRandomly(*sequential, hierarchy, 11);
    auto const parallelChanged = parallel->extractDelta();
    auto const sequentialChanged = sequential->extractDelta();
    CHECK(not parallelChanged.instances.empty());
    CHECK(not parallelChanged.removed.instances.empty());
    checkSameInstances(parallelChanged, sequentialChanged);

    auto mismatches = 0;
    for (auto const entity : hierarchy.entities) {
        if (parallel->isAlive(entity) and not sameBits(parallel->worldMatrix(entity), sequential->worldMatrix(entity))) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
    CHECK(parallel->stats().parallelPropagations == 2);
    CHECK(sequential->stats().parallelPropagations == 0);
    CHECK(parallel->stats().propagatedTransforms == sequential->stats().propagatedTransforms);
}

} // namespace
