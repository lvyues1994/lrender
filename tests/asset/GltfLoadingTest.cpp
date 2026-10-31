#include "AssetTestSupport.h"

#include <lr/asset/SceneDelta.h>

#include <doctest/doctest.h>

#include <cmath>
#include <variant>
#include <vector>

namespace {

using namespace lr;
using asset::AssetErrorKind;
using core::vec3;
using test::AssetHarness;
using test::loadNow;

std::vector<std::uint8_t> const kColourTexels{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};

// Two meshes under a translated root, rotated and scaled, two materials sharing two
// textures (one image as colour, the other as data), a sun, a lamp and a camera.
test::TestGltf testScene() {
    auto scene = test::TestGltf{};
    scene.meshes.push_back(test::quadPrimitive(1.0f));
    auto triangle = test::TestPrimitive{};
    triangle.positions = {vec3{0.0f}, vec3{1.0f, 0.0f, 0.0f}, vec3{0.0f, 1.0f, 0.0f}};
    triangle.material = 1;
    scene.meshes.push_back(triangle);
    scene.images = {test::pngOf(core::ivec2{2, 2}, kColourTexels),
                    test::pngOf(core::ivec2{2, 2}, test::solid(core::ivec2{2, 2}, 128, 128, 255))};
    scene.extensionsUsed = R"(["KHR_lights_punctual","KHR_texture_transform"])";
    scene.extensions = R"({"KHR_lights_punctual":{"lights":[
        {"type":"directional","color":[1,0.5,0.25],"intensity":3},
        {"type":"point","color":[0.25,0.5,1],"intensity":10,"range":20}]}})";
    scene.scenes = R"([{"nodes":[0,3,4,5]}])";
    scene.nodes = R"([
        {"name":"root","translation":[1,2,3],"children":[1,2]},
        {"name":"quad","mesh":0,"rotation":[0,0.70710678,0,0.70710678]},
        {"name":"triangle","mesh":1,"scale":[2,2,2]},
        {"name":"sun","rotation":[-0.70710678,0,0,0.70710678],"extensions":{"KHR_lights_punctual":{"light":0}}},
        {"name":"lamp","translation":[0,4,0],"extensions":{"KHR_lights_punctual":{"light":1}}},
        {"name":"camera","translation":[0,0,5],"camera":0}])";
    scene.materials = R"([
        {"name":"painted","pbrMetallicRoughness":{"baseColorFactor":[0.5,0.25,0.125,1],"metallicFactor":0.25,
          "roughnessFactor":0.75,"baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":
          {"offset":[0.5,0.25],"rotation":0.5,"scale":[2,3]}}}},
         "normalTexture":{"index":1,"scale":0.5},"emissiveFactor":[0.125,0.25,0.5],"alphaMode":"MASK",
         "alphaCutoff":0.25,"doubleSided":true},
        {"name":"glass","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,0.5],"metallicRoughnessTexture":{"index":1}},
         "emissiveTexture":{"index":0},"alphaMode":"BLEND"}])";
    scene.textures = R"([{"source":0,"sampler":0},{"source":1}])";
    scene.samplers = R"([{"wrapS":33071,"wrapT":33648}])";
    scene.cameras = R"([{"type":"perspective","perspective":{"yfov":0.7,"znear":0.1,"zfar":100,"aspectRatio":1.5}}])";
    return scene;
}

bool near(vec3 const &a, vec3 const &b, float const tolerance = 1e-5f) { return tc::length(a - b) < tolerance; }

vec3 transformedPoint(core::mat4 const &matrix, vec3 const &point) { return (matrix * core::vec4{point, 1.0f}).xyz(); }

void checkMeshes(asset::SceneData const &scene) {
    REQUIRE(scene.meshes.size() == 2);
    auto const &quad = *scene.meshes[0].mesh;
    CHECK(quad.positions.size() == 4);
    auto const indexed = quad.indices == std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3};
    CHECK(indexed);
    CHECK(quad.uvs.size() == 4);
    auto const quadNormals = near(quad.normals.at(2), vec3{0.0f, 0.0f, 1.0f});
    CHECK(quadNormals);
    auto const &triangle = *scene.meshes[1].mesh;
    CHECK(triangle.positions.size() == 3);
    CHECK(triangle.uvs.empty());
    REQUIRE(triangle.normals.size() == 3);
    auto const flat = near(triangle.normals[0], vec3{0.0f, 0.0f, 1.0f}) and near(triangle.normals[2], triangle.normals[0]);
    CHECK(flat);
    CHECK(scene.meshes[0].material != scene.meshes[1].material);
}

void checkMaterials(asset::SceneData const &scene) {
    REQUIRE(scene.materials.size() == 2);
    auto const &painted = scene.materials[scene.meshes[0].material];
    auto const &glass = scene.materials[scene.meshes[1].material];
    auto const factors = near(painted.baseColor.xyz(), vec3{0.5f, 0.25f, 0.125f})
                     and near(painted.emissive, vec3{0.125f, 0.25f, 0.5f});
    CHECK(factors);
    CHECK(painted.metallic == 0.25f);
    CHECK(painted.roughness == 0.75f);
    CHECK(painted.alphaMode == core::AlphaMode::Mask);
    CHECK(painted.alphaCutoff == 0.25f);
    CHECK(painted.doubleSided);
    CHECK(painted.normalScale == 0.5f);
    auto const transform = painted.uvTransform.offset.x == 0.5f and painted.uvTransform.offset.y == 0.25f
                       and painted.uvTransform.rotation == 0.5f and painted.uvTransform.scale.x == 2.0f
                       and painted.uvTransform.scale.y == 3.0f;
    CHECK(transform);
    CHECK(glass.alphaMode == core::AlphaMode::Blend);
    CHECK(glass.baseColor.w == 0.5f);
    CHECK(glass.metallic == 1.0f);
    CHECK(not glass.doubleSided);

    // The colour image as sRGB for base colour and emission, the other as linear data for
    // normals and metallic-roughness: two textures.
    REQUIRE(scene.textures.size() == 2);
    REQUIRE(painted.baseColorTexture.has_value());
    REQUIRE(painted.normalTexture.has_value());
    REQUIRE(glass.emissiveTexture.has_value());
    REQUIRE(glass.metallicRoughnessTexture.has_value());
    CHECK(painted.baseColorTexture->value == glass.emissiveTexture->value);
    CHECK(painted.normalTexture->value == glass.metallicRoughnessTexture->value);
    auto const &colour = scene.textures[painted.baseColorTexture->value];
    auto const &data = scene.textures[painted.normalTexture->value];
    CHECK(colour.image->encoding == core::ColorEncoding::Srgb);
    CHECK(data.image->encoding == core::ColorEncoding::Linear);
    auto const texels = colour.image->levels.at(0) == kColourTexels;
    CHECK(texels);
    CHECK(colour.sampler.wrapU == core::TextureWrap::ClampToEdge);
    CHECK(colour.sampler.wrapV == core::TextureWrap::MirroredRepeat);
    CHECK(data.sampler.wrapU == core::TextureWrap::Repeat);
}

void checkNodes(asset::SceneData const &scene) {
    REQUIRE(scene.instances.size() == 2);
    // The root's translation after the quad's quarter turn about +y, and the triangle's scale.
    auto const quadCorner = transformedPoint(scene.instances[0].objectToWorld, vec3{1.0f, 0.0f, 0.0f});
    auto const triangleCorner = transformedPoint(scene.instances[1].objectToWorld, vec3{1.0f, 0.0f, 0.0f});
    auto const placed = near(quadCorner, vec3{1.0f, 2.0f, 2.0f}) and near(triangleCorner, vec3{3.0f, 2.0f, 3.0f});
    CHECK(placed);
    auto const bounded = near(scene.bounds.min, vec3{1.0f, 1.0f, 2.0f}) and near(scene.bounds.max, vec3{3.0f, 4.0f, 4.0f});
    CHECK(bounded);

    REQUIRE(scene.lights.size() == 2);
    auto const *const sun = std::get_if<core::DirectionalLight>(&scene.lights[0]);
    auto const *const lamp = std::get_if<core::PointLight>(&scene.lights[1]);
    REQUIRE(sun != nullptr);
    REQUIRE(lamp != nullptr);
    auto const sunlit = near(sun->direction, vec3{0.0f, -1.0f, 0.0f}) and near(sun->color, vec3{1.0f, 0.5f, 0.25f})
                    and sun->illuminance == 3.0f;
    CHECK(sunlit);
    auto const lamplit = near(lamp->position, vec3{0.0f, 4.0f, 0.0f}) and near(lamp->color, vec3{0.25f, 0.5f, 1.0f})
                     and lamp->intensity == 10.0f and lamp->range == 20.0f;
    CHECK(lamplit);

    REQUIRE(scene.cameras.size() == 1);
    auto const &camera = scene.cameras[0];
    auto const looking = near(camera.camera.position, vec3{0.0f, 0.0f, 5.0f})
                     and near(camera.camera.target, vec3{0.0f, 0.0f, 4.0f}) and near(camera.camera.up, vec3{0.0f, 1.0f, 0.0f});
    CHECK(looking);
    CHECK(camera.camera.verticalFov == doctest::Approx(0.7f));
    CHECK(camera.camera.nearPlane == doctest::Approx(0.1f));
    CHECK(camera.camera.farPlane == doctest::Approx(100.0f));
    REQUIRE(camera.aspectRatio.has_value());
    CHECK(*camera.aspectRatio == doctest::Approx(1.5f));
}

TEST_CASE("glTF scenes load meshes, materials, transforms, lights and cameras from every layout") {
    for (auto const layout : {test::GltfLayout::Embedded, test::GltfLayout::External, test::GltfLayout::Binary}) {
        CAPTURE(static_cast<int>(layout));
        auto const files = test::writeGltf(testScene(), layout, "models/test");
        auto harness = AssetHarness{files.files};

        auto const loaded = loadNow(harness.library->loadScene(files.main));

        REQUIRE(loaded.has_value());
        auto const &scene = **loaded;
        CHECK(scene.warnings.empty());
        checkMeshes(scene);
        checkMaterials(scene);
        checkNodes(scene);
        CHECK(harness.testSource().readsOf("models/test.bin") == (layout == test::GltfLayout::External ? 1u : 0u));
    }
}

TEST_CASE("loadMesh merges a glTF scene's meshes into one in scene space") {
    auto const files = test::writeGltf(testScene(), test::GltfLayout::External, "test");
    auto harness = AssetHarness{files.files};

    auto const loaded = loadNow(harness.library->loadMesh(files.main));

    REQUIRE(loaded.has_value());
    auto const &mesh = **loaded;
    CHECK(mesh.positions.size() == 7);
    CHECK(mesh.indices.size() == 9);
    CHECK(mesh.uvs.empty()); // the triangle has none
    auto const placed = near(mesh.positions[4], vec3{1.0f, 2.0f, 3.0f}) and near(mesh.positions[5], vec3{3.0f, 2.0f, 3.0f});
    CHECK(placed);
    // Without textures, the images are not even read.
    CHECK(harness.testSource().readsOf("test_image0.png") == 0);
}

TEST_CASE("a scene becomes a render world delta with the ids an allocator hands out, and a removal of them") {
    auto const files = test::writeGltf(testScene(), test::GltfLayout::Embedded, "test");
    auto harness = AssetHarness{files.files};
    auto const loaded = loadNow(harness.library->loadScene(files.main));
    REQUIRE(loaded.has_value());
    auto ids = asset::makeSequentialIds(100);

    auto const result = asset::sceneDelta(**loaded, *ids);

    auto const &delta = result.delta;
    REQUIRE(delta.meshes.size() == 2);
    REQUIRE(delta.textures.size() == 2);
    REQUIRE(delta.materials.size() == 2);
    REQUIRE(delta.instances.size() == 2);
    CHECK(delta.lights.size() == 2);
    CHECK(delta.meshes[0].id.value == 100);
    CHECK(delta.instances[1].id.value == 101);
    auto const drawsTriangle = delta.instances[1].mesh == delta.meshes[1].id;
    CHECK(drawsTriangle);
    auto const &painted = delta.materials[(*loaded)->meshes[0].material].material;
    REQUIRE(painted.baseColorTexture.has_value());
    CHECK(painted.baseColorTexture->value >= 100);
    CHECK(result.removal.meshes.size() == 2);
    CHECK(result.removal.textures.size() == 2);
    CHECK(result.removal.materials.size() == 2);
    CHECK(result.removal.instances.size() == 2);
    CHECK(result.removal.lights.size() == 2);
    auto const next = asset::sceneDelta(**loaded, *ids);
    CHECK(next.delta.meshes[0].id.value == 102);
}

TEST_CASE("missing and broken glTF files come back as AssetError values") {
    auto broken = test::writeGltf(testScene(), test::GltfLayout::External, "broken");
    broken.files.erase("broken.bin");
    auto files = broken.files;
    files["garbage.gltf"] = test::Bytes{'{', 'n', 'o', 't'};
    auto truncated = test::writeGltf(testScene(), test::GltfLayout::Binary, "truncated");
    auto &glb = truncated.files[truncated.main];
    glb.resize(glb.size() / 2);
    files[truncated.main] = glb;
    auto harness = AssetHarness{files};
    auto &library = *harness.library;

    auto const missing = loadNow(library.loadScene("missing.gltf"));
    auto const withoutBuffer = loadNow(library.loadScene(broken.main));
    auto const garbage = loadNow(library.loadScene("garbage.gltf"));
    auto const cut = loadNow(library.loadScene(truncated.main));
    auto const notMesh = loadNow(library.loadMesh("garbage.gltf"));

    CHECK(test::isError(missing, AssetErrorKind::NotFound));
    CHECK(test::isError(withoutBuffer, AssetErrorKind::NotFound));
    CHECK(test::isError(garbage, AssetErrorKind::Corrupt));
    CHECK(test::isError(cut, AssetErrorKind::Corrupt));
    CHECK(test::isError(notMesh, AssetErrorKind::Corrupt));
    REQUIRE(not withoutBuffer);
    CAPTURE(withoutBuffer.error().message);
    CHECK(withoutBuffer.error().message.find("broken.bin") != std::string::npos);
}

} // namespace
