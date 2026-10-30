#include <lr/core/Id.h>
#include <lr/core/Math.h>
#include <lr/core/Primitives.h>
#include <lr/core/Scene.h>

#include <doctest/doctest.h>

#include <cmath>
#include <unordered_set>

namespace {

using namespace lr::core;

bool near(float const a, float const b, float const tolerance = 1e-5f) { return std::abs(a - b) <= tolerance; }

bool near(vec3 const &a, vec3 const &b, float const tolerance = 1e-5f) {
    return near(a.x, b.x, tolerance) and near(a.y, b.y, tolerance) and near(a.z, b.z, tolerance);
}

vec3 project(mat4 const &matrix, vec3 const &point) {
    auto const clip = matrix * vec4{point, 1.0f};
    return clip.xyz() / clip.w;
}

TEST_CASE("the camera puts its target at the centre and maps near and far to depth 0 and 1") {
    auto camera = CameraDesc{};
    camera.position = vec3{1.0f, 2.0f, 3.0f};
    camera.target = vec3{-1.0f, 0.5f, -2.0f};
    camera.nearPlane = 0.5f;
    camera.farPlane = 50.0f;
    auto const toClip = projectionMatrix(camera, 1.5f) * viewMatrix(camera);
    auto const forward = normalize(camera.target - camera.position);

    CHECK(near(project(toClip, camera.target).x, 0.0f));
    CHECK(near(project(toClip, camera.target).y, 0.0f));
    CHECK(near(project(toClip, camera.position + forward * 0.5f).z, 0.0f));
    CHECK(near(project(toClip, camera.position + forward * 50.0f).z, 1.0f, 1e-4f));
}

TEST_CASE("up in the world is up on the screen") {
    auto const camera = CameraDesc{};
    auto const toClip = projectionMatrix(camera, 1.0f) * viewMatrix(camera);

    CHECK(project(toClip, vec3{0.0f, 1.0f, 0.0f}).y > 0.0f);
    CHECK(project(toClip, vec3{1.0f, 0.0f, 0.0f}).x > 0.0f);
}

TEST_CASE("a transform scales, then rotates, then translates") {
    auto transform = Transform{};
    transform.translation = vec3{1.0f, 2.0f, 3.0f};
    transform.rotation = Rotation::axisAngle(vec3{0.0f, 0.0f, 1.0f}, kPi * 0.5f);
    transform.scale = vec3{2.0f, 1.0f, 1.0f};

    auto const moved = (matrixOf(transform) * vec4{1.0f, 0.0f, 0.0f, 1.0f}).xyz();

    CHECK(near(moved, vec3{1.0f, 4.0f, 3.0f}));
}

TEST_CASE("a transformed box contains every transformed corner") {
    auto const box = Aabb{vec3{-1.0f}, vec3{1.0f}};
    auto transform = Transform{};
    transform.rotation = Rotation::axisAngle(vec3{0.0f, 1.0f, 0.0f}, kPi * 0.25f);
    transform.translation = vec3{5.0f, 0.0f, 0.0f};

    auto const result = transformed(box, matrixOf(transform));

    auto const reach = std::sqrt(2.0f);
    CHECK(near(result.min, vec3{5.0f - reach, -1.0f, -reach}, 1e-4f));
    CHECK(near(result.max, vec3{5.0f + reach, 1.0f, reach}, 1e-4f));
}

TEST_CASE("ids of one kind compare and hash by value") {
    auto const ids = std::unordered_set<MeshId>{MeshId{1}, MeshId{2}, MeshId{1}};

    CHECK(ids.size() == 2);
    CHECK(MeshId{3} == MeshId{3});
    CHECK(not MeshId{}.isValid());
}

// Every triangle winds counter-clockwise seen from where its vertex normals point.
bool frontFacesOutwards(MeshData const &mesh) {
    for (auto index = std::size_t{0}; index + 2 < mesh.indices.size(); index += 3) {
        auto const a = mesh.positions[mesh.indices[index]];
        auto const b = mesh.positions[mesh.indices[index + 1]];
        auto const c = mesh.positions[mesh.indices[index + 2]];
        auto const faceNormal = cross(b - a, c - a);
        if (length(faceNormal) < 1e-7f) {
            continue; // the sphere's poles
        }
        if (dot(faceNormal, mesh.normals[mesh.indices[index]]) <= 0.0f) {
            return false;
        }
    }
    return true;
}

TEST_CASE("primitives wind their front faces outwards") {
    CHECK(frontFacesOutwards(makeBox(vec3{1.0f, 2.0f, 3.0f})));
    CHECK(frontFacesOutwards(makeSphere(1.0f, 16, 8)));
    CHECK(frontFacesOutwards(makePlane(vec2{1.0f, 1.0f})));
    CHECK(frontFacesOutwards(makeQuad(vec2{1.0f, 1.0f})));
}

TEST_CASE("a box has four vertices and two triangles per face") {
    auto const box = makeBox(vec3{1.0f});

    CHECK(box.positions.size() == 24);
    CHECK(box.indices.size() == 36);
    CHECK(near(boundsOf(box).min, vec3{-1.0f}));
    CHECK(near(boundsOf(box).max, vec3{1.0f}));
}

} // namespace
