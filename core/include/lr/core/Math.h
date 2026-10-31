#pragma once

#include <tc/glsl_math.hpp>
#include <tc/mat.hpp>
#include <tc/vec.hpp>

namespace lr::core {

// The kernels' vocabulary: host code that feeds kernels uses the same types.
using tc::ivec2;
using tc::mat3;
using tc::mat4;
using tc::uint;
using tc::uvec2;
using tc::uvec3;
using tc::vec2;
using tc::vec3;
using tc::vec4;

inline constexpr float kPi = 3.14159265358979323846f;

// A unit quaternion (x, y, z, w).
struct Rotation {
    static Rotation identity() noexcept { return Rotation{}; }
    // Counter-clockwise about `axis` (normalized here) when looking down the axis.
    static Rotation axisAngle(vec3 const &axis, float radians) noexcept;
    // The rotation whose matrix is `matrix`, which must be orthonormal with determinant 1.
    static Rotation fromMatrix(mat3 const &matrix) noexcept;
    // Turns -z towards `forward` and +y as close to `up` as it can, as a camera looks.
    static Rotation lookingAlong(vec3 const &forward, vec3 const &up) noexcept;

    vec4 quaternion{0.0f, 0.0f, 0.0f, 1.0f};
};

struct Transform {
    static Transform identity() noexcept { return Transform{}; }

    vec3 translation{0.0f};
    Rotation rotation;
    vec3 scale{1.0f};
};

struct Aabb {
    vec3 min{0.0f};
    vec3 max{0.0f};
};

mat3 rotationMatrix(Rotation const &rotation) noexcept;
mat4 matrixOf(Transform const &transform) noexcept;
// The transform whose matrix is `matrix`: exact for translations, rotations and scales
// along the axes (glTF's nodes); a matrix that also shears loses the shear.
Transform decomposed(mat4 const &matrix) noexcept;
// The box around `box` after `matrix` (an affine transform).
Aabb transformed(Aabb const &box, mat4 const &matrix) noexcept;

// Right-handed view space looking down -z. Clip depth runs from 0 at the near plane to 1
// at the far plane, so larger depth is farther away.
mat4 perspective(float verticalFov, float aspect, float nearPlane, float farPlane) noexcept;
mat4 lookAt(vec3 const &eye, vec3 const &target, vec3 const &up) noexcept;

} // namespace lr::core
