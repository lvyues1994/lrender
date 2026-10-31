#include <lr/core/Math.h>

#include <algorithm>
#include <cmath>

namespace lr::core {

Rotation Rotation::axisAngle(vec3 const &axis, float const radians) noexcept {
    auto const unit = tc::normalize(axis);
    auto const half = radians * 0.5f;
    auto const s = std::sin(half);
    return Rotation{vec4{unit.x * s, unit.y * s, unit.z * s, std::cos(half)}};
}

Rotation Rotation::fromMatrix(mat3 const &matrix) noexcept {
    // r(row, column); the largest of w, x, y, z is found first, so no division is by a small number.
    auto const r = [&matrix](int const row, int const column) { return matrix[column][row]; };
    auto const trace = r(0, 0) + r(1, 1) + r(2, 2);
    auto q = vec4{};
    if (trace > 0.0f) {
        auto const s = std::sqrt(trace + 1.0f) * 2.0f;
        q = vec4{(r(2, 1) - r(1, 2)) / s, (r(0, 2) - r(2, 0)) / s, (r(1, 0) - r(0, 1)) / s, 0.25f * s};
    } else if (r(0, 0) > r(1, 1) and r(0, 0) > r(2, 2)) {
        auto const s = std::sqrt(1.0f + r(0, 0) - r(1, 1) - r(2, 2)) * 2.0f;
        q = vec4{0.25f * s, (r(0, 1) + r(1, 0)) / s, (r(0, 2) + r(2, 0)) / s, (r(2, 1) - r(1, 2)) / s};
    } else if (r(1, 1) > r(2, 2)) {
        auto const s = std::sqrt(1.0f + r(1, 1) - r(0, 0) - r(2, 2)) * 2.0f;
        q = vec4{(r(0, 1) + r(1, 0)) / s, 0.25f * s, (r(1, 2) + r(2, 1)) / s, (r(0, 2) - r(2, 0)) / s};
    } else {
        auto const s = std::sqrt(1.0f + r(2, 2) - r(0, 0) - r(1, 1)) * 2.0f;
        q = vec4{(r(0, 2) + r(2, 0)) / s, (r(1, 2) + r(2, 1)) / s, 0.25f * s, (r(1, 0) - r(0, 1)) / s};
    }
    return Rotation{tc::normalize(q)};
}

Rotation Rotation::lookingAlong(vec3 const &forward, vec3 const &up) noexcept {
    auto const back = -tc::normalize(forward);
    auto const side = tc::normalize(tc::cross(up, back));
    return fromMatrix(mat3{side, tc::cross(back, side), back});
}

mat3 rotationMatrix(Rotation const &rotation) noexcept {
    auto const q = rotation.quaternion;
    auto const xx = q.x * q.x;
    auto const yy = q.y * q.y;
    auto const zz = q.z * q.z;
    auto const xy = q.x * q.y;
    auto const xz = q.x * q.z;
    auto const yz = q.y * q.z;
    auto const wx = q.w * q.x;
    auto const wy = q.w * q.y;
    auto const wz = q.w * q.z;
    return mat3{vec3{1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy)},
                vec3{2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx)},
                vec3{2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy)}};
}

mat4 matrixOf(Transform const &transform) noexcept {
    auto const r = rotationMatrix(transform.rotation);
    auto const s = transform.scale;
    return mat4{vec4{r[0] * s.x, 0.0f}, vec4{r[1] * s.y, 0.0f}, vec4{r[2] * s.z, 0.0f},
                vec4{transform.translation, 1.0f}};
}

Transform decomposed(mat4 const &matrix) noexcept {
    auto const x = matrix[0].xyz();
    auto const y = matrix[1].xyz();
    auto const z = matrix[2].xyz();
    // A mirroring matrix flips one axis: x's, by convention.
    auto const mirrored = tc::dot(tc::cross(x, y), z) < 0.0f;
    auto const scale = vec3{mirrored ? -tc::length(x) : tc::length(x), tc::length(y), tc::length(z)};
    auto transform = Transform{};
    transform.translation = matrix[3].xyz();
    transform.scale = scale;
    if (scale.x != 0.0f and scale.y != 0.0f and scale.z != 0.0f) {
        transform.rotation = Rotation::fromMatrix(mat3{x / scale.x, y / scale.y, z / scale.z});
    }
    return transform;
}

Aabb transformed(Aabb const &box, mat4 const &matrix) noexcept {
    auto result = Aabb{vec3{INFINITY}, vec3{-INFINITY}};
    for (auto corner = 0; corner < 8; ++corner) {
        auto const point = vec3{(corner & 1) != 0 ? box.max.x : box.min.x, (corner & 2) != 0 ? box.max.y : box.min.y,
                                (corner & 4) != 0 ? box.max.z : box.min.z};
        auto const moved = (matrix * vec4{point, 1.0f}).xyz();
        result.min = tc::min(result.min, moved);
        result.max = tc::max(result.max, moved);
    }
    return result;
}

mat4 perspective(float const verticalFov, float const aspect, float const nearPlane, float const farPlane) noexcept {
    auto const focal = 1.0f / std::tan(verticalFov * 0.5f);
    auto result = mat4{};
    result[0][0] = focal / aspect;
    result[1][1] = focal;
    result[2][2] = farPlane / (nearPlane - farPlane);
    result[2][3] = -1.0f;
    result[3][2] = nearPlane * farPlane / (nearPlane - farPlane);
    return result;
}

mat4 lookAt(vec3 const &eye, vec3 const &target, vec3 const &up) noexcept {
    auto const forward = tc::normalize(target - eye);
    auto const side = tc::normalize(tc::cross(forward, up));
    auto const upward = tc::cross(side, forward);
    auto result = mat4{1.0f};
    result[0][0] = side.x;
    result[1][0] = side.y;
    result[2][0] = side.z;
    result[0][1] = upward.x;
    result[1][1] = upward.y;
    result[2][1] = upward.z;
    result[0][2] = -forward.x;
    result[1][2] = -forward.y;
    result[2][2] = -forward.z;
    result[3][0] = -tc::dot(side, eye);
    result[3][1] = -tc::dot(upward, eye);
    result[3][2] = tc::dot(forward, eye);
    return result;
}

} // namespace lr::core
