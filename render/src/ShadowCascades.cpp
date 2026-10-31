#include "ShadowCascades.h"

#include <lr/render/GpuTypes.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace lr::render {

namespace {

constexpr auto kLogarithmicShare = 0.75f;

float splitDistance(float const nearPlane, float const farDistance, std::uint32_t const index,
                    std::uint32_t const count) {
    auto const fraction = static_cast<float>(index) / static_cast<float>(count);
    auto const logarithmic = nearPlane * std::pow(farDistance / nearPlane, fraction);
    auto const uniform = nearPlane + (farDistance - nearPlane) * fraction;
    return kLogarithmicShare * logarithmic + (1.0f - kLogarithmicShare) * uniform;
}

// The corners of the view frustum between distances `nearDistance` and `farDistance`.
std::array<core::vec3, 8> sliceCorners(CascadeRequest const &request, float const nearDistance,
                                       float const farDistance) {
    auto const &camera = request.camera;
    auto const forward = tc::normalize(camera.target - camera.position);
    auto const side = tc::normalize(tc::cross(forward, camera.up));
    auto const up = tc::cross(side, forward);
    auto const tangent = std::tan(camera.verticalFov * 0.5f);
    auto corners = std::array<core::vec3, 8>{};
    for (auto index = 0u; index < 8u; ++index) {
        auto const distance = (index & 4u) != 0 ? farDistance : nearDistance;
        auto const x = ((index & 1u) != 0 ? 1.0f : -1.0f) * tangent * request.aspect * distance;
        auto const y = ((index & 2u) != 0 ? 1.0f : -1.0f) * tangent * distance;
        corners[index] = camera.position + forward * distance + side * x + up * y;
    }
    return corners;
}

struct LightBasis {
    core::vec3 side;
    core::vec3 up;
    core::vec3 forward;
};

LightBasis lightBasis(core::vec3 const &direction) {
    auto const reference = std::abs(direction.y) > 0.99f ? core::vec3{0.0f, 0.0f, -1.0f} : core::vec3{0.0f, 1.0f, 0.0f};
    auto const side = tc::normalize(tc::cross(direction, reference));
    return LightBasis{side, tc::cross(side, direction), direction};
}

// A matrix whose rows are `x`, `y`, `z` and (0, 0, 0, 1), each with its own offset.
core::mat4 rowsOf(core::vec4 const &x, core::vec4 const &y, core::vec4 const &z) {
    auto result = core::mat4{1.0f};
    for (auto column = 0; column < 4; ++column) {
        result[column][0] = x[column];
        result[column][1] = y[column];
        result[column][2] = z[column];
        result[column][3] = column == 3 ? 1.0f : 0.0f;
    }
    return result;
}

} // namespace

std::vector<ShadowCascade> fitShadowCascades(CascadeRequest const &request) {
    auto const &settings = request.settings;
    auto const count = std::clamp(settings.cascades, 1u, maxShadowCascades());
    auto const resolution = static_cast<float>(std::max(settings.resolution, 16u));
    auto const nearPlane = request.camera.nearPlane;
    auto const farDistance = std::clamp(settings.distance, nearPlane * 2.0f, request.camera.farPlane);
    auto const basis = lightBasis(request.lightDirection);
    auto const &bounds = request.sceneBounds;
    auto const hasBounds = bounds.min.x <= bounds.max.x;

    auto cascades = std::vector<ShadowCascade>{};
    for (auto index = 0u; index < count; ++index) {
        auto const sliceNear = splitDistance(nearPlane, farDistance, index, count);
        auto const sliceFar = splitDistance(nearPlane, farDistance, index + 1, count);
        auto const corners = sliceCorners(request, sliceNear, sliceFar);
        auto centre = core::vec3{0.0f};
        for (auto const &corner : corners) {
            centre += corner * 0.125f;
        }
        auto radius = 0.0f;
        for (auto const &corner : corners) {
            radius = std::max(radius, tc::length(corner - centre));
        }
        // Rounded up, so that rounding in the corners never changes the size.
        radius = std::ceil(radius * 16.0f) / 16.0f;
        auto const texel = 2.0f * radius / resolution;
        auto const centreX = std::floor(tc::dot(centre, basis.side) / texel) * texel;
        auto const centreY = std::floor(tc::dot(centre, basis.up) / texel) * texel;

        auto const centreDepth = tc::dot(centre, basis.forward);
        auto nearest = centreDepth - radius;
        auto farthest = centreDepth + radius;
        if (hasBounds) {
            for (auto corner = 0u; corner < 8u; ++corner) {
                auto const point = core::vec3{(corner & 1u) != 0 ? bounds.max.x : bounds.min.x,
                                              (corner & 2u) != 0 ? bounds.max.y : bounds.min.y,
                                              (corner & 4u) != 0 ? bounds.max.z : bounds.min.z};
                nearest = std::min(nearest, tc::dot(point, basis.forward));
                farthest = std::max(farthest, tc::dot(point, basis.forward));
            }
        }
        auto const margin = (farthest - nearest) * 0.01f + texel;
        nearest -= margin;
        farthest += margin;
        auto const depthScale = 1.0f / (farthest - nearest);

        auto const x = core::vec4{basis.side / radius, -centreX / radius};
        auto const y = core::vec4{basis.up / radius, -centreY / radius};
        auto const z = core::vec4{basis.forward * depthScale, -nearest * depthScale};
        auto const half = core::vec4{0.0f, 0.0f, 0.0f, 0.5f};
        auto cascade = ShadowCascade{};
        cascade.viewProjection = rowsOf(x, y, z);
        cascade.worldToShadow = rowsOf(x * 0.5f + half, y * -0.5f + half, z);
        cascade.texelSize = texel;
        cascade.depthScale = depthScale;
        cascade.farDistance = sliceFar;
        cascades.push_back(cascade);
    }
    return cascades;
}

} // namespace lr::render
