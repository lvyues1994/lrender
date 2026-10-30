#pragma once

#include <lr/core/Math.h>

#include <cstdint>
#include <variant>

namespace lr::core {

enum class AlphaMode : std::uint8_t {
    Opaque,
    Mask,
    Blend,
};

// glTF's metallic-roughness model.
struct MaterialDesc {
    vec4 baseColor{1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    vec3 emissive{0.0f}; // radiance
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};

// Parallel rays travelling along `direction`; `illuminance` falls on a surface facing them.
struct DirectionalLight {
    vec3 direction{0.0f, -1.0f, 0.0f};
    vec3 color{1.0f};
    float illuminance = 1.0f;
};

// Light from a point: `intensity` / distance^2 falls on a surface facing it. A positive
// `range` fades it out smoothly to nothing at that distance.
struct PointLight {
    vec3 position{0.0f};
    vec3 color{1.0f};
    float intensity = 1.0f;
    float range = 0.0f;
};

using LightDesc = std::variant<DirectionalLight, PointLight>;

// A uniform sky: what rays that hit nothing see, and the ambient light it casts.
struct EnvironmentDesc {
    vec3 radiance{0.0f};
};

struct CameraDesc {
    vec3 position{0.0f, 0.0f, 5.0f};
    vec3 target{0.0f};
    vec3 up{0.0f, 1.0f, 0.0f};
    float verticalFov = 0.8f; // radians
    float nearPlane = 0.05f;
    float farPlane = 500.0f;
};

mat4 viewMatrix(CameraDesc const &camera) noexcept;
mat4 projectionMatrix(CameraDesc const &camera, float aspect) noexcept;

} // namespace lr::core
