#pragma once

#include <lr/core/Id.h>
#include <lr/core/Math.h>

#include <cstdint>
#include <optional>
#include <variant>

namespace lr::core {

enum class AlphaMode : std::uint8_t {
    Opaque,
    Mask,  // cut out where alpha < alphaCutoff
    Blend, // blended over what lies behind, by alpha
};

// glTF's KHR_texture_transform: texture coordinates are scaled, then rotated
// counter-clockwise by `rotation` radians, then offset.
struct UvTransform {
    vec2 offset{0.0f};
    float rotation = 0.0f;
    vec2 scale{1.0f};
};

// glTF's metallic-roughness model. A texture that does not exist, or has not finished
// streaming in, counts as absent: the factors apply alone. A material may name another
// model the renderer shades it with; that model reads these fields as it likes.
struct MaterialDesc {
    MaterialModelId model{0};        // 0: metallic-roughness
    vec4 modelParameters{0.0f};      // for other models; metallic-roughness ignores them
    vec4 baseColor{1.0f}; // factor; alpha is the coverage of Mask and Blend
    float metallic = 0.0f;
    float roughness = 0.5f;
    vec3 emissive{0.0f}; // radiance
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;

    std::optional<TextureId> baseColorTexture;         // times baseColor
    std::optional<TextureId> metallicRoughnessTexture; // G times roughness, B times metallic
    std::optional<TextureId> emissiveTexture;          // times emissive
    std::optional<TextureId> normalTexture;            // tangent space, +Y towards decreasing v
    float normalScale = 1.0f;                          // of the normal's x and y
    UvTransform uvTransform;                           // of every texture's coordinates
};

// Parallel rays travelling along `direction`; `illuminance` falls on a surface facing them.
// The visibility pipeline gives shadows to the first light in the scene that casts them;
// the path tracer shadows every light.
struct DirectionalLight {
    vec3 direction{0.0f, -1.0f, 0.0f};
    vec3 color{1.0f};
    float illuminance = 1.0f;
    bool castsShadows = false;
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
