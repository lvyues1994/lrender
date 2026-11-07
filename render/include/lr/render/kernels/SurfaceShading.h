#pragma once

// Shading what the visibility buffer holds. A pixel's camera ray meets its triangle where
// the kernel interpolates the surface, samples the material's textures there and hands the
// result, ShadingInputs, to a material model. The model decides what the surface is at
// that point, a ShadedSurface; the kernel then lights it with the scene's lights, the
// shadow map and the uniform environment, the same way for every model.
//
// Textures are sampled at the level of detail of the pixel's footprint: the rays through
// the neighbouring pixel centres meet the triangle's plane at texture coordinates whose
// distance from the pixel's own is the footprint (ray differentials).
//
// compute_shaders only finds bindings that are members of the kernel itself, so shading
// kernels expand LR_SURFACE_SHADING, which declares the bindings, the uniforms and the
// member functions that use them (bindSurfaceShading() in lr/render/Bind.h binds them).

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/Lighting.h>
#include <lr/render/kernels/Material.h>
#include <lr/render/kernels/VisibleInstances.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

inline uint noShadowLight() { return 0xFFFFFFFFu; }
inline uint noLightCell() { return 0xFFFFFFFFu; }

// What a material model gets of the point a pixel sees.
struct ShadingInputs {
    vec3 position;
    // Interpolated, with the normal map applied; like geometricNormal, it faces the viewer
    // where a double-sided surface is seen from behind.
    vec3 normal;
    vec3 geometricNormal;
    vec3 toViewer; // unit length
    vec2 uv;       // the material's texture coordinates, its UV transform applied
    GpuMaterial material;
    MaterialSample textures; // the material's factors times its textures at uv
    uint instance;
};

// What a material model makes of the point: the light it emits and the parameters of the
// surface the kernel lights (Lambert diffuse and GGX specular, see Lighting.h).
struct ShadedSurface {
    vec3 emitted; // radiance towards the viewer
    vec3 baseColor;
    float metallic;
    float roughness;
    vec3 normal;    // unit length, on the viewer's side
    float coverage; // blended materials' alpha
    bool reflects;  // false: lights and the environment leave the surface alone
};

// A surface that only emits `radiance`.
inline ShadedSurface emissionOnly(vec3 radiance) {
    return ShadedSurface{radiance, vec3(0.0f), 0.0f, 1.0f, vec3{0.0f, 0.0f, 1.0f}, 1.0f, false};
}

// The built-in model: glTF's metallic-roughness, as the material's factors and textures say.
struct MetallicRoughness {
    static ShadedSurface evaluate(ShadingInputs inputs) {
        MaterialSample sampled = inputs.textures;
        return ShadedSurface{sampled.emissive,  sampled.baseColor.xyz(), sampled.metallic, sampled.roughness,
                             inputs.normal,     sampled.baseColor.w,     true};
    }
};

// A pixel's camera ray and those of its neighbours to the right and below.
struct PixelRays {
    vec3 origin;
    vec3 direction;
    vec3 rightOrigin;
    vec3 rightDirection;
    vec3 downOrigin;
    vec3 downDirection;
};

// A weight of the bilinear 3x3 PCF kernel over a 4x4 block of texels, along one axis.
inline float pcfWeight(uint texel, float fraction) {
    if (texel == 0u) {
        return 1.0f - fraction;
    }
    if (texel == 3u) {
        return fraction;
    }
    return 1.0f;
}

} // namespace lr::render::kernels

// Bindings FIRST to FIRST + 19: the material textures (LR_MATERIAL_TEXTURES), the
// environment table, then the scene's tables, the view's light grid (lightGridHeaderWords()),
// the visible instances (VisibilityOutputs) and the shadow map (shadowHeaderWords()).
// Uniforms 0 to 4.
#define LR_SURFACE_SHADING(FIRST)                                                                                      \
    LR_MATERIAL_TEXTURES(FIRST)                                                                                        \
    EnvironmentBrdfTexture<(FIRST) + 12u> environmentBrdf;                                                             \
    InstanceTable<(FIRST) + 13u> instances;                                                                            \
    VertexTable<(FIRST) + 14u> vertices;                                                                               \
    BufferBinding<uint, (FIRST) + 15u> indices;                                                                        \
    MaterialTable<(FIRST) + 16u> materials;                                                                            \
    BufferBinding<uint, (FIRST) + 17u> lightGrid;                                                                      \
    BufferBinding<uvec4, (FIRST) + 18u> visible;                                                                       \
    BufferBinding<uint, (FIRST) + 19u> shadow;                                                                         \
    Uniform<mat4, 0> inverseViewProjection{};                                                                          \
    Uniform<vec4, 1> environment{}; /* radiance in xyz */                                                              \
    Uniform<ivec2, 2> size{};                                                                                          \
    Uniform<uint, 3> lightCount{};                                                                                     \
    Uniform<uint, 4> shadowLight{}; /* the light the shadow map is for, or noShadowLight() */                         \
                                                                                                                       \
    LR_FIND_VISIBLE_SLOT                                                                                               \
                                                                                                                       \
    vec3 worldPoint(vec2 ndc, float depth) {                                                                           \
        vec4 point = inverseViewProjection.value * vec4(ndc, depth, 1.0f);                                             \
        return point.xyz() / point.w;                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    float shadowFloat(uint word) { return uintBitsToFloat(shadow[word]); }                                             \
                                                                                                                       \
    float lightGridFloat(uint word) { return uintBitsToFloat(lightGrid[word]); }                                       \
                                                                                                                       \
    vec4 lightGridRow(uint row) {                                                                                      \
        uint base = 8u + 4u * row;                                                                                     \
        return vec4{lightGridFloat(base), lightGridFloat(base + 1u), lightGridFloat(base + 2u),                        \
                    lightGridFloat(base + 3u)};                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    GpuLight gridLight(uint index) {                                                                                   \
        uint base = lightGridHeaderWords() + lightRecordWords() * index;                                               \
        return GpuLight{lightGrid[base],                                                                               \
                        vec3{lightGridFloat(base + 1u), lightGridFloat(base + 2u), lightGridFloat(base + 3u)},         \
                        vec3{lightGridFloat(base + 4u), lightGridFloat(base + 5u), lightGridFloat(base + 6u)},         \
                        lightGridFloat(base + 7u)};                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    /* The light grid's cell that holds `position`, or noLightCell() off the screen. */                               \
    uint lightCellAt(vec3 position) {                                                                                  \
        vec4 row3 = lightGridRow(2u);                                                                                  \
        float w = dot(row3.xyz(), position) + row3.w;                                                                  \
        if (not(w > 0.0f)) {                                                                                           \
            return noLightCell();                                                                                      \
        }                                                                                                              \
        vec4 row0 = lightGridRow(0u);                                                                                  \
        vec4 row1 = lightGridRow(1u);                                                                                  \
        vec2 extent = vec2(size.value);                                                                                \
        vec2 pixel = vec2{((dot(row0.xyz(), position) + row0.w) / w * 0.5f + 0.5f) * extent.x,                        \
                          (0.5f - (dot(row1.xyz(), position) + row1.w) / w * 0.5f) * extent.y};                       \
        if (not(pixel.x >= 0.0f and pixel.y >= 0.0f and pixel.x < extent.x and pixel.y < extent.y)) {                 \
            return noLightCell();                                                                                      \
        }                                                                                                              \
        uint tilesAcross = lightGrid[0];                                                                               \
        uint tilesDown = lightGrid[1];                                                                                 \
        uint slices = lightGrid[2];                                                                                    \
        uint x = min(uint(pixel.x) / lightTileSize(), tilesAcross - 1u);                                               \
        uint y = min(uint(pixel.y) / lightTileSize(), tilesDown - 1u);                                                 \
        float near = lightGridFloat(4u);                                                                               \
        float slice = w > near ? log(w / near) * lightGridFloat(5u) : 0.0f;                                            \
        uint z = uint(min(slice, float(slices - 1u)));                                                                 \
        return (z * tilesDown + y) * tilesAcross + x;                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    mat4 shadowMatrix(uint cascade) {                                                                                  \
        uint base = shadowCascadeWord(cascade);                                                                        \
        return mat4{                                                                                                   \
            vec4{shadowFloat(base), shadowFloat(base + 1u), shadowFloat(base + 2u), shadowFloat(base + 3u)},           \
            vec4{shadowFloat(base + 4u), shadowFloat(base + 5u), shadowFloat(base + 6u), shadowFloat(base + 7u)},      \
            vec4{shadowFloat(base + 8u), shadowFloat(base + 9u), shadowFloat(base + 10u), shadowFloat(base + 11u)},    \
            vec4{shadowFloat(base + 12u), shadowFloat(base + 13u), shadowFloat(base + 14u), shadowFloat(base + 15u)}}; \
    }                                                                                                                  \
                                                                                                                       \
    /* The lit fraction of a 3x3 bilinear PCF kernel around `at` (in texels of the cascade). */                       \
    float filteredShadow(uint cascade, uint resolution, vec2 at, float depth) {                                        \
        vec2 texel = at - vec2(0.5f);                                                                                  \
        vec2 corner = floor(texel);                                                                                    \
        vec2 fraction = texel - corner;                                                                                \
        uint receiver = floatBitsToUint(max(depth, 0.0f));                                                             \
        uint first = shadowHeaderWords() + cascade * resolution * resolution;                                          \
        int last = int(resolution) - 1;                                                                                \
        float lit = 0.0f;                                                                                              \
        for (uint j = 0u; j < 4u; ++j) {                                                                               \
            int y = clamp(int(corner.y) - 1 + int(j), 0, last);                                                        \
            for (uint i = 0u; i < 4u; ++i) {                                                                           \
                int x = clamp(int(corner.x) - 1 + int(i), 0, last);                                                    \
                if (receiver <= shadow[first + uint(y) * resolution + uint(x)]) {                                      \
                    lit += pcfWeight(i, fraction.x) * pcfWeight(j, fraction.y);                                        \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return lit / 9.0f;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    /* How much of the shadowed light reaches `position`: the finest cascade that holds it                            \
       with room for the filter decides. The receiver moves along its normal and its depth                            \
       towards the light, both by fractions of a texel that grow with the slope, so that a                           \
       surface does not shadow itself. */                                                                              \
    float shadowFactor(vec3 position, vec3 geometric, vec3 toLight) {                                                  \
        uint cascades = shadow[0];                                                                                     \
        uint resolution = shadow[1];                                                                                   \
        float cosine = clamp(abs(dot(geometric, toLight)), 0.05f, 1.0f);                                               \
        float sine = sqrt(1.0f - cosine * cosine);                                                                     \
        float slope = min(sine / cosine, 4.0f);                                                                        \
        float side = float(resolution);                                                                                \
        float margin = 2.5f / side;                                                                                    \
        for (uint cascade = 0u; cascade < cascades; ++cascade) {                                                       \
            float texel = shadowFloat(shadowCascadeWord(cascade) + 16u);                                               \
            float depthPerUnit = shadowFloat(shadowCascadeWord(cascade) + 17u);                                        \
            vec3 receiver = position + geometric * (texel * sine);                                                     \
            vec4 at = shadowMatrix(cascade) * vec4(receiver, 1.0f);                                                    \
            if (at.x >= margin and at.x <= 1.0f - margin and at.y >= margin and at.y <= 1.0f - margin                  \
                and at.z <= 1.0f) {                                                                                    \
                float bias = texel * (0.5f + 1.5f * slope) * depthPerUnit;                                             \
                return filteredShadow(cascade, resolution, at.xy() * side, at.z - bias);                               \
            }                                                                                                          \
        }                                                                                                              \
        return 1.0f;                                                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    vec2 environmentScaleBias(float nDotV, float roughness) {                                                          \
        return textureLod(environmentBrdf, environmentBrdfAt(nDotV, roughness), 0.0f).xy();                            \
    }                                                                                                                  \
                                                                                                                       \
    PixelRays raysThrough(ivec2 pixel) {                                                                               \
        vec2 ndc = vec2{(float(pixel.x) + 0.5f) / float(size.value.x) * 2.0f - 1.0f,                                   \
                        1.0f - (float(pixel.y) + 0.5f) / float(size.value.y) * 2.0f};                                  \
        vec2 rightNdc = ndc + vec2{2.0f / float(size.value.x), 0.0f};                                                  \
        vec2 downNdc = ndc - vec2{0.0f, 2.0f / float(size.value.y)};                                                   \
        vec3 origin = worldPoint(ndc, 0.0f);                                                                           \
        vec3 rightOrigin = worldPoint(rightNdc, 0.0f);                                                                 \
        vec3 downOrigin = worldPoint(downNdc, 0.0f);                                                                   \
        return PixelRays{origin,                                                                                       \
                         normalize(worldPoint(ndc, 1.0f) - origin),                                                    \
                         rightOrigin,                                                                                  \
                         normalize(worldPoint(rightNdc, 1.0f) - rightOrigin),                                          \
                         downOrigin,                                                                                   \
                         normalize(worldPoint(downNdc, 1.0f) - downOrigin)};                                           \
    }                                                                                                                  \
                                                                                                                       \
    /* Where `rays.direction` meets visible triangle `triangle`. */                                                   \
    ShadingInputs surfaceAt(uint triangle, PixelRays rays) {                                                           \
        uint slot = findSlot(triangle);                                                                                \
        uvec4 entry = visible[1u + slot];                                                                              \
        uint instance = entry.x;                                                                                       \
        mat4 objectToWorld = instances[instance].objectToWorld.value;                                                  \
        mat4 worldToObject = instances[instance].worldToObject.value;                                                  \
        GpuMaterial material = materials[instances[instance].material];                                                \
        uint first = entry.z + 3u * (triangle - entry.y);                                                              \
        std::array<vec3, 3> positions;                                                                                 \
        std::array<vec3, 3> normals;                                                                                   \
        std::array<vec2, 3> uvs;                                                                                       \
        for (uint k = 0u; k < 3u; ++k) {                                                                               \
            uint vertex = entry.w + indices[first + k];                                                                \
            positions[k] = (objectToWorld * vec4(vertices[vertex].position, 1.0f)).xyz();                              \
            normals[k] = vertices[vertex].normal.value;                                                                \
            uvs[k] = materialUv(material, vertices[vertex].uv.value);                                                  \
        }                                                                                                              \
        vec3 edge1 = positions[1] - positions[0];                                                                      \
        vec3 edge2 = positions[2] - positions[0];                                                                      \
        /* The rasterizer found the triangle; rounding may put the ray just outside it. */                            \
        vec3 centre = planeWeights(rays.origin, rays.direction, positions[0], edge1, edge2);                           \
        vec3 weights = vec3(1.0f / 3.0f);                                                                              \
        if (centre != vec3(0.0f)) {                                                                                    \
            weights = clamp(centre, vec3(0.0f), vec3(1.0f));                                                           \
            weights = weights / (weights.x + weights.y + weights.z);                                                   \
        }                                                                                                              \
        vec2 uv = uvs[0] * weights.x + uvs[1] * weights.y + uvs[2] * weights.z;                                        \
        vec3 right = planeWeights(rays.rightOrigin, rays.rightDirection, positions[0], edge1, edge2);                  \
        vec3 down = planeWeights(rays.downOrigin, rays.downDirection, positions[0], edge1, edge2);                     \
        float footprint = 1e9f;                                                                                        \
        if (centre != vec3(0.0f) and right != vec3(0.0f) and down != vec3(0.0f)) {                                     \
            vec2 uvCentre = uvs[0] * centre.x + uvs[1] * centre.y + uvs[2] * centre.z;                                 \
            vec2 uvRight = uvs[0] * right.x + uvs[1] * right.y + uvs[2] * right.z;                                     \
            vec2 uvDown = uvs[0] * down.x + uvs[1] * down.y + uvs[2] * down.z;                                         \
            footprint = max(length(uvRight - uvCentre), length(uvDown - uvCentre));                                    \
        }                                                                                                              \
        MaterialSample sampled = sampleMaterial(material, uv, footprint);                                              \
                                                                                                                       \
        vec3 position = positions[0] * weights.x + positions[1] * weights.y + positions[2] * weights.z;                 \
        vec3 objectNormal = normals[0] * weights.x + normals[1] * weights.y + normals[2] * weights.z;                   \
        vec3 normal = normalize(transpose(mat3(worldToObject)) * objectNormal);                                        \
        normal = perturbNormal(normal, uvAxes(edge1, edge2, uvs[1] - uvs[0], uvs[2] - uvs[0]), sampled.normal);        \
        vec3 geometric = normalize(cross(edge1, edge2));                                                               \
        /* A mirroring transform reverses the winding the geometric normal comes from. */                             \
        if (determinant(mat3(worldToObject)) < 0.0f) {                                                                 \
            geometric = -geometric;                                                                                    \
        }                                                                                                              \
        vec3 toViewer = -rays.direction;                                                                               \
        /* Seen from behind (double-sided materials only), the surface faces the other way. */                       \
        if (dot(geometric, toViewer) < 0.0f) {                                                                         \
            normal = -normal;                                                                                          \
            geometric = -geometric;                                                                                    \
        }                                                                                                              \
        return ShadingInputs{position, normal, geometric, toViewer, uv, material, sampled, instance};                  \
    }                                                                                                                  \
                                                                                                                       \
    /* Radiance towards the viewer from `surface` at `inputs.position`. */                                            \
    vec3 lightSurface(ShadingInputs inputs, ShadedSurface surface) {                                                   \
        if (not surface.reflects) {                                                                                    \
            return surface.emitted;                                                                                    \
        }                                                                                                              \
        vec3 baseColor = surface.baseColor;                                                                            \
        vec3 f0 = mix(vec3(0.04f), baseColor, surface.metallic);                                                       \
        vec2 scaleBias = environmentScaleBias(max(dot(surface.normal, inputs.toViewer), 0.0f), surface.roughness);     \
        vec3 radiance = surface.emitted                                                                                \
                      + environment.value.xyz()                                                                        \
                            * (baseColor * (1.0f - surface.metallic) + environmentSpecular(scaleBias, f0));            \
        /* The lights the cell leaves out add exactly nothing: they are out of range. The                             \
           others come in the order of the light table, as if every light were added. */                              \
        uint count = lightCount.value;                                                                                 \
        uint words = lightMaskWords(count);                                                                            \
        uint cell = lightCellAt(inputs.position);                                                                      \
        uint masks = lightGridHeaderWords() + lightRecordWords() * count + cell * words;                               \
        for (uint word = 0u; word < words; ++word) {                                                                   \
            uint left = count - 32u * word;                                                                            \
            uint bits = cell != noLightCell() ? lightGrid[masks + word]                                                \
                      : left >= 32u           ? 0xFFFFFFFFu                                                            \
                                              : (1u << left) - 1u;                                                     \
            while (bits != 0u) {                                                                                       \
                uint i = 32u * word + uint(findLSB(bits));                                                             \
                bits &= bits - 1u;                                                                                     \
                IncidentLight light = incidentLight(gridLight(i), inputs.position);                                    \
                vec3 reflected = reflectedLight(surface.normal, inputs.toViewer, light.toLight, light.irradiance,      \
                                                baseColor, surface.metallic, surface.roughness);                       \
                if (i == shadowLight.value and reflected != vec3(0.0f)) {                                              \
                    reflected = reflected * shadowFactor(inputs.position, inputs.geometricNormal, light.toLight);      \
                }                                                                                                      \
                radiance += reflected;                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        return radiance;                                                                                               \
    }
