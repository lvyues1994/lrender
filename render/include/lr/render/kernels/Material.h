#pragma once

// Materials as kernels sample them: the texture pools, texture coordinates, levels of
// detail and normal mapping. compute_shaders only finds bindings that are members of the
// kernel itself, so kernels that sample materials expand LR_MATERIAL_TEXTURES, which
// declares the pool bindings and the member functions that sample them.

#include <lr/render/GpuTypes.h>

namespace lr::render::kernels {

using namespace tc;

// Clamped and mirrored coordinates are folded in the kernel, so pools always repeat.
template <uint B>
using ColorPoolTexture =
    TextureBinding<ColorPoolFormat, Dim::D2Array, ColorPoolPixel, B, Filter::Linear, Wrap::Repeat, Mip::Linear>;
template <uint B>
using CompressedPoolTexture = CompressedTextureBinding<B, Filter::Linear, Wrap::Repeat, Mip::Linear, Dim::D2Array>;
template <uint B>
using EnvironmentBrdfTexture =
    TextureBinding<EnvironmentBrdfFormat, Dim::D2, EnvironmentBrdfPixel, B, Filter::Linear, Wrap::ClampToEdge, Mip::None>;

inline float srgbToLinear(float encoded) {
    if (encoded <= 0.04045f) {
        return encoded / 12.92f;
    }
    return pow((encoded + 0.055f) / 1.055f, 2.4f);
}

inline float linearToSrgb(float linear) {
    float value = clamp(linear, 0.0f, 1.0f);
    if (value <= 0.0031308f) {
        return value * 12.92f;
    }
    return 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

// A coordinate as the texture's wrap mode reads it, `inset` inside the edges unless it
// repeats.
inline float wrapCoordinate(float coordinate, uint wrap, float inset) {
    if (wrap == wrapRepeat()) {
        return coordinate;
    }
    float folded = coordinate;
    if (wrap == wrapMirroredRepeat()) {
        float period = coordinate - 2.0f * floor(coordinate * 0.5f);
        folded = period > 1.0f ? 2.0f - period : period;
    }
    return clamp(folded, inset, 1.0f - inset);
}

struct TextureLookup {
    vec3 at; // texture coordinates and layer
    float lod;
};

// `footprint` is how far apart, in texture coordinates, neighbouring pixels see the
// surface: the larger of the distances along the rows and down the columns; 0 samples
// level 0. Clamped and mirrored coordinates keep half a texel of the coarser of the two
// levels trilinear filtering blends inside the edges, so that the repeating pool does
// not blend in the opposite edge.
inline TextureLookup textureLookup(uint ref, vec2 uv, float footprint) {
    float sizeLog2 = float(textureSizeLog2(ref));
    float lod = footprint > 0.0f ? max(log2(footprint) + sizeLog2, 0.0f) : 0.0f;
    float inset = 0.5f * exp2(min(ceil(lod), sizeLog2) - sizeLog2);
    return TextureLookup{vec3{wrapCoordinate(uv.x, textureWrapU(ref), inset),
                              wrapCoordinate(uv.y, textureWrapV(ref), inset), float(textureLayer(ref))},
                         lod};
}

inline vec4 decodeTexel(uint ref, vec4 texel) {
    if (textureIsSrgb(ref)) {
        return vec4{srgbToLinear(texel.x), srgbToLinear(texel.y), srgbToLinear(texel.z), texel.w};
    }
    return texel;
}

inline vec2 materialUv(GpuMaterial material, vec2 uv) {
    vec3 at = vec3(uv, 1.0f);
    return vec2{dot(material.uvRow0, at), dot(material.uvRow1, at)};
}

struct MaterialSample {
    vec4 baseColor; // linear; alpha is coverage
    float metallic;
    float roughness;
    vec3 emissive;
    vec3 normal; // in tangent space: x along u, y towards decreasing v
};

inline MaterialSample combineMaterial(GpuMaterial material, vec4 baseColor, vec4 metallicRoughness, vec4 emissive,
                                      vec4 normal) {
    vec3 tangentNormal = vec3{0.0f, 0.0f, 1.0f};
    if (material.textures.w != noTexture()) {
        vec3 stored = normal.xyz() * 2.0f - vec3(1.0f);
        tangentNormal = normalize(vec3{stored.x * material.normalScale, stored.y * material.normalScale, stored.z});
    }
    return MaterialSample{material.baseColor * baseColor, material.metallic * metallicRoughness.z,
                          material.roughness * metallicRoughness.y, material.emissive * emissive.xyz(),
                          tangentNormal};
}

// The directions texture coordinates u and v grow along in a triangle's plane, from its
// edges and their differences in texture coordinates; zero where the coordinates are
// degenerate.
struct UvAxes {
    vec3 alongU;
    vec3 alongV;
};

inline UvAxes uvAxes(vec3 edge1, vec3 edge2, vec2 uvEdge1, vec2 uvEdge2) {
    float determinant = uvEdge1.x * uvEdge2.y - uvEdge2.x * uvEdge1.y;
    if (abs(determinant) < 1e-20f) {
        return UvAxes{vec3(0.0f), vec3(0.0f)};
    }
    float inverse = 1.0f / determinant;
    return UvAxes{(edge1 * uvEdge2.y - edge2 * uvEdge1.y) * inverse, (edge2 * uvEdge1.x - edge1 * uvEdge2.x) * inverse};
}

// `normal` with a tangent-space normal applied. glTF's tangent space has +Y up in the
// image, towards decreasing v, whichever way the coordinates wind.
inline vec3 perturbNormal(vec3 normal, UvAxes axes, vec3 tangentNormal) {
    vec3 tangent = axes.alongU - normal * dot(normal, axes.alongU);
    if (dot(tangent, tangent) < 1e-20f) {
        return normal;
    }
    tangent = normalize(tangent);
    vec3 bitangent = cross(normal, tangent);
    if (dot(bitangent, axes.alongV) > 0.0f) {
        bitangent = -bitangent;
    }
    return normalize(tangent * tangentNormal.x + bitangent * tangentNormal.y + normal * tangentNormal.z);
}

// Barycentric weights of where a ray meets the plane of the triangle (a, a + edge1,
// a + edge2), outside the triangle too; all zero when the ray runs parallel to it.
inline vec3 planeWeights(vec3 origin, vec3 direction, vec3 a, vec3 edge1, vec3 edge2) {
    vec3 p = cross(direction, edge2);
    float determinant = dot(edge1, p);
    if (abs(determinant) < 1e-20f) {
        return vec3(0.0f);
    }
    vec3 offset = origin - a;
    float u = dot(offset, p) / determinant;
    float v = dot(direction, cross(offset, edge1)) / determinant;
    return vec3{1.0f - u - v, u, v};
}

// What GGX with Schlick's Fresnel reflects of a uniform environment, as a fraction of it:
// F0 A + B, with (A, B) from the table over (n . v, roughness).
inline vec3 environmentSpecular(vec2 scaleBias, vec3 f0) { return f0 * scaleBias.x + vec3(scaleBias.y); }

// Where the environment table holds (n . v, roughness): its first and last texel centres
// are at 0 and 1.
inline vec2 environmentBrdfAt(float nDotV, float roughness) {
    float size = float(environmentBrdfSize());
    vec2 at = clamp(vec2{nDotV, roughness}, vec2(0.0f), vec2(1.0f));
    return (at * (size - 1.0f) + vec2(0.5f)) / size;
}

} // namespace lr::render::kernels

// Bindings FIRST to FIRST + texturePoolSlots() - 1: the colour pools, then the compressed
// ones. MaterialTextures binds them (bindMaterialTextures in Bind.h).
#define LR_MATERIAL_TEXTURES(FIRST)                                                                                    \
    ColorPoolTexture<(FIRST) + 0u> colorPool0;                                                                         \
    ColorPoolTexture<(FIRST) + 1u> colorPool1;                                                                         \
    ColorPoolTexture<(FIRST) + 2u> colorPool2;                                                                         \
    ColorPoolTexture<(FIRST) + 3u> colorPool3;                                                                         \
    ColorPoolTexture<(FIRST) + 4u> colorPool4;                                                                         \
    ColorPoolTexture<(FIRST) + 5u> colorPool5;                                                                         \
    ColorPoolTexture<(FIRST) + 6u> colorPool6;                                                                         \
    ColorPoolTexture<(FIRST) + 7u> colorPool7;                                                                         \
    CompressedPoolTexture<(FIRST) + 8u> compressedPool0;                                                               \
    CompressedPoolTexture<(FIRST) + 9u> compressedPool1;                                                               \
    CompressedPoolTexture<(FIRST) + 10u> compressedPool2;                                                              \
    CompressedPoolTexture<(FIRST) + 11u> compressedPool3;                                                              \
                                                                                                                       \
    vec4 samplePool(uint slot, vec3 at, float lod) {                                                                   \
        if (slot == 0u) {                                                                                              \
            return textureLod(colorPool0, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 1u) {                                                                                              \
            return textureLod(colorPool1, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 2u) {                                                                                              \
            return textureLod(colorPool2, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 3u) {                                                                                              \
            return textureLod(colorPool3, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 4u) {                                                                                              \
            return textureLod(colorPool4, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 5u) {                                                                                              \
            return textureLod(colorPool5, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 6u) {                                                                                              \
            return textureLod(colorPool6, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 7u) {                                                                                              \
            return textureLod(colorPool7, at, lod);                                                                    \
        }                                                                                                              \
        if (slot == 8u) {                                                                                              \
            return textureLod(compressedPool0, at, lod);                                                               \
        }                                                                                                              \
        if (slot == 9u) {                                                                                              \
            return textureLod(compressedPool1, at, lod);                                                               \
        }                                                                                                              \
        if (slot == 10u) {                                                                                             \
            return textureLod(compressedPool2, at, lod);                                                               \
        }                                                                                                              \
        return textureLod(compressedPool3, at, lod);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    vec4 sampleTexture(uint ref, vec2 uv, float footprint, vec4 absent) {                                              \
        if (ref == noTexture()) {                                                                                      \
            return absent;                                                                                             \
        }                                                                                                              \
        TextureLookup lookup = textureLookup(ref, uv, footprint);                                                      \
        return decodeTexel(ref, samplePool(textureSlot(ref), lookup.at, lookup.lod));                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* One call site of sampleTexture(), so drivers compile each pool's sampling once. */                             \
    MaterialSample sampleMaterial(GpuMaterial material, vec2 uv, float footprint) {                                    \
        std::array<vec4, 4> texels;                                                                                    \
        for (uint k = 0u; k < 4u; ++k) {                                                                               \
            texels[k] = sampleTexture(material.textures[k], uv, footprint,                                             \
                                      k == 3u ? vec4{0.5f, 0.5f, 1.0f, 1.0f} : vec4(1.0f));                            \
        }                                                                                                              \
        return combineMaterial(material, texels[0], texels[1], texels[2], texels[3]);                                  \
    }
