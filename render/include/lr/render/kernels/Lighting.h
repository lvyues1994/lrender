#pragma once

// The surface model and the lights every pipeline shades with: the visibility pipeline
// evaluates them once per pixel, the path tracer at every vertex of a path.

#include <lr/render/GpuTypes.h>

namespace lr::render::kernels {

using namespace tc;

// Radiance a surface reflects towards the viewer from light arriving from `toLight` with
// `irradiance` on a surface facing it: Lambert diffuse plus GGX specular (Smith
// height-correlated visibility, Schlick Fresnel).
inline vec3 reflectedLight(vec3 normal, vec3 toViewer, vec3 toLight, vec3 irradiance, vec3 baseColor, float metallic,
                           float roughness) {
    float pi = 3.14159265f;
    float nDotL = dot(normal, toLight);
    if (nDotL <= 0.0f) {
        return vec3(0.0f);
    }
    float nDotV = max(dot(normal, toViewer), 0.0001f);
    vec3 halfway = normalize(toLight + toViewer);
    float nDotH = max(dot(normal, halfway), 0.0f);
    // Rounding can push it past 1, and pow() of a negative base is undefined on GPUs.
    float vDotH = clamp(dot(toViewer, halfway), 0.0f, 1.0f);
    float alpha = max(roughness * roughness, 0.001f);
    float alpha2 = alpha * alpha;
    float d = nDotH * nDotH * (alpha2 - 1.0f) + 1.0f;
    float distribution = alpha2 / (pi * d * d);
    float visibility = 0.5f / (nDotL * sqrt(nDotV * nDotV * (1.0f - alpha2) + alpha2)
                               + nDotV * sqrt(nDotL * nDotL * (1.0f - alpha2) + alpha2));
    vec3 f0 = mix(vec3(0.04f), baseColor, metallic);
    vec3 fresnel = f0 + (vec3(1.0f) - f0) * pow(1.0f - vDotH, 5.0f);
    vec3 diffuse = baseColor * ((1.0f - metallic) / pi);
    return (diffuse + fresnel * (distribution * visibility)) * irradiance * nDotL;
}

// Point-light falloff to zero at `range` (glTF's KHR_lights_punctual recommendation).
inline float rangeWindow(float distance, float range) {
    if (range <= 0.0f) {
        return 1.0f;
    }
    float ratio = distance / range;
    float window = clamp(1.0f - ratio * ratio * ratio * ratio, 0.0f, 1.0f);
    return window * window;
}

// What a light casts on a point.
struct IncidentLight {
    vec3 toLight;      // unit length
    vec3 irradiance;   // on a surface facing the light
    float distance;    // to the light; directional lights are infinitely far
};

inline float infiniteDistance() { return 1e30f; }

inline IncidentLight incidentLight(GpuLight light, vec3 position) {
    if (light.kind == lightPoint()) {
        vec3 offset = light.vector - position;
        float distance2 = max(dot(offset, offset), 1e-8f);
        float distance = sqrt(distance2);
        return IncidentLight{normalize(offset / distance),
                             light.radiance * (rangeWindow(distance, light.range) / distance2), distance};
    }
    return IncidentLight{normalize(-light.vector), light.radiance, infiniteDistance()};
}

} // namespace lr::render::kernels
