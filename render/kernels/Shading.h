#pragma once

// Shades the visibility buffer: each pixel finds its triangle, intersects the camera ray
// with it for perspective-correct barycentrics, and evaluates glTF's metallic-roughness
// model under the scene's lights and the uniform environment.

#include <lr/render/GpuTypes.h>

#include <array>

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
    float vDotH = max(dot(toViewer, halfway), 0.0f);
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

struct TC_KERNEL ShadeVisibility {
    static constexpr char fileLocation[] = "lr_shade_visibility";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    InstanceTable<1> instances;
    MeshTable<2> meshes;
    VertexTable<3> vertices;
    BufferBinding<uint, 4> indices;
    MaterialTable<5> materials;
    LightTable<6> lights;
    BufferBinding<uvec2, 7> visible;
    BufferBinding<uint, 8> triangles;
    Uniform<mat4, 0> inverseViewProjection{};
    Uniform<vec4, 1> environment{}; // radiance in xyz
    Uniform<ivec2, 2> size{};
    Uniform<uint, 3> lightCount{};

    uint findSlot(uint triangle) {
        uint low = 0u;
        uint high = visible[0].x;
        while (high - low > 1u) {
            uint middle = (low + high) / 2u;
            if (visible[1u + middle].y <= triangle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }

    vec3 worldPoint(vec2 ndc, float depth) {
        vec4 point = inverseViewProjection.value * vec4(ndc, depth, 1.0f);
        return point.xyz() / point.w;
    }

    vec3 shadeTriangle(uint triangle, vec3 origin, vec3 direction) {
        uint slot = findSlot(triangle);
        uvec2 entry = visible[1u + slot];
        uint instance = entry.x;
        uint meshIndex = instances[instance].mesh;
        uint materialIndex = instances[instance].material;
        mat4 objectToWorld = instances[instance].objectToWorld.value;
        mat4 worldToObject = instances[instance].worldToObject.value;
        GpuMesh mesh = meshes[meshIndex];
        GpuMaterial material = materials[materialIndex];
        uint first = mesh.firstIndex + 3u * (triangle - entry.y);
        std::array<vec3, 3> positions;
        std::array<vec3, 3> normals;
        for (uint k = 0u; k < 3u; ++k) {
            uint vertex = mesh.firstVertex + indices[first + k];
            positions[k] = (objectToWorld * vec4(vertices[vertex].position, 1.0f)).xyz();
            normals[k] = vertices[vertex].normal.value;
        }
        // Moller-Trumbore, without rejection: the rasterizer already found the triangle.
        vec3 edge1 = positions[1] - positions[0];
        vec3 edge2 = positions[2] - positions[0];
        vec3 p = cross(direction, edge2);
        float determinant = dot(edge1, p);
        vec3 weights = vec3(1.0f / 3.0f);
        if (abs(determinant) > 1e-12f) {
            vec3 t = origin - positions[0];
            float u = dot(t, p) / determinant;
            float v = dot(direction, cross(t, edge1)) / determinant;
            weights = clamp(vec3{1.0f - u - v, u, v}, vec3(0.0f), vec3(1.0f));
            weights = weights / (weights.x + weights.y + weights.z);
        }
        vec3 position = positions[0] * weights.x + positions[1] * weights.y + positions[2] * weights.z;
        vec3 objectNormal = normals[0] * weights.x + normals[1] * weights.y + normals[2] * weights.z;
        vec3 normal = normalize(transpose(mat3(worldToObject)) * objectNormal);
        vec3 toViewer = -direction;
        // Seen from behind (double-sided materials only), the surface faces the other way.
        if (dot(cross(edge1, edge2), toViewer) < 0.0f) {
            normal = -normal;
        }
        vec3 baseColor = material.baseColor.xyz();
        vec3 radiance = material.emissive + environment.value.xyz() * baseColor * (1.0f - material.metallic);
        for (uint i = 0u; i < lightCount.value; ++i) {
            GpuLight light = lights[i];
            vec3 toLight = -light.vector;
            vec3 irradiance = light.radiance;
            if (light.kind == lightPoint()) {
                vec3 offset = light.vector - position;
                float distance2 = max(dot(offset, offset), 1e-8f);
                float distance = sqrt(distance2);
                toLight = offset / distance;
                irradiance = light.radiance * (rangeWindow(distance, light.range) / distance2);
            }
            radiance += reflectedLight(normal, toViewer, normalize(toLight), irradiance, baseColor, material.metallic,
                                       material.roughness);
        }
        return radiance;
    }

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint triangle = triangles[uint(pixel.y * size.value.x + pixel.x)];
        vec3 radiance = environment.value.xyz();
        if (triangle != noTriangle()) {
            vec2 ndc = vec2{(float(pixel.x) + 0.5f) / float(size.value.x) * 2.0f - 1.0f,
                            1.0f - (float(pixel.y) + 0.5f) / float(size.value.y) * 2.0f};
            vec3 origin = worldPoint(ndc, 0.0f);
            vec3 direction = normalize(worldPoint(ndc, 1.0f) - origin);
            radiance = shadeTriangle(triangle, origin, direction);
        }
        imageStore(color, pixel, vec4(radiance, 1.0f));
    }
};

} // namespace lr::render::kernels
