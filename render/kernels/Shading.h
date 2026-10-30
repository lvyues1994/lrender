#pragma once

// Shades the visibility buffer: each pixel finds its triangle, intersects the camera ray
// with it for perspective-correct barycentrics, and evaluates glTF's metallic-roughness
// model under the scene's lights and the uniform environment.

#include "Lighting.h"

#include <lr/render/GpuTypes.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

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
            IncidentLight light = incidentLight(lights[i], position);
            radiance += reflectedLight(normal, toViewer, light.toLight, light.irradiance, baseColor, material.metallic,
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
