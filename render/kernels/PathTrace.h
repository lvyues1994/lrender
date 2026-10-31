#pragma once

// The reference path tracer: one path per pixel per frame through two levels of
// hierarchies, shaded with the visibility pipeline's surface model and lights
// (Lighting.h), averaged into the view's accumulation history.
//
// Paths gather emission and the environment where they end, and the lights by next-event
// estimation with shadow rays at every vertex; they continue by sampling the diffuse or
// the GGX lobe of the BRDF (a one-sample mixture, weighted by its combined density) and
// end by Russian roulette. Random numbers are hashes of the pixel, the sample and the
// dimension, so every device draws the same ones.
//
// Textures are sampled at level 0. Masked materials are cut out where their alpha falls
// below the cutoff; blended ones are stochastically opaque: a ray, shadow rays included,
// passes through with probability 1 - alpha, so on average they let that much through.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/Lighting.h>
#include <lr/render/kernels/Material.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

inline uint maxBounces() { return 16u; }
inline uint noHit() { return 0xFFFFFFFFu; }

// The PCG hash (Jarzynski and Olano, "Hash Functions for GPU Rendering").
inline uint pcgHash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline uint randomBits(uint pixel, uint sampleNumber, uint dimension) {
    return pcgHash(pixel + pcgHash(sampleNumber + pcgHash(dimension)));
}

inline float unitFloat(uint bits) { return float(bits >> 8u) * (1.0f / 16777216.0f); }

// Uniform in [0, 1).
inline float randomNumber(uint pixel, uint sampleNumber, uint dimension) {
    return unitFloat(randomBits(pixel, sampleNumber, dimension));
}

// The dimensions a path's random numbers come from: 0 and 1 jitter the pixel, four per
// bounce sample the next direction, and each ray a vertex traces has one of its own,
// which decides the blended surfaces it passes.
inline uint traceDimension(uint bounce, uint ray) { return 1024u + 64u * bounce + ray; }

inline float maxComponent(vec3 v) { return max(max(v.x, v.y), v.z); }

inline float luminance(vec3 color) { return dot(color, vec3{0.2126f, 0.7152f, 0.0722f}); }

// An orthonormal frame whose third axis is `normal` (Duff et al., "Building an
// Orthonormal Basis, Revisited").
inline mat3 frameAround(vec3 normal) {
    float flip = normal.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (flip + normal.z);
    float b = normal.x * normal.y * a;
    return mat3{vec3{1.0f + flip * normal.x * normal.x * a, flip * b, -flip * normal.x},
                vec3{b, flip + normal.y * normal.y * a, -normal.y}, normal};
}

// A direction with density cos(theta) / pi about `normal`.
inline vec3 cosineDirection(vec3 normal, float u1, float u2) {
    float pi = 3.14159265f;
    float radius = sqrt(u1);
    float angle = 2.0f * pi * u2;
    vec3 local = vec3{radius * cos(angle), radius * sin(angle), sqrt(max(1.0f - u1, 0.0f))};
    return normalize(frameAround(normal) * local);
}

// A half vector with density D(h) (n . h) for GGX's distribution of normals.
inline vec3 ggxHalfVector(vec3 normal, float alpha, float u1, float u2) {
    float pi = 3.14159265f;
    float cos2 = (1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1);
    float angle = 2.0f * pi * u2;
    float sinTheta = sqrt(max(1.0f - cos2, 0.0f));
    vec3 local = vec3{sinTheta * cos(angle), sinTheta * sin(angle), sqrt(max(cos2, 0.0f))};
    return normalize(frameAround(normal) * local);
}

// The density, per solid angle, of the directions reflected about ggxHalfVector().
inline float ggxDensity(vec3 normal, vec3 toViewer, vec3 toLight, float alpha) {
    float pi = 3.14159265f;
    vec3 halfway = normalize(toLight + toViewer);
    float vDotH = dot(toViewer, halfway);
    if (vDotH <= 0.0f) {
        return 0.0f;
    }
    float nDotH = max(dot(normal, halfway), 0.0f);
    float alpha2 = alpha * alpha;
    float d = nDotH * nDotH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (pi * d * d) * nDotH / (4.0f * vDotH);
}

// How often a path continues by the specular lobe rather than the diffuse one: in
// proportion to their estimated weights, but never rarely for either.
inline float specularChance(vec3 baseColor, float metallic, float nDotV) {
    vec3 f0 = mix(vec3(0.04f), baseColor, metallic);
    float specular = luminance(f0 + (vec3(1.0f) - f0) * pow(1.0f - clamp(nDotV, 0.0f, 1.0f), 5.0f));
    float diffuse = luminance(baseColor) * (1.0f - metallic);
    if (diffuse <= 0.0f) {
        return 1.0f;
    }
    return clamp(specular / (specular + diffuse), 0.1f, 0.9f);
}

// Componentwise 1 / direction, finite: parallel rays then get huge, not infinite, slab
// distances, which avoids 0 * infinity in boxEntry().
inline vec3 safeInverse(vec3 direction) {
    float tiny = 1e-20f;
    return vec3(1.0f) / vec3{abs(direction.x) > tiny ? direction.x : tiny, abs(direction.y) > tiny ? direction.y : tiny,
                             abs(direction.z) > tiny ? direction.z : tiny};
}

// Where the ray enters the box, or -1 when it misses the box before `far`.
inline float boxEntry(vec3 origin, vec3 inverseDirection, vec3 low, vec3 high, float far) {
    vec3 t0 = (low - origin) * inverseDirection;
    vec3 t1 = (high - origin) * inverseDirection;
    vec3 nearest = min(t0, t1);
    vec3 farthest = max(t0, t1);
    float enter = max(max(nearest.x, nearest.y), max(nearest.z, 0.0f));
    // A few ulps wider, so that rounding keeps boxes that are flat along an axis.
    float leave = min(min(farthest.x, farthest.y), min(farthest.z, far)) * 1.0000004f;
    return enter <= leave ? enter : -1.0f;
}

// The child of an inner node to visit next and the one to come back to, nearest first;
// noBvhNode() for none.
struct ChildOrder {
    uint next;
    uint later;
};

inline ChildOrder orderChildren(uint first, float leftEntry, float rightEntry) {
    if (leftEntry < 0.0f) {
        return ChildOrder{rightEntry < 0.0f ? noBvhNode() : first + 1u, noBvhNode()};
    }
    if (rightEntry < 0.0f) {
        return ChildOrder{first, noBvhNode()};
    }
    return leftEntry <= rightEntry ? ChildOrder{first, first + 1u} : ChildOrder{first + 1u, first};
}

struct TriangleHit {
    float distance; // along the ray; -1 when it misses
    float u;        // barycentrics of corners b and c
    float v;
    // Positive when the triangle winds counter-clockwise seen from the ray's origin.
    float determinant;
};

// Moller-Trumbore, hits on either side.
inline TriangleHit intersectTriangle(vec3 origin, vec3 direction, vec3 a, vec3 b, vec3 c) {
    TriangleHit miss = TriangleHit{-1.0f, 0.0f, 0.0f, 0.0f};
    vec3 edge1 = b - a;
    vec3 edge2 = c - a;
    vec3 p = cross(direction, edge2);
    float determinant = dot(edge1, p);
    if (determinant == 0.0f) {
        return miss;
    }
    float inverse = 1.0f / determinant;
    vec3 offset = origin - a;
    float u = dot(offset, p) * inverse;
    if (u < 0.0f or u > 1.0f) {
        return miss;
    }
    vec3 q = cross(offset, edge1);
    float v = dot(direction, q) * inverse;
    if (v < 0.0f or u + v > 1.0f) {
        return miss;
    }
    float distance = dot(edge2, q) * inverse;
    return distance > 0.0f ? TriangleHit{distance, u, v, determinant} : miss;
}

struct RayHit {
    float distance;
    uint instance; // noHit() when the ray hit nothing
    uint triangle; // into the triangle table
    float u;       // barycentrics of corners 1 and 2
    float v;
};

struct SurfacePoint {
    vec3 position;
    vec3 normal;    // for shading
    vec3 geometric; // of the triangle; both face the side the ray came from
    MaterialSample material;
};

struct TC_KERNEL PathTrace {
    static constexpr char fileLocation[] = "lr_path_trace";
    uvec3 local_size{8, 8, 1};

    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 0> previous;
    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 1> current;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 2> color;
    LR_MATERIAL_TEXTURES(3u)
    InstanceTable<15> instances;
    VertexTable<16> vertices;
    MaterialTable<17> materials;
    LightTable<18> lights;
    BvhNodeTable<19> tlasNodes;
    BvhNodeTable<20> blasNodes;
    BvhTriangleTable<21> blasTriangles;
    BufferBinding<uint, 22> blasRoots;
    Uniform<mat4, 0> inverseViewProjection{};
    Uniform<vec4, 1> environment{}; // radiance in xyz
    Uniform<ivec2, 2> size{};
    Uniform<uint, 3> lightCount{};
    Uniform<uint, 4> tlasNodeCount{};
    Uniform<uint, 5> sampleIndex{}; // the samples `previous` averages

    // Whether a ray that meets a masked or blended surface at barycentrics (u, v) stops
    // there; `seed` decides for blended ones, independently for each triangle of each
    // instance (instances of one mesh share its triangles).
    bool stopsAt(GpuMaterial material, uvec3 corners, float u, float v, uint triangle, uint instance, uint seed) {
        vec2 uv = vertices[corners.x].uv.value * (1.0f - u - v) + vertices[corners.y].uv.value * u
                + vertices[corners.z].uv.value * v;
        float alpha = material.baseColor.w
                    * sampleTexture(material.textures.x, materialUv(material, uv), 0.0f, vec4(1.0f)).w;
        if ((material.flags & materialAlphaMask()) != 0u) {
            return alpha >= material.alphaCutoff;
        }
        return unitFloat(pcgHash(seed + pcgHash(triangle + pcgHash(instance)))) < alpha;
    }

    // The ray is in the object space of `instance`. Single-sided materials hide back
    // faces, which wind clockwise in world space: a mirroring transform reverses the
    // winding of object space.
    void testTriangle(uint triangle, uint instance, vec3 origin, vec3 direction, GpuMaterial material, float mirror,
                      uint seed, RayHit &hit) {
        uvec3 corners = blasTriangles[triangle].corners;
        TriangleHit found = intersectTriangle(origin, direction, vertices[corners.x].position,
                                              vertices[corners.y].position, vertices[corners.z].position);
        bool doubleSided = (material.flags & materialDoubleSided()) != 0u;
        bool partial = (material.flags & (materialAlphaMask() | materialAlphaBlend())) != 0u;
        if (found.distance > 0.0f and found.distance < hit.distance
            and (doubleSided or found.determinant * mirror > 0.0f)
            and (not partial or stopsAt(material, corners, found.u, found.v, triangle, instance, seed))) {
            hit = RayHit{found.distance, instance, triangle, found.u, found.v};
        }
    }

    // Object space keeps distances along the ray: its direction is transformed, not
    // normalized.
    void intersectInstance(uint instance, vec3 worldOrigin, vec3 worldDirection, bool anyHit, uint seed, RayHit &hit) {
        uint root = blasRoots[instances[instance].mesh];
        if (root == noBvhNode()) {
            return;
        }
        mat4 worldToObject = instances[instance].worldToObject.value;
        GpuMaterial material = materials[instances[instance].material];
        float mirror = determinant(mat3(worldToObject)) < 0.0f ? -1.0f : 1.0f;
        vec3 origin = (worldToObject * vec4(worldOrigin, 1.0f)).xyz();
        vec3 direction = mat3(worldToObject) * worldDirection;
        vec3 inverseDirection = safeInverse(direction);
        GpuBvhNode top = blasNodes[root];
        if (boxEntry(origin, inverseDirection, top.boundsMin, top.boundsMax, hit.distance) < 0.0f) {
            return;
        }
        std::array<uint, bvhMaxDepth() + 1u> stack;
        uint depth = 0u;
        uint node = root;
        while (true) {
            GpuBvhNode visited = blasNodes[node];
            uint next = noBvhNode();
            if (visited.count > 0u) {
                for (uint k = 0u; k < visited.count; ++k) {
                    testTriangle(visited.first + k, instance, origin, direction, material, mirror, seed, hit);
                }
                if (anyHit and hit.instance != noHit()) {
                    return;
                }
            } else {
                GpuBvhNode left = blasNodes[visited.first];
                GpuBvhNode right = blasNodes[visited.first + 1u];
                ChildOrder order = orderChildren(
                    visited.first, boxEntry(origin, inverseDirection, left.boundsMin, left.boundsMax, hit.distance),
                    boxEntry(origin, inverseDirection, right.boundsMin, right.boundsMax, hit.distance));
                next = order.next;
                if (order.later != noBvhNode()) {
                    stack[depth] = order.later;
                    depth += 1u;
                }
            }
            if (next == noBvhNode()) {
                if (depth == 0u) {
                    return;
                }
                depth -= 1u;
                next = stack[depth];
            }
            node = next;
        }
    }

    // The nearest hit closer than `far`, or with `anyHit` the first one found.
    RayHit trace(vec3 origin, vec3 direction, float far, bool anyHit, uint seed) {
        RayHit hit = RayHit{far, noHit(), 0u, 0.0f, 0.0f};
        if (tlasNodeCount.value == 0u) {
            return hit;
        }
        vec3 inverseDirection = safeInverse(direction);
        GpuBvhNode top = tlasNodes[0];
        if (boxEntry(origin, inverseDirection, top.boundsMin, top.boundsMax, far) < 0.0f) {
            return hit;
        }
        std::array<uint, bvhMaxDepth() + 1u> stack;
        uint depth = 0u;
        uint node = 0u;
        while (true) {
            GpuBvhNode visited = tlasNodes[node];
            uint next = noBvhNode();
            if (visited.count > 0u) {
                intersectInstance(visited.first, origin, direction, anyHit, seed, hit);
                if (anyHit and hit.instance != noHit()) {
                    return hit;
                }
            } else {
                GpuBvhNode left = tlasNodes[visited.first];
                GpuBvhNode right = tlasNodes[visited.first + 1u];
                ChildOrder order = orderChildren(
                    visited.first, boxEntry(origin, inverseDirection, left.boundsMin, left.boundsMax, hit.distance),
                    boxEntry(origin, inverseDirection, right.boundsMin, right.boundsMax, hit.distance));
                next = order.next;
                if (order.later != noBvhNode()) {
                    stack[depth] = order.later;
                    depth += 1u;
                }
            }
            if (next == noBvhNode()) {
                if (depth == 0u) {
                    return hit;
                }
                depth -= 1u;
                next = stack[depth];
            }
            node = next;
        }
        return hit;
    }

    // As the visibility pipeline shades it: interpolated and normal-mapped normals, and
    // surfaces seen from behind (double-sided materials only) facing the other way.
    SurfacePoint surfaceAt(RayHit hit, vec3 origin, vec3 direction) {
        uvec3 corners = blasTriangles[hit.triangle].corners;
        mat4 objectToWorld = instances[hit.instance].objectToWorld.value;
        mat4 worldToObject = instances[hit.instance].worldToObject.value;
        GpuMaterial material = materials[instances[hit.instance].material];
        vec3 a = (objectToWorld * vec4(vertices[corners.x].position, 1.0f)).xyz();
        vec3 b = (objectToWorld * vec4(vertices[corners.y].position, 1.0f)).xyz();
        vec3 c = (objectToWorld * vec4(vertices[corners.z].position, 1.0f)).xyz();
        vec2 uvA = materialUv(material, vertices[corners.x].uv.value);
        vec2 uvB = materialUv(material, vertices[corners.y].uv.value);
        vec2 uvC = materialUv(material, vertices[corners.z].uv.value);
        float w = 1.0f - hit.u - hit.v;
        MaterialSample sampled = sampleMaterial(material, uvA * w + uvB * hit.u + uvC * hit.v, 0.0f);
        vec3 objectNormal = vertices[corners.x].normal.value * w + vertices[corners.y].normal.value * hit.u
                          + vertices[corners.z].normal.value * hit.v;
        vec3 normal = normalize(transpose(mat3(worldToObject)) * objectNormal);
        normal = perturbNormal(normal, uvAxes(b - a, c - a, uvB - uvA, uvC - uvA), sampled.normal);
        vec3 geometric = normalize(cross(b - a, c - a));
        if (dot(geometric, direction) > 0.0f) {
            normal = -normal;
            geometric = -geometric;
        }
        return SurfacePoint{origin + direction * hit.distance, normal, geometric, sampled};
    }

    // The light the scene's lights reflect towards the viewer where nothing shadows them.
    vec3 directLight(SurfacePoint surface, vec3 toViewer, vec3 above, uint pixelIndex, uint sampleNumber, uint bounce) {
        vec3 reflected = vec3(0.0f);
        vec3 baseColor = surface.material.baseColor.xyz();
        for (uint i = 0u; i < lightCount.value; ++i) {
            IncidentLight light = incidentLight(lights[i], surface.position);
            vec3 contribution = reflectedLight(surface.normal, toViewer, light.toLight, light.irradiance, baseColor,
                                               surface.material.metallic, surface.material.roughness);
            if (contribution != vec3(0.0f)) {
                uint seed = randomBits(pixelIndex, sampleNumber, traceDimension(bounce, 1u + i));
                if (trace(above, light.toLight, light.distance * 0.9999f, true, seed).instance == noHit()) {
                    reflected += contribution;
                }
            }
        }
        return reflected;
    }

    vec3 worldPoint(vec2 ndc, float depth) {
        vec4 point = inverseViewProjection.value * vec4(ndc, depth, 1.0f);
        return point.xyz() / point.w;
    }

    vec3 tracePath(ivec2 pixel, uint pixelIndex, uint sampleNumber) {
        float pi = 3.14159265f;
        vec2 jitter = vec2{randomNumber(pixelIndex, sampleNumber, 0u), randomNumber(pixelIndex, sampleNumber, 1u)};
        vec2 ndc = vec2{(float(pixel.x) + jitter.x) / float(size.value.x) * 2.0f - 1.0f,
                        1.0f - (float(pixel.y) + jitter.y) / float(size.value.y) * 2.0f};
        vec3 origin = worldPoint(ndc, 0.0f);
        vec3 direction = normalize(worldPoint(ndc, 1.0f) - origin);
        vec3 radiance = vec3(0.0f);
        vec3 throughput = vec3(1.0f);
        for (uint bounce = 0u; bounce < maxBounces(); ++bounce) {
            RayHit hit =
                trace(origin, direction, infiniteDistance(), false,
                      randomBits(pixelIndex, sampleNumber, traceDimension(bounce, 0u)));
            if (hit.instance == noHit()) {
                radiance += throughput * environment.value.xyz();
                break;
            }
            SurfacePoint surface = surfaceAt(hit, origin, direction);
            vec3 toViewer = -direction;
            // Rays leave from just above the triangle, so they do not hit it again.
            vec3 above = surface.position
                       + surface.geometric * (1e-4f * (1.0f + maxComponent(abs(surface.position))));
            radiance += throughput
                      * (surface.material.emissive
                         + directLight(surface, toViewer, above, pixelIndex, sampleNumber, bounce));

            vec3 baseColor = surface.material.baseColor.xyz();
            float metallic = surface.material.metallic;
            float roughness = surface.material.roughness;
            float alpha = max(roughness * roughness, 0.001f);
            float chance = specularChance(baseColor, metallic, dot(surface.normal, toViewer));
            uint dimension = 2u + 4u * bounce;
            float u1 = randomNumber(pixelIndex, sampleNumber, dimension + 1u);
            float u2 = randomNumber(pixelIndex, sampleNumber, dimension + 2u);
            vec3 toLight = randomNumber(pixelIndex, sampleNumber, dimension) < chance
                             ? reflect(direction, ggxHalfVector(surface.normal, alpha, u1, u2))
                             : cosineDirection(surface.normal, u1, u2);
            float nDotL = dot(surface.normal, toLight);
            float density = mix(max(nDotL, 0.0f) / pi, ggxDensity(surface.normal, toViewer, toLight, alpha), chance);
            if (nDotL <= 0.0f or density <= 0.0f or dot(toLight, surface.geometric) <= 0.0f) {
                break;
            }
            throughput *= reflectedLight(surface.normal, toViewer, toLight, vec3(1.0f), baseColor, metallic, roughness)
                        / density;
            if (bounce >= 2u) {
                float survival = min(maxComponent(throughput), 0.95f);
                if (randomNumber(pixelIndex, sampleNumber, dimension + 3u) >= survival) {
                    break;
                }
                throughput = throughput / survival;
            }
            origin = above;
            direction = toLight;
        }
        return radiance;
    }

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint sampleNumber = sampleIndex.value;
        vec3 radiance = tracePath(pixel, uint(pixel.y * size.value.x + pixel.x), sampleNumber);
        // One broken sample would stay in the mean for good.
        if (any(isnan(radiance)) or any(isinf(radiance))) {
            radiance = vec3(0.0f);
        }
        vec4 mean = vec4(radiance, 1.0f);
        if (sampleNumber > 0u) {
            vec4 before = imageLoad(previous, pixel);
            mean = before + (mean - before) / float(sampleNumber + 1u);
        }
        imageStore(current, pixel, mean);
        imageStore(color, pixel, vec4(mean.xyz(), 1.0f));
    }
};

} // namespace lr::render::kernels
