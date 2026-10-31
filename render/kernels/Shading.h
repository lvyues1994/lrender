#pragma once

// Shading the visibility buffer by material model. Classification sorts the covered pixels
// into one list per model (sky pixels get the environment there and then); each model's
// instance of ShadeMaterialModel then runs over its own list, as many workgroups as the
// list needs. Transparent fragments are shaded with the built-in model alone, sorted by
// depth and blended front to back over the opaque colour.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/ShadeMaterialModel.h>
#include <lr/render/kernels/SurfaceShading.h>
#include <lr/render/kernels/VisibleInstances.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

TC_KERNEL_INSTANCE("lr_shade_metallic_roughness", ShadeMaterialModel<MetallicRoughness>);

inline constexpr uint maxTransparentLayers() { return 8u; }

// A covered pixel's place: its material model in the top bits, its place in the model's
// list below.
inline uint modelPlace(uint model, uint place) { return model << 28u | place; }
inline uint placeModel(uint packed) { return packed >> 28u; }
inline uint placeInList(uint packed) { return packed & 0x0FFFFFFFu; }

inline bool nearerFragment(uvec2 a, uvec2 b) { return a.x < b.x or (a.x == b.x and a.y < b.y); }

} // namespace lr::render::kernels

// The classification's bindings and the model of a pixel. Pixels no triangle covers get
// the environment's radiance and model `modelCount`, which is none.
#define LR_PIXEL_CLASSIFICATION                                                                                        \
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;                                                               \
    InstanceTable<1> instances;                                                                                        \
    MaterialTable<2> materials;                                                                                        \
    BufferBinding<uvec4, 3> visible;                                                                                   \
    BufferBinding<uint, 4> triangles;                                                                                  \
    BufferBinding<uint, 5> counts; /* pixels per model, cleared before */                                             \
    BufferBinding<uint, 6> places; /* per covered pixel, see modelPlace() */                                          \
    Uniform<ivec2, 0> size{};                                                                                          \
    Uniform<vec4, 1> environment{}; /* radiance in xyz */                                                              \
    Uniform<uint, 2> modelCount{};                                                                                     \
    Shared<std::array<uint, maxMaterialModels()>> groupCounts;                                                         \
    Shared<std::array<uint, maxMaterialModels()>> groupBases;                                                          \
                                                                                                                       \
    LR_FIND_VISIBLE_SLOT                                                                                               \
                                                                                                                       \
    uint classify(ivec2 pixel, uint pixelIndex, bool inside) {                                                         \
        if (not inside) {                                                                                              \
            return modelCount.value;                                                                                   \
        }                                                                                                              \
        uint triangle = triangles[pixelIndex];                                                                         \
        if (triangle == noTriangle()) {                                                                                \
            imageStore(color, pixel, vec4(environment.value.xyz(), 1.0f));                                             \
            return modelCount.value;                                                                                   \
        }                                                                                                              \
        uint model = materials[instances[visible[1u + findSlot(triangle)].x].material].model;                         \
        return model < modelCount.value ? model : 0u;                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* Once the workgroup has counted its pixels of each model: the range of each model's                             \
       list the workgroup takes. */                                                                                    \
    void takeGroupRanges() {                                                                                           \
        uint local = gl_LocalInvocationIndex;                                                                          \
        if (local < modelCount.value and groupCounts[local] > 0u) {                                                    \
            groupBases[local] = atomicAdd(counts[local], groupCounts[local]);                                          \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    void writePlace(uint pixelIndex, uint model, uint rank) {                                                          \
        if (model < modelCount.value) {                                                                                \
            places[pixelIndex] = modelPlace(model, groupBases[model] + rank);                                          \
        }                                                                                                              \
    }

namespace lr::render::kernels {

// Ranks the pixels of each model within the workgroup with shared atomics.
struct TC_KERNEL ClassifyPixels {
    static constexpr char fileLocation[] = "lr_classify_pixels";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{8, 8, 1};

    LR_PIXEL_CLASSIFICATION

    void main() {
        uint local = gl_LocalInvocationIndex;
        if (local < maxMaterialModels()) {
            groupCounts[local] = 0u;
        }
        barrier();
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        bool inside = pixel.x < size.value.x and pixel.y < size.value.y;
        uint pixelIndex = inside ? uint(pixel.y * size.value.x + pixel.x) : 0u;
        uint model = classify(pixel, pixelIndex, inside);
        uint rank = 0u;
        if (model < modelCount.value) {
            rank = atomicAdd(groupCounts[model], 1u);
        }
        barrier();
        takeGroupRanges();
        barrier();
        writePlace(pixelIndex, model, rank);
    }
};

// Ranks the pixels of each model within each subgroup by a scan, and takes one place per
// subgroup and model from the workgroup's count.
struct TC_KERNEL ClassifyPixelsSubgroups {
    static constexpr char fileLocation[] = "lr_classify_pixels_subgroups";
    static constexpr bool usesBarrier = true;
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{8, 8, 1};

    LR_PIXEL_CLASSIFICATION

    void main() {
        uint local = gl_LocalInvocationIndex;
        if (local < maxMaterialModels()) {
            groupCounts[local] = 0u;
        }
        barrier();
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        bool inside = pixel.x < size.value.x and pixel.y < size.value.y;
        uint pixelIndex = inside ? uint(pixel.y * size.value.x + pixel.x) : 0u;
        uint model = classify(pixel, pixelIndex, inside);
        uint rank = 0u;
        for (uint candidate = 0u; candidate < modelCount.value; ++candidate) {
            uint mine = model == candidate ? 1u : 0u;
            uint total = subgroupAdd(mine);
            if (total > 0u) {
                uint before = subgroupExclusiveAdd(mine);
                uint base = 0u;
                if (subgroupElect()) {
                    base = atomicAdd(groupCounts[candidate], total);
                }
                base = subgroupBroadcastFirst(base);
                if (mine == 1u) {
                    rank = base + before;
                }
            }
        }
        barrier();
        takeGroupRanges();
        barrier();
        writePlace(pixelIndex, model, rank);
    }
};

// One workgroup: where each model's list starts (after those of the models before it), its
// length, and its shading's workgroup count.
struct TC_KERNEL AllocateModelLists {
    static constexpr char fileLocation[] = "lr_allocate_model_lists";
    uvec3 local_size{16, 1, 1}; // maxMaterialModels()

    BufferBinding<uint, 0> counts;
    BufferBinding<uvec2, 1> pixels; // see modelListEntry()
    BufferBinding<DispatchIndirectCommand, 2> arguments;
    Uniform<uint, 0> modelCount{};

    void main() {
        uint model = gl_LocalInvocationIndex;
        if (model >= maxMaterialModels()) {
            return;
        }
        uint start = 0u;
        for (uint before = 0u; before < model and before < modelCount.value; ++before) {
            start += counts[before];
        }
        uint count = model < modelCount.value ? counts[model] : 0u;
        pixels[model] = uvec2{count, start};
        arguments[model].x = (count + 63u) / 64u;
        arguments[model].y = 1u;
        arguments[model].z = 1u;
    }
};

// Writes each covered pixel, and its triangle, into its place in its model's list.
struct TC_KERNEL ScatterModelPixels {
    static constexpr char fileLocation[] = "lr_scatter_model_pixels";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> triangles;
    BufferBinding<uint, 1> places;
    BufferBinding<uvec2, 2> pixels; // see modelListEntry()
    Uniform<uint, 0> count{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= count.value) {
            return;
        }
        uint triangle = triangles[i];
        if (triangle == noTriangle()) {
            return;
        }
        uint place = places[i];
        uint start = pixels[placeModel(place)].y;
        pixels[modelListEntry(start, placeInList(place))] = uvec2{i, triangle};
    }
};

// Each pixel sorts the fragments of its list by depth, keeps the nearest
// maxTransparentLayers(), shades them with the metallic-roughness model and blends them
// front to back over the opaque colour, `behind`.
struct TC_KERNEL BlendTransparent {
    static constexpr char fileLocation[] = "lr_blend_transparent";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 1> behind;
    LR_SURFACE_SHADING(2u)
    // The head of each pixel's fragment list, the fragment count, then the fragments (see
    // fragmentWords()).
    BufferBinding<uint, 22> fragments;

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint pixelIndex = uint(pixel.y * size.value.x + pixel.x);
        uint firstFragment = fragmentWords(uint(size.value.x * size.value.y));
        std::array<uvec2, maxTransparentLayers()> nearest;
        uint kept = 0u;
        uint fragment = fragments[pixelIndex];
        while (fragment != noFragment()) {
            uint word = firstFragment + 3u * fragment;
            uvec2 key = uvec2{fragments[word], fragments[word + 2u]};
            if (kept < maxTransparentLayers() or nearerFragment(key, nearest[maxTransparentLayers() - 1u])) {
                uint at = kept < maxTransparentLayers() ? kept : maxTransparentLayers() - 1u;
                while (at > 0u and nearerFragment(key, nearest[at - 1u])) {
                    nearest[at] = nearest[at - 1u];
                    at -= 1u;
                }
                nearest[at] = key;
                kept = min(kept + 1u, maxTransparentLayers());
            }
            fragment = fragments[word + 1u];
        }
        vec3 opaque = imageLoad(behind, pixel).xyz();
        vec3 radiance = opaque;
        if (kept > 0u) {
            PixelRays rays = raysThrough(pixel);
            vec3 sum = vec3(0.0f);
            float transmittance = 1.0f;
            for (uint k = 0u; k < kept; ++k) {
                ShadingInputs inputs = surfaceAt(nearest[k].y, rays);
                ShadedSurface surface = MetallicRoughness::evaluate(inputs);
                vec3 shaded = lightSurface(inputs, surface);
                float alpha = clamp(surface.coverage, 0.0f, 1.0f);
                sum += shaded * (alpha * transmittance);
                transmittance *= 1.0f - alpha;
            }
            radiance = sum + opaque * transmittance;
        }
        imageStore(color, pixel, vec4(radiance, 1.0f));
    }
};

// Copies one HDR image into another of the same size.
struct TC_KERNEL CopyHdr {
    static constexpr char fileLocation[] = "lr_copy_hdr";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> source;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 1> target;
    Uniform<ivec2, 0> size{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x < size.value.x and pixel.y < size.value.y) {
            imageStore(target, pixel, imageLoad(source, pixel));
        }
    }
};

} // namespace lr::render::kernels
