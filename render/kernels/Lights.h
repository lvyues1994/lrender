#pragma once

// The light grid shading reads (see lightGridHeaderWords()).

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/Lighting.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

// How much wider than they are the grid's cells are taken, so that rounding where shading
// finds a point's cell and its distance from a light cannot leave out a light that reaches
// it: pixels around each tile, a fraction of the depth before and after each slice, and of
// a light's range past it.
inline float lightTileMargin() { return 1.0f; }
inline float lightSliceMargin() { return 1.0e-3f; }
inline float lightRangeMargin() { return 1.0e-3f; }

inline float lightSliceStart(uint slice, vec2 depths) { return depths.x * exp(float(slice) / depths.y); }

// Fills a view's light grid, an invocation per word of the cells' masks (and at least one
// per light, which copies it); those next to each other test the same lights. A cell holds every light that has no range, and every point
// light whose sphere of its range reaches the cell, widened by the margins above: past
// each side plane of its tile, and between the planes of its slice. With `shape.w` 0 the
// grid is one cell that holds every light.
struct TC_KERNEL AssignLights {
    static constexpr char fileLocation[] = "lr_assign_lights";
    uvec3 local_size{64, 1, 1};

    LightTable<0> lights;
    BufferBinding<uint, 1> grid;
    Uniform<mat4, 0> viewProjection{};
    Uniform<ivec2, 1> size{};
    Uniform<uvec4, 2> shape{}; // tiles across, tiles down, slices, whether cells hold their lights
    Uniform<vec2, 3> depths{}; // the near plane, slices per unit of log(depth / near plane)
    Uniform<uint, 4> lightCount{};

    // Whether the sphere reaches the side of `plane` (inside: dot(plane, (p, 1)) >= 0).
    static bool reachesPlane(vec4 plane, vec3 centre, float radius) {
        return dot(plane.xyz(), centre) + plane.w >= -radius * length(plane.xyz());
    }

    void writeHeader(vec4 row0, vec4 row1, vec4 row3) {
        grid[0] = shape.value.x;
        grid[1] = shape.value.y;
        grid[2] = shape.value.z;
        grid[4] = floatBitsToUint(depths.value.x);
        grid[5] = floatBitsToUint(depths.value.y);
        std::array<vec4, 3> rows{row0, row1, row3};
        for (uint row = 0u; row < 3u; ++row) {
            for (uint k = 0u; k < 4u; ++k) {
                grid[8u + 4u * row + k] = floatBitsToUint(rows[row][k]);
            }
        }
    }

    void copyLight(uint index) {
        GpuLight light = lights[index];
        uint base = lightGridHeaderWords() + lightRecordWords() * index;
        grid[base] = light.kind;
        for (uint k = 0u; k < 3u; ++k) {
            grid[base + 1u + k] = floatBitsToUint(light.radiance[k]);
            grid[base + 4u + k] = floatBitsToUint(light.vector[k]);
        }
        grid[base + 7u] = floatBitsToUint(light.range);
    }

    void main() {
        uint index = gl_GlobalInvocationID.x;
        mat4 m = viewProjection.value;
        vec4 row0 = vec4{m[0].x, m[1].x, m[2].x, m[3].x};
        vec4 row1 = vec4{m[0].y, m[1].y, m[2].y, m[3].y};
        vec4 row3 = vec4{m[0].w, m[1].w, m[2].w, m[3].w};
        if (index == 0u) {
            writeHeader(row0, row1, row3);
        }
        uint count = lightCount.value;
        if (index < count) {
            copyLight(index);
        }
        uint words = lightMaskWords(count);
        uint tilesAcross = shape.value.x;
        uint tilesDown = shape.value.y;
        uint slices = shape.value.z;
        uint cells = tilesAcross * tilesDown * slices;
        if (index >= cells * words) {
            return;
        }
        uint cell = index % cells;
        uint word = index / cells;
        uint tileX = cell % tilesAcross;
        uint tileY = (cell / tilesAcross) % tilesDown;
        uint slice = cell / (tilesAcross * tilesDown);

        vec2 extent = vec2(size.value);
        float tile = float(lightTileSize());
        float leftX = (float(tileX) * tile - lightTileMargin()) / extent.x * 2.0f - 1.0f;
        float rightX = (float(tileX + 1u) * tile + lightTileMargin()) / extent.x * 2.0f - 1.0f;
        float topY = 1.0f - (float(tileY) * tile - lightTileMargin()) / extent.y * 2.0f;
        float bottomY = 1.0f - (float(tileY + 1u) * tile + lightTileMargin()) / extent.y * 2.0f;
        vec4 left = row0 - row3 * leftX;
        vec4 right = row3 * rightX - row0;
        vec4 bottom = row1 - row3 * bottomY;
        vec4 top = row3 * topY - row1;
        float nearest = slice == 0u ? -infiniteDistance()
                                    : lightSliceStart(slice, depths.value) * (1.0f - lightSliceMargin());
        float farthest = slice + 1u == slices ? infiniteDistance()
                                              : lightSliceStart(slice + 1u, depths.value) * (1.0f + lightSliceMargin());
        float depthScale = length(row3.xyz());

        uint bits = 0u;
        for (uint bit = 0u; bit < 32u; ++bit) {
            uint light = 32u * word + bit;
            if (light >= count) {
                break;
            }
            GpuLight record = lights[light];
            bool held = shape.value.w == 0u or record.kind != lightPoint() or record.range <= 0.0f;
            if (not held) {
                vec3 centre = record.vector;
                float radius = record.range * (1.0f + lightRangeMargin());
                float depth = dot(row3.xyz(), centre) + row3.w;
                held = depth + radius * depthScale >= nearest and depth - radius * depthScale <= farthest
                   and reachesPlane(left, centre, radius) and reachesPlane(right, centre, radius)
                   and reachesPlane(bottom, centre, radius) and reachesPlane(top, centre, radius);
            }
            if (held) {
                bits |= 1u << bit;
            }
        }
        grid[lightGridHeaderWords() + lightRecordWords() * count + cell * words + word] = bits;
    }
};

} // namespace lr::render::kernels
