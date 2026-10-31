#pragma once

// Upkeep of the texture pools. Each dispatch works on one layer: llvmpipe's texelFetch
// reads layer 0 when the invocations of a dispatch fetch from different layers.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/Material.h>

namespace lr::render::kernels {

using namespace tc;

template <uint B>
using PoolTexels = TextureBinding<ColorPoolFormat, Dim::D2Array, ColorPoolPixel, B, Filter::Nearest,
                                  Wrap::ClampToEdge, Mip::Nearest>;

inline vec4 toLinear(vec4 texel, bool srgb) {
    if (srgb) {
        return vec4{srgbToLinear(texel.x), srgbToLinear(texel.y), srgbToLinear(texel.z), texel.w};
    }
    return texel;
}

inline vec4 fromLinear(vec4 texel, bool srgb) {
    if (srgb) {
        return vec4{linearToSrgb(texel.x), linearToSrgb(texel.y), linearToSrgb(texel.z), texel.w};
    }
    return texel;
}

// Level sourceLevel + 1 of a layer from sourceLevel: the mean of each 2x2 block, taken in
// linear colour, so an sRGB checkerboard of black and white turns into linear grey 0.5.
struct TC_KERNEL DownsampleTexture {
    static constexpr char fileLocation[] = "lr_downsample_texture";
    uvec3 local_size{8, 8, 1};

    ImageBinding<ColorPoolFormat, Dim::D2Array, ColorPoolPixel, 0> target;
    PoolTexels<1> source;
    Uniform<int, 0> sourceLevel{};
    Uniform<int, 1> layer{};
    Uniform<int, 2> size{}; // of the level written
    Uniform<uint, 3> srgb{};

    void main() {
        ivec2 texel = ivec2(gl_GlobalInvocationID.xy());
        if (texel.x >= size.value or texel.y >= size.value) {
            return;
        }
        bool encoded = srgb.value != 0u;
        vec4 sum = vec4(0.0f);
        for (int k = 0; k < 4; ++k) {
            ivec3 at = ivec3{2 * texel.x + (k & 1), 2 * texel.y + (k >> 1), layer.value};
            sum += toLinear(texelFetch(source, at, sourceLevel.value), encoded);
        }
        imageStore(target, ivec3{texel.x, texel.y, layer.value}, fromLinear(sum * 0.25f, encoded));
    }
};

// One level of one layer from a pool into the pool that replaces it.
struct TC_KERNEL CopyTextureLayer {
    static constexpr char fileLocation[] = "lr_copy_texture_layer";
    uvec3 local_size{8, 8, 1};

    ImageBinding<ColorPoolFormat, Dim::D2Array, ColorPoolPixel, 0> target;
    PoolTexels<1> source;
    Uniform<int, 0> level{};
    Uniform<int, 1> layer{};
    Uniform<int, 2> size{}; // of the level

    void main() {
        ivec2 texel = ivec2(gl_GlobalInvocationID.xy());
        if (texel.x < size.value and texel.y < size.value) {
            ivec3 at = ivec3{texel.x, texel.y, layer.value};
            imageStore(target, at, texelFetch(source, at, level.value));
        }
    }
};

} // namespace lr::render::kernels
