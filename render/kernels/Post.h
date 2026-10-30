#pragma once

#include <lr/render/GpuTypes.h>

namespace lr::render::kernels {

using namespace tc;

inline float encodeSrgb(float linear) {
    float value = clamp(linear, 0.0f, 1.0f);
    if (value <= 0.0031308f) {
        return value * 12.92f;
    }
    return 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

// Narkowicz's fit of the ACES filmic curve.
inline vec3 acesFilm(vec3 x) {
    return clamp((x * (x * 2.51f + 0.03f)) / (x * (x * 2.43f + 0.59f) + 0.14f), vec3(0.0f), vec3(1.0f));
}

// HDR radiance to sRGB-encoded RGBA8, packed with the red channel in the low byte.
struct TC_KERNEL Tonemap {
    static constexpr char fileLocation[] = "lr_tonemap";
    static constexpr Precision floatPrecision = Precision::Medium;
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    BufferBinding<uint, 1> output;
    Uniform<ivec2, 0> size{};
    Uniform<float, 1> exposure{};
    Uniform<uint, 2> filmic{}; // 0: clamp, 1: ACES

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        vec3 radiance = imageLoad(color, pixel).xyz() * exposure.value;
        vec3 mapped = filmic.value == 1u ? acesFilm(radiance) : clamp(radiance, vec3(0.0f), vec3(1.0f));
        vec4 encoded = vec4{encodeSrgb(mapped.x), encodeSrgb(mapped.y), encodeSrgb(mapped.z), 1.0f};
        output[uint(pixel.y * size.value.x + pixel.x)] = packUnorm4x8(encoded);
    }
};

} // namespace lr::render::kernels
