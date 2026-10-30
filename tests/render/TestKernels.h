#pragma once

// Kernels of the tests' own, used by features of the tests' own: the way applications
// extend a pipeline.

#include <lr/render/GpuTypes.h>

namespace lr::test {

using namespace tc;

struct TC_KERNEL FillHdr {
    static constexpr char fileLocation[] = "lr_test_fill_hdr";
    uvec3 local_size{8, 8, 1};

    ImageBinding<render::HdrFormat, Dim::D2, render::HdrPixel, 0> color;
    Uniform<vec4, 0> fill{};
    Uniform<ivec2, 1> size{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x < size.value.x and pixel.y < size.value.y) {
            imageStore(color, pixel, fill.value);
        }
    }
};

// Copies the HDR colour into a buffer, rows from the top.
struct TC_KERNEL ReadHdr {
    static constexpr char fileLocation[] = "lr_test_read_hdr";
    uvec3 local_size{8, 8, 1};

    ImageBinding<render::HdrFormat, Dim::D2, render::HdrPixel, 0> color;
    BufferBinding<vec4, 1> pixels;
    Uniform<ivec2, 0> size{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x < size.value.x and pixel.y < size.value.y) {
            pixels[uint(pixel.y * size.value.x + pixel.x)] = imageLoad(color, pixel);
        }
    }
};

} // namespace lr::test
