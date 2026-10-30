#pragma once

// A kernel of the tests' own, used by a feature of the tests' own: the way applications
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

} // namespace lr::test
