#pragma once

// The shading kernels: one instance of ShadeMaterialModel per material model, run over the
// pixels the material classification gave the model. A model is a plain struct with
//
//     static ShadedSurface evaluate(ShadingInputs inputs);
//
// (SurfaceShading.h). An application adds one by naming its instance in a header it
// transpiles with tc_add_kernels, in this namespace, after including this header:
//
//     namespace lr::render::kernels {
//     TC_KERNEL_INSTANCE("app_flat_colour", ShadeMaterialModel<app::FlatColour>);
//     }
//
// and registering lr::render::makeMaterialModel<app::FlatColour>() (MaterialModel.h)
// with the renderer, which gives the model the id its materials name.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/SurfaceShading.h>

namespace lr::render::kernels {

template <class Model>
struct TC_KERNEL ShadeMaterialModel {
    static constexpr char const *fileLocation = tc::instanceName<ShadeMaterialModel>();
    uvec3 local_size{64, 1, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    LR_SURFACE_SHADING(1u)
    BufferBinding<uvec2, 21> pixels; // the models' pixel lists, see modelListEntry()
    Uniform<uint, 5> model{};

    void main() {
        uvec2 list = pixels[model.value];
        uint i = gl_GlobalInvocationID.x;
        if (i >= list.x) {
            return;
        }
        uvec2 entry = pixels[modelListEntry(list.y, i)];
        uint width = uint(size.value.x);
        ivec2 pixel = ivec2{int(entry.x % width), int(entry.x / width)};
        ShadingInputs inputs = surfaceAt(entry.y, raysThrough(pixel));
        imageStore(color, pixel, vec4(lightSurface(inputs, Model::evaluate(inputs)), 1.0f));
    }
};

} // namespace lr::render::kernels
