#pragma once

// Material models of the tests' own, added the way an application adds one: a model type
// and its kernel instance in a header the test target transpiles.

#include <lr/render/kernels/ShadeMaterialModel.h>

namespace lr::test {

// The colour in the material's model parameters, unlit.
struct FlatColour {
    static render::kernels::ShadedSurface evaluate(render::kernels::ShadingInputs inputs) {
        return render::kernels::emissionOnly(inputs.material.parameters.xyz());
    }
};

// Metallic-roughness with the base colour from the model parameters, lit as the built-in
// model is.
struct PaintedMetallicRoughness {
    static render::kernels::ShadedSurface evaluate(render::kernels::ShadingInputs inputs) {
        render::kernels::ShadedSurface surface = render::kernels::MetallicRoughness::evaluate(inputs);
        surface.baseColor = inputs.material.parameters.xyz();
        return surface;
    }
};

} // namespace lr::test

namespace lr::render::kernels {

TC_KERNEL_INSTANCE("lr_test_flat_colour", ShadeMaterialModel<test::FlatColour>);
TC_KERNEL_INSTANCE("lr_test_painted_metallic_roughness", ShadeMaterialModel<test::PaintedMetallicRoughness>);

} // namespace lr::render::kernels
