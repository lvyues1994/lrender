#pragma once

// Material models on the host: what the renderer holds of each, and how an application
// adds its own (see kernels/ShadeMaterialModel.h for the kernel side).
//
//     struct FlatColour {   // in a header the application transpiles with tc_add_kernels
//         static lr::render::kernels::ShadedSurface evaluate(lr::render::kernels::ShadingInputs inputs) {
//             return lr::render::kernels::emissionOnly(inputs.material.parameters.xyz());
//         }
//     };
//     namespace lr::render::kernels {
//     TC_KERNEL_INSTANCE("app_flat_colour", ShadeMaterialModel<FlatColour>);
//     }
//
//     auto const flat = renderer->materialModels().add(lr::render::makeMaterialModel<FlatColour>());
//     material.model = flat;
//     material.modelParameters = lr::core::vec4{0.9f, 0.2f, 0.1f, 0.0f};

#include <lr/core/Id.h>
#include <lr/render/Bind.h>
#include <lr/render/FrameGraph.h>
#include <lr/render/kernels/ShadeMaterialModel.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace lr::render {

// What the kernel of one material model shades in one view's frame.
struct ModelShadingResources {
    SurfaceShadingResources surface;
    GraphImage<HdrFormat> *color = nullptr;
    tc::BufferResource<tc::uvec2> *pixels = nullptr; // the models' pixel lists, see modelListEntry()
    BufferRef<tc::DispatchIndirectCommand> arguments; // per model, workgroups of 64 pixels
    core::MaterialModelId model;
};

// On the GPU thread, while a MaterialModel shades: binds everything an instance of
// ShadeMaterialModel binds and dispatches it over the model's pixels.
struct ModelShading {
    virtual ~ModelShading() = default;

    template <class Model>
    void dispatch(kernels::ShadeMaterialModel<Model> &kernel) {
        auto &context = pass();
        auto &device = context.device();
        auto const &frame = resources();
        device.useKernel(kernel);
        bind(device, kernel.color, *frame.color);
        bindSurfaceShading(device, kernel, frame.surface);
        bind(device, kernel.pixels, *frame.pixels);
        set(device, kernel.model, tc::uint{frame.model.value});
        context.dispatchIndirect(kernel, frame.arguments, frame.model.value);
    }

protected:
    virtual PassContext &pass() noexcept = 0;
    virtual ModelShadingResources const &resources() const noexcept = 0;
};

// A material model as the renderer holds it.
struct MaterialModel {
    virtual ~MaterialModel() = default;

    // Its shading kernel's program, which the renderer checks against its device.
    virtual std::string_view program() const noexcept = 0;
    // On the GPU thread: shades the model's pixels, through shading.dispatch().
    virtual void shade(ModelShading &shading) = 0;
};

// The model ShadeMaterialModel<Model> shades; the application's kernel header names the
// instance with TC_KERNEL_INSTANCE.
template <class Model>
struct KernelMaterialModel final : MaterialModel {
    std::string_view program() const noexcept override { return kernels::ShadeMaterialModel<Model>::fileLocation; }

    void shade(ModelShading &shading) override {
        auto kernel = kernels::ShadeMaterialModel<Model>{};
        shading.dispatch(kernel);
    }
};

template <class Model>
std::shared_ptr<MaterialModel> makeMaterialModel() {
    return std::make_shared<KernelMaterialModel<Model>>();
}

// A renderer's material models, by MaterialModelId. Thread-safe.
struct MaterialModels {
    virtual ~MaterialModels() = default;

    // Model 0, glTF's metallic-roughness, which materials name unless they say otherwise.
    static constexpr core::MaterialModelId metallicRoughness() noexcept { return core::MaterialModelId{0}; }

    // Adds a model for the frames submitted from now on. Throws tc::DeviceError naming the
    // model's program when the device cannot run it, and std::length_error once
    // maxMaterialModels() are taken. Waits for the GPU thread, so it must not run there.
    virtual core::MaterialModelId add(std::shared_ptr<MaterialModel> model) = 0;
    virtual std::uint32_t count() const = 0;
};

} // namespace lr::render
