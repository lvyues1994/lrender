#pragma once

// Attaches a resource to a kernel's binding and binds it; call after Device::useKernel().

#include <lr/render/FrameGraph.h>
#include <lr/render/GpuTypes.h>

namespace lr::render {

template <class T, tc::uint B, class Layout>
void bind(tc::Device &device, tc::TableBinding<T, B, Layout> &binding, tc::TableResource<T, Layout> &table) {
    binding.attach(table);
    device.bindTable(binding);
}

template <class T, tc::uint B>
void bind(tc::Device &device, tc::BufferBinding<T, B> &binding, tc::BufferResource<T> &buffer) {
    binding.attach(buffer);
    device.bindBuffer(binding);
}

template <class T, tc::uint B>
void bind(tc::Device &device, tc::BufferBinding<T, B> &binding, tc::DeviceBuffer<T> const &buffer) {
    binding.attach(buffer);
    device.bindBuffer(binding);
}

template <tc::uint B>
void bind(tc::Device &device, tc::AccelerationStructureBinding<B> &binding, tc::TopLevelAS const &structure) {
    binding.attach(structure);
    device.bindAccelerationStructure(binding);
}

template <class G, class P, tc::uint B>
void bind(tc::Device &device, tc::ImageBinding<G, tc::Dim::D2, P, B> &binding, tc::DeviceImage<G, tc::Dim::D2, P> &image) {
    binding.attach(image);
    device.bindImage(binding);
}

template <class G, tc::Dim D, class P, tc::uint B, tc::Filter F, tc::Wrap W, tc::Mip M>
void bind(tc::Device &device, tc::TextureBinding<G, D, P, B, F, W, M> &binding, tc::DeviceImage<G, D, P> const &image) {
    binding.attach(image);
    device.bindTexture(binding);
}

template <tc::uint B, tc::Filter F, tc::Wrap W, tc::Mip M>
void bind(tc::Device &device, tc::CompressedTextureBinding<B, F, W, M, tc::Dim::D2Array> &binding,
          tc::DeviceCompressedImage<tc::Dim::D2Array> const &image) {
    binding.attach(image);
    device.bindTexture(binding);
}

template <class T, tc::uint L>
void set(tc::Device &device, tc::Uniform<T, L> &uniform, T const &value) {
    uniform = value;
    device.setUniform(uniform);
}

// The pools of a kernel that expands LR_MATERIAL_TEXTURES.
template <class K>
void bindMaterialTextures(tc::Device &device, K &kernel, MaterialTextures const &textures) {
    auto const &color = textures.colorPools;
    auto const &compressed = textures.compressedPools;
    bind(device, kernel.colorPool0, *color[0]);
    bind(device, kernel.colorPool1, *color[1]);
    bind(device, kernel.colorPool2, *color[2]);
    bind(device, kernel.colorPool3, *color[3]);
    bind(device, kernel.colorPool4, *color[4]);
    bind(device, kernel.colorPool5, *color[5]);
    bind(device, kernel.colorPool6, *color[6]);
    bind(device, kernel.colorPool7, *color[7]);
    bind(device, kernel.compressedPool0, *compressed[0]);
    bind(device, kernel.compressedPool1, *compressed[1]);
    bind(device, kernel.compressedPool2, *compressed[2]);
    bind(device, kernel.compressedPool3, *compressed[3]);
}

// What the bindings and uniforms of LR_SURFACE_SHADING take.
struct SurfaceShadingResources {
    SceneTables const *scene = nullptr;
    tc::BufferResource<tc::uint> *lights = nullptr;   // LightGridOutputs::grid
    tc::BufferResource<tc::uvec4> *visible = nullptr; // VisibilityOutputs::visible, or its like
    tc::BufferResource<tc::uint> *shadow = nullptr;   // ShadowOutputs::map
    tc::mat4 inverseViewProjection{1.0f};
    tc::vec4 environment{0.0f}; // radiance in xyz
    tc::ivec2 size{0};
    tc::uint lightCount = 0;
    tc::uint shadowLight = 0xFFFFFFFFu; // ShadowOutputs::light
};

// The bindings and uniforms of a kernel that expands LR_SURFACE_SHADING.
template <class K>
void bindSurfaceShading(tc::Device &device, K &kernel, SurfaceShadingResources const &resources) {
    auto const &scene = *resources.scene;
    bindMaterialTextures(device, kernel, scene.textures);
    bind(device, kernel.environmentBrdf, *scene.textures.environmentBrdf);
    bind(device, kernel.instances, *scene.instances);
    bind(device, kernel.vertices, *scene.vertices);
    bind(device, kernel.indices, *scene.indices);
    bind(device, kernel.materials, *scene.materials);
    bind(device, kernel.lightGrid, *resources.lights);
    bind(device, kernel.visible, *resources.visible);
    bind(device, kernel.shadow, *resources.shadow);
    set(device, kernel.inverseViewProjection, resources.inverseViewProjection);
    set(device, kernel.environment, resources.environment);
    set(device, kernel.size, resources.size);
    set(device, kernel.lightCount, resources.lightCount);
    set(device, kernel.shadowLight, resources.shadowLight);
}

inline tc::uint groupsFor(std::size_t const items, tc::uint const groupSize) {
    return static_cast<tc::uint>((items + groupSize - 1) / groupSize);
}

} // namespace lr::render
