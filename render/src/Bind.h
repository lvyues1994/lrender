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

template <class G, class P, tc::uint B>
void bind(tc::Device &device, tc::ImageBinding<G, tc::Dim::D2, P, B> &binding, tc::DeviceImage<G, tc::Dim::D2, P> &image) {
    binding.attach(image);
    device.bindImage(binding);
}

template <class T, tc::uint L>
void set(tc::Device &device, tc::Uniform<T, L> &uniform, T const &value) {
    uniform = value;
    device.setUniform(uniform);
}

inline tc::uint groupsFor(std::size_t const items, tc::uint const groupSize) {
    return static_cast<tc::uint>((items + groupSize - 1) / groupSize);
}

} // namespace lr::render
