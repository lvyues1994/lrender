#pragma once

// What the built-in features' passes share.

#include "kernels/Visibility.h"

#include <lr/render/Bind.h>
#include <lr/render/FrameGraph.h>

#include <cstddef>

namespace lr::render {

inline std::size_t pixelsOf(core::ivec2 const extent) {
    return static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y);
}

// Workgroups of 8x8 invocations over an image.
inline tc::uvec3 groupsCovering(core::ivec2 const extent) {
    return tc::uvec3{groupsFor(static_cast<std::size_t>(extent.x), 8), groupsFor(static_cast<std::size_t>(extent.y), 8),
                     1};
}

// Sets words [first, first + count) of `target` to `value`.
inline void clearWords(PassContext &context, BufferRef<tc::uint> const target, std::size_t const first,
                       std::size_t const count, tc::uint const value) {
    auto &device = context.device();
    auto kernel = kernels::ClearUint{};
    device.useKernel(kernel);
    bind(device, kernel.target, context.buffer(target));
    set(device, kernel.fill, value);
    set(device, kernel.first, static_cast<tc::uint>(first));
    set(device, kernel.count, static_cast<tc::uint>(count));
    context.dispatch(kernel, tc::uvec3{groupsFor(count, 64), 1, 1});
}

} // namespace lr::render
