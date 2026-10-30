#pragma once

#include <lr/core/Math.h>
#include <lr/render/GpuTypes.h>

#include <cstdint>
#include <vector>

namespace lr::render {

// A hierarchy over primitives given by their bounds. Nodes are as the kernels read them
// (GpuBvhNode), the root first and siblings next to each other; a leaf's `first` indexes
// `order`, which lists the primitives leaf by leaf.
struct Bvh {
    std::vector<GpuBvhNode> nodes;
    std::vector<std::uint32_t> order;
};

struct BvhBuildConfig {
    std::uint32_t maxLeafSize = 4;
};

// Binned surface area heuristic, leaves no deeper than bvhMaxDepth() below the root. No
// primitives give no nodes. Throws std::length_error for more primitives than a
// hierarchy of that depth can hold.
Bvh buildBvh(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config);

core::Aabb boundsOf(GpuBvhNode const &node) noexcept;

} // namespace lr::render
