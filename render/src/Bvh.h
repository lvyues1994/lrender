#pragma once

#include "RenderWork.h"

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
    // For buildBvh() over a ParallelFor: nodes of more primitives are split with their
    // work spread over it, and the subtrees below them built each as one of its items.
    std::uint32_t parallelGrain = 16384;
};

// Binned surface area heuristic, leaves no deeper than bvhMaxDepth() below the root. No
// primitives give no nodes. Throws std::length_error for more primitives than a
// hierarchy of that depth can hold.
Bvh buildBvh(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config);
// By the same heuristic, with the nodes above the subtrees of config.parallelGrain
// primitives laid out first; the result does not depend on which ParallelFor runs it. Must
// not run as an item of `parallel`.
Bvh buildBvh(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config, ParallelFor &parallel);

core::Aabb boundsOf(GpuBvhNode const &node) noexcept;

} // namespace lr::render
