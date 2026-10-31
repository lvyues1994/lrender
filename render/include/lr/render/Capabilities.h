#pragma once

#include <cstdint>

namespace lr::render {

// Whether the renderer may use an optional kernel variant: wherever its device runs it
// (Auto), or never (Off), for instance to compare the variants on one device.
enum class VariantUse : std::uint8_t {
    Auto,
    Off,
};

// The kernel variants a renderer chose for its device when it created it.
struct RendererCapabilities {
    // Compaction and material classification scan with subgroup operations; otherwise
    // with shared memory alone.
    bool subgroups = false;
    // The visibility buffer is rasterized once, keeping the nearest (depth, triangle) per
    // pixel with a 64-bit atomicMin; otherwise twice, depth first, then the triangle.
    bool singlePassVisibility = false;
    // Invocations per subgroup on the device; 0 without subgroups.
    std::uint32_t subgroupSize = 0;
};

} // namespace lr::render
