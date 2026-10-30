#include "Bvh.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace lr::render {

namespace {

constexpr auto kBins = std::uint32_t{16};
constexpr auto kInfinity = std::numeric_limits<float>::infinity();

core::Aabb emptyBox() noexcept { return core::Aabb{core::vec3{kInfinity}, core::vec3{-kInfinity}}; }

core::Aabb merged(core::Aabb const &a, core::Aabb const &b) noexcept {
    return core::Aabb{tc::min(a.min, b.min), tc::max(a.max, b.max)};
}

core::Aabb merged(core::Aabb const &box, core::vec3 const &point) noexcept {
    return core::Aabb{tc::min(box.min, point), tc::max(box.max, point)};
}

// Half the surface area: what the heuristic compares.
float halfArea(core::Aabb const &box) noexcept {
    auto const size = tc::max(box.max - box.min, core::vec3{0.0f});
    return size.x * size.y + size.y * size.z + size.z * size.x;
}

std::uint32_t ceilLog2(std::size_t const value) noexcept {
    auto bits = std::uint32_t{0};
    while ((std::size_t{1} << bits) < value) {
        ++bits;
    }
    return bits;
}

struct Task {
    std::uint32_t node;
    std::uint32_t begin;
    std::uint32_t end;
    std::uint32_t depth;
};

struct Split {
    int axis = -1; // none found
    std::uint32_t bin = 0; // the first bin on the right
    float cost = kInfinity;
};

struct Builder {
    Builder(std::vector<core::Aabb> const &bounds_, BvhBuildConfig const &config_)
        : bounds(bounds_), config(config_) {
        centres.reserve(bounds.size());
        for (auto const &box : bounds) {
            centres.push_back((box.min + box.max) * 0.5f);
        }
    }

    Bvh run() && {
        auto const count = static_cast<std::uint32_t>(bounds.size());
        result.order.resize(count);
        std::iota(result.order.begin(), result.order.end(), std::uint32_t{0});
        result.nodes.reserve(2 * std::size_t{count} - 1);
        result.nodes.push_back(GpuBvhNode{});
        auto tasks = std::vector<Task>{Task{0, 0, count, 0}};
        while (not tasks.empty()) {
            auto const task = tasks.back();
            tasks.pop_back();
            build(task, tasks);
        }
        return std::move(result);
    }

private:
    void build(Task const &task, std::vector<Task> &tasks) {
        auto box = emptyBox();
        auto centreBox = emptyBox();
        for (auto index = task.begin; index < task.end; ++index) {
            auto const primitive = result.order[index];
            box = merged(box, bounds[primitive]);
            centreBox = merged(centreBox, centres[primitive]);
        }
        auto const middle = splitPoint(task, box, centreBox);
        if (not middle) {
            result.nodes[task.node] = GpuBvhNode{box.min, task.begin, box.max, task.end - task.begin};
            return;
        }
        auto const left = static_cast<std::uint32_t>(result.nodes.size());
        result.nodes.resize(result.nodes.size() + 2);
        result.nodes[task.node] = GpuBvhNode{box.min, left, box.max, 0};
        tasks.push_back(Task{left + 1, *middle, task.end, task.depth + 1});
        tasks.push_back(Task{left, task.begin, *middle, task.depth + 1});
    }

    // Where the primitives of `task` split, after reordering them; nullopt for a leaf.
    std::optional<std::uint32_t> splitPoint(Task const &task, core::Aabb const &box, core::Aabb const &centreBox) {
        auto const count = task.end - task.begin;
        auto const fitsLeaf = count <= config.maxLeafSize;
        if (count == 1) {
            return std::nullopt;
        }
        // Halving from here on still ends within the depth limit.
        if (task.depth + ceilLog2(count) >= bvhMaxDepth()) {
            return fitsLeaf ? std::nullopt : std::optional<std::uint32_t>{medianSplit(task, centreBox)};
        }
        auto const area = halfArea(box);
        auto const best = area > 0.0f ? bestSplit(task, centreBox) : Split{};
        if (best.axis < 0) {
            return fitsLeaf ? std::nullopt : std::optional<std::uint32_t>{medianSplit(task, centreBox)};
        }
        // Costs in primitive tests, with visiting a node as dear as testing a primitive.
        auto const splitCost = 1.0f + best.cost / area;
        if (fitsLeaf and static_cast<float>(count) <= splitCost) {
            return std::nullopt;
        }
        auto const first = result.order.begin() + task.begin;
        auto const middle = std::partition(first, result.order.begin() + task.end, [&](std::uint32_t const primitive) {
            return binOf(centres[primitive], best.axis, centreBox) < best.bin;
        });
        return task.begin + static_cast<std::uint32_t>(middle - first);
    }

    // Only for axes along which the centres spread.
    static std::uint32_t binOf(core::vec3 const &centre, int const axis, core::Aabb const &centreBox) noexcept {
        auto const low = centreBox.min[axis];
        auto const ratio = (centre[axis] - low) / (centreBox.max[axis] - low);
        return static_cast<std::uint32_t>(std::min(ratio * static_cast<float>(kBins), static_cast<float>(kBins - 1)));
    }

    Split bestSplit(Task const &task, core::Aabb const &centreBox) const {
        auto best = Split{};
        for (auto axis = 0; axis < 3; ++axis) {
            if (not(centreBox.max[axis] > centreBox.min[axis])) {
                continue;
            }
            auto boxes = std::array<core::Aabb, kBins>{};
            boxes.fill(emptyBox());
            auto counts = std::array<std::uint32_t, kBins>{};
            for (auto index = task.begin; index < task.end; ++index) {
                auto const primitive = result.order[index];
                auto const bin = binOf(centres[primitive], axis, centreBox);
                boxes[bin] = merged(boxes[bin], bounds[primitive]);
                counts[bin] += 1;
            }
            // rightCosts[b]: the cost of bins b and above.
            auto rightCosts = std::array<float, kBins>{};
            auto right = emptyBox();
            auto rightCount = std::uint32_t{0};
            for (auto bin = kBins - 1; bin > 0; --bin) {
                right = merged(right, boxes[bin]);
                rightCount += counts[bin];
                rightCosts[bin] = rightCount > 0 ? halfArea(right) * static_cast<float>(rightCount) : kInfinity;
            }
            auto left = emptyBox();
            auto leftCount = std::uint32_t{0};
            for (auto bin = std::uint32_t{1}; bin < kBins; ++bin) {
                left = merged(left, boxes[bin - 1]);
                leftCount += counts[bin - 1];
                auto const cost = leftCount > 0 ? halfArea(left) * static_cast<float>(leftCount) + rightCosts[bin]
                                                : kInfinity;
                if (cost < best.cost) {
                    best = Split{axis, bin, cost};
                }
            }
        }
        return best;
    }

    // Halves the primitives along the widest extent of their centres.
    std::uint32_t medianSplit(Task const &task, core::Aabb const &centreBox) {
        auto const extent = centreBox.max - centreBox.min;
        auto const axis = extent.x >= extent.y and extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
        auto const middle = task.begin + (task.end - task.begin) / 2;
        std::nth_element(result.order.begin() + task.begin, result.order.begin() + middle,
                         result.order.begin() + task.end, [&](std::uint32_t const a, std::uint32_t const b) {
                             return centres[a][axis] < centres[b][axis];
                         });
        return middle;
    }

    std::vector<core::Aabb> const &bounds;
    BvhBuildConfig config;
    std::vector<core::vec3> centres;
    Bvh result;
};

} // namespace

Bvh buildBvh(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config) {
    if (bounds.empty()) {
        return Bvh{};
    }
    if (ceilLog2(bounds.size()) > bvhMaxDepth() or config.maxLeafSize == 0) {
        throw std::length_error{"lr::render: cannot build a hierarchy over " + std::to_string(bounds.size())
                                + " primitives with leaves of " + std::to_string(config.maxLeafSize)};
    }
    return Builder{bounds, config}.run();
}

core::Aabb boundsOf(GpuBvhNode const &node) noexcept { return core::Aabb{node.boundsMin, node.boundsMax}; }

} // namespace lr::render
