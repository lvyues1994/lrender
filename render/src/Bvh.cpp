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
// Primitives per item where buildBvh() spreads one node's work over a ParallelFor: fixed,
// so that the result does not depend on how many threads run the items.
constexpr auto kChunk = std::uint32_t{8192};

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

// The primitives of a node binned along each axis by their centres.
struct Bins {
    std::array<std::array<core::Aabb, kBins>, 3> boxes;
    std::array<std::array<std::uint32_t, kBins>, 3> counts{};

    Bins() {
        for (auto &axis : boxes) {
            axis.fill(emptyBox());
        }
    }

    void add(Bins const &other) {
        for (auto axis = std::size_t{0}; axis < 3; ++axis) {
            for (auto bin = std::uint32_t{0}; bin < kBins; ++bin) {
                boxes[axis][bin] = merged(boxes[axis][bin], other.boxes[axis][bin]);
                counts[axis][bin] += other.counts[axis][bin];
            }
        }
    }
};

struct Bounds {
    core::Aabb box = emptyBox();
    core::Aabb centres = emptyBox();
};


struct Builder {
    Builder(std::vector<core::Aabb> const &bounds_, BvhBuildConfig const &config_, ParallelFor *const parallel_ = nullptr)
        : bounds(bounds_), config(config_), parallel(parallel_), centres(bounds.size()) {
        result.order.resize(bounds.size());
        forChunks(bounds.size(), [&](std::size_t const index) {
            centres[index] = (bounds[index].min + bounds[index].max) * 0.5f;
            result.order[index] = static_cast<std::uint32_t>(index);
        });
    }

    Bvh run() && {
        result.nodes.reserve(2 * bounds.size() - 1);
        result.nodes.push_back(GpuBvhNode{});
        buildSubtree(Task{0, 0, static_cast<std::uint32_t>(bounds.size()), 0}, result.nodes);
        return std::move(result);
    }

    // Nodes with more primitives than the grain are split with their work in chunks over
    // `parallel`, depth first; the subtrees below them are built as items of it, and laid
    // after those nodes in the order they were reached.
    Bvh runParallel() && {
        result.nodes.reserve(2 * bounds.size() - 1);
        result.nodes.push_back(GpuBvhNode{});
        auto subtrees = std::vector<Task>{};
        auto tasks = std::vector<Task>{Task{0, 0, static_cast<std::uint32_t>(bounds.size()), 0}};
        while (not tasks.empty()) {
            auto const task = tasks.back();
            tasks.pop_back();
            if (task.end - task.begin <= std::max(config.parallelGrain, config.maxLeafSize)) {
                subtrees.push_back(task);
                continue;
            }
            auto const found = extentOf(task);
            auto const middle = splitPoint(task, found, [&] { return binsInChunks(task, found.centres); }, true);
            split(task, found.box, middle, result.nodes, tasks);
        }
        auto built = std::vector<std::vector<GpuBvhNode>>(subtrees.size());
        parallel->run(subtrees.size(), [&](std::size_t const index) {
            auto local = subtrees[index];
            local.node = 0;
            built[index].push_back(GpuBvhNode{});
            buildSubtree(local, built[index]);
        });
        // Each subtree's root where its parent put it, the rest after the nodes before.
        auto bases = std::vector<std::uint32_t>(subtrees.size());
        auto end = static_cast<std::uint32_t>(result.nodes.size());
        for (auto index = std::size_t{0}; index < subtrees.size(); ++index) {
            bases[index] = end - 1;
            end += static_cast<std::uint32_t>(built[index].size()) - 1;
        }
        result.nodes.resize(end);
        parallel->run(subtrees.size(), [&](std::size_t const index) {
            auto const relocated = [base = bases[index]](GpuBvhNode node) {
                node.first += node.count == 0 ? base : 0u;
                return node;
            };
            result.nodes[subtrees[index].node] = relocated(built[index][0]);
            for (auto local = std::size_t{1}; local < built[index].size(); ++local) {
                result.nodes[bases[index] + local] = relocated(built[index][local]);
            }
        });
        return std::move(result);
    }

private:
    // f(index) for every index below `count`, in chunks over `parallel` when there is one.
    template <class F>
    void forChunks(std::size_t const count, F const &f) const {
        auto const chunks = (count + kChunk - 1) / kChunk;
        auto const chunk = [&](std::size_t const at) {
            for (auto index = at * kChunk; index < std::min(count, (at + 1) * kChunk); ++index) {
                f(index);
            }
        };
        if (parallel != nullptr) {
            parallel->run(chunks, chunk);
        } else {
            for (auto at = std::size_t{0}; at < chunks; ++at) {
                chunk(at);
            }
        }
    }

    void buildSubtree(Task const &root, std::vector<GpuBvhNode> &nodes) {
        auto tasks = std::vector<Task>{root};
        while (not tasks.empty()) {
            auto const task = tasks.back();
            tasks.pop_back();
            auto found = Bounds{};
            for (auto index = task.begin; index < task.end; ++index) {
                auto const primitive = result.order[index];
                found.box = merged(found.box, bounds[primitive]);
                found.centres = merged(found.centres, centres[primitive]);
            }
            auto const middle = splitPoint(task, found, [&] { return binsOf(task, found.centres); }, false);
            split(task, found.box, middle, nodes, tasks);
        }
    }

    static void split(Task const &task, core::Aabb const &box, std::optional<std::uint32_t> const middle,
                      std::vector<GpuBvhNode> &nodes, std::vector<Task> &tasks) {
        if (not middle) {
            nodes[task.node] = GpuBvhNode{box.min, task.begin, box.max, task.end - task.begin};
            return;
        }
        auto const left = static_cast<std::uint32_t>(nodes.size());
        nodes.resize(nodes.size() + 2);
        nodes[task.node] = GpuBvhNode{box.min, left, box.max, 0};
        tasks.push_back(Task{left + 1, *middle, task.end, task.depth + 1});
        tasks.push_back(Task{left, task.begin, *middle, task.depth + 1});
    }

    static std::uint32_t chunksOf(Task const &task) noexcept { return (task.end - task.begin + kChunk - 1) / kChunk; }

    Bounds extentOf(Task const &task) const {
        auto partial = std::vector<Bounds>(chunksOf(task));
        parallel->run(partial.size(), [&](std::size_t const chunk) {
            auto const first = task.begin + static_cast<std::uint32_t>(chunk) * kChunk;
            for (auto index = first; index < std::min(first + kChunk, task.end); ++index) {
                auto const primitive = result.order[index];
                partial[chunk].box = merged(partial[chunk].box, bounds[primitive]);
                partial[chunk].centres = merged(partial[chunk].centres, centres[primitive]);
            }
        });
        auto found = Bounds{};
        for (auto const &part : partial) {
            found.box = merged(found.box, part.box);
            found.centres = merged(found.centres, part.centres);
        }
        return found;
    }

    // Along axes along which the centres spread; the others stay empty. One pass over the
    // primitives for all three.
    void binRange(std::uint32_t const begin, std::uint32_t const end, core::Aabb const &centreBox, Bins &bins) const {
        auto const spreads = std::array<bool, 3>{centreBox.max.x > centreBox.min.x, centreBox.max.y > centreBox.min.y,
                                                 centreBox.max.z > centreBox.min.z};
        for (auto index = begin; index < end; ++index) {
            auto const primitive = result.order[index];
            for (auto axis = std::size_t{0}; axis < 3; ++axis) {
                if (spreads[axis]) {
                    auto const bin = binOf(centres[primitive], static_cast<int>(axis), centreBox);
                    bins.boxes[axis][bin] = merged(bins.boxes[axis][bin], bounds[primitive]);
                    bins.counts[axis][bin] += 1;
                }
            }
        }
    }

    Bins binsOf(Task const &task, core::Aabb const &centreBox) const {
        auto bins = Bins{};
        binRange(task.begin, task.end, centreBox, bins);
        return bins;
    }

    Bins binsInChunks(Task const &task, core::Aabb const &centreBox) const {
        auto partial = std::vector<Bins>(chunksOf(task));
        parallel->run(partial.size(), [&](std::size_t const chunk) {
            auto const first = task.begin + static_cast<std::uint32_t>(chunk) * kChunk;
            binRange(first, std::min(first + kChunk, task.end), centreBox, partial[chunk]);
        });
        auto bins = Bins{};
        for (auto const &part : partial) {
            bins.add(part);
        }
        return bins;
    }

    // Where the primitives of `task` split, after reordering them; nullopt for a leaf.
    // `binned()` bins them, only where a split is looked for. `inChunks` reorders them
    // keeping each side's order, in chunks over `parallel`.
    template <class Binned>
    std::optional<std::uint32_t> splitPoint(Task const &task, Bounds const &found, Binned const &binned,
                                            bool const inChunks) {
        auto const count = task.end - task.begin;
        auto const fitsLeaf = count <= config.maxLeafSize;
        if (count == 1) {
            return std::nullopt;
        }
        // Halving from here on still ends within the depth limit.
        if (task.depth + ceilLog2(count) >= bvhMaxDepth()) {
            return fitsLeaf ? std::nullopt : std::optional<std::uint32_t>{medianSplit(task, found.centres)};
        }
        auto const area = halfArea(found.box);
        auto const best = area > 0.0f ? bestSplit(binned(), found.centres) : Split{};
        if (best.axis < 0) {
            return fitsLeaf ? std::nullopt : std::optional<std::uint32_t>{medianSplit(task, found.centres)};
        }
        // Costs in primitive tests, with visiting a node as dear as testing a primitive.
        auto const splitCost = 1.0f + best.cost / area;
        if (fitsLeaf and static_cast<float>(count) <= splitCost) {
            return std::nullopt;
        }
        auto const left = [&](std::uint32_t const primitive) {
            return binOf(centres[primitive], best.axis, found.centres) < best.bin;
        };
        if (inChunks) {
            return stablePartition(task, left);
        }
        auto const first = result.order.begin() + task.begin;
        auto const middle = std::partition(first, result.order.begin() + task.end, left);
        return task.begin + static_cast<std::uint32_t>(middle - first);
    }

    // As std::stable_partition: each chunk's primitives on each side go after those of
    // the chunks before it.
    template <class Left>
    std::uint32_t stablePartition(Task const &task, Left const &left) {
        auto const chunks = chunksOf(task);
        auto lefts = std::vector<std::uint32_t>(chunks, 0);
        parallel->run(chunks, [&](std::size_t const chunk) {
            auto const first = task.begin + static_cast<std::uint32_t>(chunk) * kChunk;
            for (auto index = first; index < std::min(first + kChunk, task.end); ++index) {
                lefts[chunk] += left(result.order[index]) ? 1u : 0u;
            }
        });
        auto const totalLeft = std::accumulate(lefts.begin(), lefts.end(), std::uint32_t{0});
        auto firstLeft = std::vector<std::uint32_t>(chunks, 0);
        auto firstRight = std::vector<std::uint32_t>(chunks, 0);
        for (auto chunk = std::uint32_t{0}, leftsBefore = 0u; chunk < chunks; ++chunk) {
            firstLeft[chunk] = leftsBefore;
            firstRight[chunk] = totalLeft + chunk * kChunk - leftsBefore;
            leftsBefore += lefts[chunk];
        }
        auto placed = std::vector<std::uint32_t>(task.end - task.begin);
        parallel->run(chunks, [&](std::size_t const chunk) {
            auto toLeft = firstLeft[chunk];
            auto toRight = firstRight[chunk];
            auto const first = task.begin + static_cast<std::uint32_t>(chunk) * kChunk;
            for (auto index = first; index < std::min(first + kChunk, task.end); ++index) {
                auto const primitive = result.order[index];
                placed[left(primitive) ? toLeft++ : toRight++] = primitive;
            }
        });
        std::copy(placed.begin(), placed.end(), result.order.begin() + task.begin);
        return task.begin + totalLeft;
    }

    // Only for axes along which the centres spread.
    static std::uint32_t binOf(core::vec3 const &centre, int const axis, core::Aabb const &centreBox) noexcept {
        auto const low = centreBox.min[axis];
        auto const ratio = (centre[axis] - low) / (centreBox.max[axis] - low);
        return static_cast<std::uint32_t>(std::min(ratio * static_cast<float>(kBins), static_cast<float>(kBins - 1)));
    }

    static Split bestSplit(Bins const &bins, core::Aabb const &centreBox) {
        auto best = Split{};
        for (auto axis = 0; axis < 3; ++axis) {
            if (not(centreBox.max[axis] > centreBox.min[axis])) {
                continue;
            }
            auto const &boxes = bins.boxes[static_cast<std::size_t>(axis)];
            auto const &counts = bins.counts[static_cast<std::size_t>(axis)];
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
    ParallelFor *parallel;
    std::vector<core::vec3> centres;
    Bvh result;
};

void checkBuildable(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config) {
    if (ceilLog2(bounds.size()) > bvhMaxDepth() or config.maxLeafSize == 0) {
        throw std::length_error{"lr::render: cannot build a hierarchy over " + std::to_string(bounds.size())
                                + " primitives with leaves of " + std::to_string(config.maxLeafSize)};
    }
}

} // namespace

Bvh buildBvh(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config) {
    if (bounds.empty()) {
        return Bvh{};
    }
    checkBuildable(bounds, config);
    return Builder{bounds, config}.run();
}

Bvh buildBvh(std::vector<core::Aabb> const &bounds, BvhBuildConfig const &config, ParallelFor &parallel) {
    if (bounds.empty()) {
        return Bvh{};
    }
    checkBuildable(bounds, config);
    return Builder{bounds, config, &parallel}.runParallel();
}

core::Aabb boundsOf(GpuBvhNode const &node) noexcept { return core::Aabb{node.boundsMin, node.boundsMax}; }

} // namespace lr::render
