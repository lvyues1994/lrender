#include "FrameGraphImpl.h"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace lr::render {

namespace {

struct PassContextImpl final : PassContext {
    PassContextImpl(tc::Device &device_, SceneTables const &scene_, std::vector<CompiledResource> const &descs_,
                    std::vector<std::shared_ptr<void>> const &resources_,
                    std::vector<RecordedDispatch> &dispatches_) noexcept
        : gpu(device_), tables(scene_), descs(descs_), resources(resources_), dispatches(dispatches_) {}

    tc::Device &device() noexcept override { return gpu; }
    SceneTables const &scene() const noexcept override { return tables; }

    void enterPass(std::uint32_t const index) noexcept { pass = index; }

protected:
    void *resource(std::uint32_t const index, std::type_info const &type) override {
        if (index >= resources.size()) {
            throw std::logic_error{"lr::render: a pass used a resource of another graph"};
        }
        if (descs[index].desc.type != std::type_index{type}) {
            throw std::logic_error{"lr::render: graph resource '" + descs[index].desc.name + "' used as another type"};
        }
        return resources[index].get();
    }

    void dispatched(std::string_view const program) override {
        dispatches.push_back(RecordedDispatch{pass, std::string{program}});
    }

private:
    tc::Device &gpu;
    SceneTables const &tables;
    std::vector<CompiledResource> const &descs;
    std::vector<std::shared_ptr<void>> const &resources;
    std::vector<RecordedDispatch> &dispatches;
    std::uint32_t pass = 0;
};

} // namespace

HistoryPair FrameHistories::acquire(HistoryRequest const &request) {
    auto const key = Key{request.view, request.name};
    auto const found = records.find(key);
    if (found != records.end() and found->second.type == request.type and found->second.size == request.size) {
        auto &record = found->second;
        if (record.lastFrame != request.frame) {
            record.current ^= 1u;
            record.lastFrame = request.frame;
        }
        record.target = request.target;
        return HistoryPair{record.ids[record.current ^ 1u], record.ids[record.current],
                           record.createdFrame != request.frame};
    }
    if (found != records.end()) {
        replaced.insert(replaced.end(), found->second.ids.begin(), found->second.ids.end());
        records.erase(found);
    }
    auto const ids = std::array<std::uint64_t, 2>{nextId, nextId + 1};
    nextId += 2;
    records.emplace(key, Record{request.target, request.type, request.size, ids, 0, request.frame, request.frame});
    return HistoryPair{ids[1], ids[0], false};
}

std::vector<std::uint64_t> FrameHistories::expire(std::uint64_t const frame,
                                                  std::function<bool(RenderTargetId)> const &targetExists) {
    auto released = std::move(replaced);
    replaced.clear();
    for (auto record = records.begin(); record != records.end();) {
        auto const &value = record->second;
        if (value.lastFrame + keepFrames < frame or not targetExists(value.target)) {
            released.insert(released.end(), value.ids.begin(), value.ids.end());
            record = records.erase(record);
        } else {
            ++record;
        }
    }
    return released;
}

void FrameHistories::restart() {
    for (auto const &[key, record] : records) {
        replaced.insert(replaced.end(), record.ids.begin(), record.ids.end());
    }
    records.clear();
}

std::shared_ptr<void> HistoryResources::acquire(std::uint64_t const id, ResourceDesc const &desc, tc::Device &device) {
    auto &resource = resources[id];
    if (resource == nullptr) {
        resource = desc.create(device);
    }
    return resource;
}

void HistoryResources::release(std::vector<std::uint64_t> const &ids, RetireList &retired) {
    for (auto const id : ids) {
        auto const found = resources.find(id);
        if (found != resources.end()) {
            retired.push_back(std::move(found->second));
            resources.erase(found);
        }
    }
}

PassBuilder &GraphBuilder::addPass(std::string name) {
    passes.push_back(std::make_unique<Pass>(std::move(name)));
    return *passes.back();
}

std::uint32_t GraphBuilder::declareResource(ResourceDesc desc) {
    resources.push_back(CompiledResource{std::move(desc), std::nullopt, std::nullopt, false});
    return static_cast<std::uint32_t>(resources.size() - 1);
}

FrameGraphBuilder::DeclaredHistory GraphBuilder::declareHistory(ResourceDesc desc) {
    if (owner.histories == nullptr) {
        throw std::logic_error{"lr::render: history '" + desc.name + "' asked of a graph that belongs to no view"};
    }
    auto const found = histories.find(desc.name);
    if (found != histories.end()) {
        auto const &declared = resources[found->second.current].desc;
        if (declared.type != desc.type or declared.size != desc.size) {
            throw std::logic_error{"lr::render: history '" + desc.name + "' asked for as two different images"};
        }
        return found->second;
    }
    auto const pair = owner.histories->acquire(
        HistoryRequest{owner.view, owner.target, desc.name, desc.type, desc.size, owner.frame});
    auto previous = desc;
    previous.name += " (previous)";
    auto current = std::move(desc);
    auto const name = current.name;
    current.name += " (current)";
    resources.push_back(CompiledResource{std::move(previous), pair.previous, std::nullopt, false});
    resources.push_back(CompiledResource{std::move(current), pair.current, std::nullopt, false});
    auto const last = static_cast<std::uint32_t>(resources.size() - 1);
    auto const declared = DeclaredHistory{last - 1, last, pair.valid};
    histories.emplace(name, declared);
    return declared;
}

void GraphBuilder::addReadback(std::uint32_t const resource, ReadbackHandler handler) {
    if (resource >= resources.size() or resources[resource].desc.access != BufferAccess::Readback) {
        throw std::logic_error{"lr::render: readback() needs a buffer from createReadbackBuffer()"};
    }
    readbacks.push_back(CompiledReadback{resource, std::move(handler)});
}

CompiledGraph GraphBuilder::compile() && {
    auto const checkIndex = [this](std::uint32_t const resource, Pass const &pass) {
        if (resource >= resources.size()) {
            throw std::logic_error{"lr::render: pass '" + pass.name + "' declared a resource of another graph"};
        }
    };
    // History is written by an earlier frame and read by a later one.
    auto needed = std::vector<bool>(resources.size(), false);
    auto written = std::vector<bool>(resources.size(), false);
    for (auto index = std::size_t{0}; index < resources.size(); ++index) {
        needed[index] = resources[index].history.has_value() or resources[index].external;
        written[index] = resources[index].history.has_value();
    }
    for (auto const &readback : readbacks) {
        needed[readback.resource] = true;
    }
    auto runs = std::vector<bool>(passes.size(), false);
    for (auto index = passes.size(); index-- > 0;) {
        auto const &pass = *passes[index];
        for (auto const resource : pass.writeSet) {
            checkIndex(resource, pass);
            runs[index] = runs[index] or needed[resource];
        }
        if (runs[index]) {
            for (auto const resource : pass.readSet) {
                checkIndex(resource, pass);
                needed[resource] = true;
            }
        }
    }

    auto compiled = CompiledGraph{};
    for (auto index = std::size_t{0}; index < passes.size(); ++index) {
        if (not runs[index]) {
            continue;
        }
        auto &pass = *passes[index];
        for (auto const resource : pass.readSet) {
            if (not written[resource]) {
                throw std::logic_error{"lr::render: pass '" + pass.name + "' reads '" + resources[resource].desc.name
                                       + "' before any pass writes it"};
            }
        }
        if (not pass.recorder) {
            throw std::logic_error{"lr::render: pass '" + pass.name + "' never called record()"};
        }
        for (auto const resource : pass.writeSet) {
            written[resource] = true;
        }
        auto const place = static_cast<std::uint32_t>(compiled.passes.size());
        for (auto const *set : {&pass.readSet, &pass.writeSet}) {
            for (auto const resource : *set) {
                auto &lifetime = resources[resource].lifetime;
                lifetime = lifetime ? Lifetime{lifetime->first, place} : Lifetime{place, place};
            }
        }
        compiled.passes.push_back(
            CompiledPass{std::move(pass.name), std::move(pass.recorder), std::move(pass.readSet), std::move(pass.writeSet)});
    }
    // The host reads readback buffers once the frame is done.
    for (auto const &readback : readbacks) {
        auto &lifetime = resources[readback.resource].lifetime;
        auto const end = static_cast<std::uint32_t>(compiled.passes.size());
        lifetime = Lifetime{lifetime ? lifetime->first : end, end};
    }
    compiled.resources = std::move(resources);
    compiled.readbacks = std::move(readbacks);
    compiled.finishers = std::move(finishers);
    return compiled;
}

HeapPlan planHeap(std::vector<HeapItem> items) {
    auto order = std::vector<std::size_t>(items.size());
    for (auto index = std::size_t{0}; index < order.size(); ++index) {
        order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&items](std::size_t const a, std::size_t const b) { return items[a].size.bytes > items[b].size.bytes; });
    auto plan = HeapPlan{};
    auto placedSoFar = std::vector<std::size_t>{};
    for (auto const index : order) {
        auto &item = items[index];
        auto const overlaps = [&item](HeapItem const &other) {
            return other.lifetime.first <= item.lifetime.last and item.lifetime.first <= other.lifetime.last;
        };
        auto neighbours = std::vector<HeapItem const *>{};
        for (auto const other : placedSoFar) {
            if (overlaps(items[other])) {
                neighbours.push_back(&items[other]);
            }
        }
        std::sort(neighbours.begin(), neighbours.end(),
                  [](HeapItem const *a, HeapItem const *b) { return a->offset < b->offset; });
        auto const align = [&item](std::size_t const offset) {
            return (offset + item.size.alignment - 1) / item.size.alignment * item.size.alignment;
        };
        auto offset = std::size_t{0};
        for (auto const *neighbour : neighbours) {
            if (offset + item.size.bytes <= neighbour->offset) {
                break;
            }
            offset = std::max(offset, align(neighbour->offset + neighbour->size.bytes));
        }
        item.offset = offset;
        plan.bytes = std::max(plan.bytes, offset + item.size.bytes);
        placedSoFar.push_back(index);
    }
    plan.items = std::move(items);
    return plan;
}

std::shared_ptr<void> ResourcePool::acquire(ResourceDesc const &desc, tc::Device &device) {
    auto &free = available[Key{desc.type, desc.size, desc.access}];
    if (free.empty()) {
        return desc.create(device);
    }
    auto resource = std::move(free.back().value);
    free.pop_back();
    return resource;
}

void ResourcePool::release(ResourceDesc const &desc, std::shared_ptr<void> resource) {
    available[Key{desc.type, desc.size, desc.access}].push_back(Kept<std::shared_ptr<void>>{std::move(resource), frame});
}

tc::PlacementSize ResourcePool::measure(ResourceDesc const &desc, tc::Device &device) {
    auto const key = Key{desc.type, desc.size, desc.access};
    auto const found = sizes.find(key);
    if (found != sizes.end()) {
        return found->second;
    }
    return sizes.emplace(key, desc.measure(device)).first->second;
}

ResourcePool::PlanKey ResourcePool::keyOf(HeapPlan const &plan, std::vector<CompiledResource> const &resources) {
    auto key = PlanKey{plan.bytes, {}};
    for (auto const &item : plan.items) {
        auto const &desc = resources[item.resource].desc;
        key.second.emplace_back(desc.type, desc.size, item.offset);
    }
    return key;
}

PlacedResources ResourcePool::acquirePlaced(HeapPlan const &plan, std::vector<CompiledResource> const &resources,
                                            tc::Device &device) {
    auto &free = placed[keyOf(plan, resources)];
    if (not free.empty()) {
        auto set = std::move(free.back().value);
        free.pop_back();
        return set;
    }
    auto set = PlacedResources{device.createHeap(plan.bytes), {}};
    for (auto const &item : plan.items) {
        set.resources.push_back(resources[item.resource].desc.createPlaced(device, tc::Placement{set.heap, item.offset}));
    }
    return set;
}

void ResourcePool::releasePlaced(HeapPlan const &plan, std::vector<CompiledResource> const &resources,
                                 PlacedResources set) {
    placed[keyOf(plan, resources)].push_back(Kept<PlacedResources>{std::move(set), frame});
}

void ResourcePool::endFrame() {
    ++frame;
    auto const expire = [this](auto &map) {
        for (auto entry = map.begin(); entry != map.end();) {
            auto &kept = entry->second;
            kept.erase(std::remove_if(kept.begin(), kept.end(),
                                      [this](auto const &item) { return item.frame + keepFrames < frame; }),
                       kept.end());
            entry = kept.empty() ? map.erase(entry) : std::next(entry);
        }
    };
    expire(available);
    expire(placed);
}

void ResourcePool::clear() noexcept {
    available.clear();
    sizes.clear();
    placed.clear();
}

GraphRun::GraphRun(CompiledGraph graph_, ResourcePool &pool_, HistoryResources &histories, tc::Device &device,
                   bool const alias, ExternalResources const &externals)
    : graph(std::move(graph_)), pool(&pool_), resources(graph.resources.size()), bytes(graph.resources.size(), 0),
      inHeap(graph.resources.size(), false) {
    auto const placedImages = alias and device.info().placedImages;
    auto shared = std::vector<HeapItem>{};
    for (auto index = std::uint32_t{0}; index < graph.resources.size(); ++index) {
        auto const &resource = graph.resources[index];
        if (resource.history) {
            resources[index] = histories.acquire(*resource.history, resource.desc, device);
            continue;
        }
        if (not resource.lifetime) {
            continue;
        }
        if (resource.external) {
            auto const found = std::find_if(externals.begin(), externals.end(),
                                            [index](auto const &external) { return external.first == index; });
            if (found == externals.end()) {
                throw std::logic_error{"lr::render: nothing was handed over for external buffer '"
                                       + resource.desc.name + "'"};
            }
            resources[index] = found->second;
            continue;
        }
        auto const &desc = resource.desc;
        auto const size = desc.measure ? pool->measure(desc, device) : tc::PlacementSize{};
        bytes[index] = size.bytes;
        if (alias and desc.createPlaced and (placedImages or not desc.image)) {
            shared.push_back(HeapItem{index, *resource.lifetime, size, 0});
        } else {
            resources[index] = pool->acquire(desc, device);
        }
    }
    if (not shared.empty()) {
        plan = planHeap(std::move(shared));
        placed = pool->acquirePlaced(plan, graph.resources, device);
        for (auto item = std::size_t{0}; item < plan.items.size(); ++item) {
            resources[plan.items[item].resource] = placed.resources[item];
            inHeap[plan.items[item].resource] = true;
        }
    }
}

GraphRun::~GraphRun() { releaseResources(); }

void GraphRun::record(tc::Device &device, SceneTables const &scene) {
    auto context = PassContextImpl{device, scene, graph.resources, resources, recorded};
    for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
        context.enterPass(static_cast<std::uint32_t>(index));
        graph.passes[index].recorder(context);
    }
}

std::vector<PassTiming> GraphRun::passTimings(std::vector<tc::KernelTiming> const &timings,
                                              std::size_t const first) const {
    auto result = std::vector<PassTiming>{};
    result.reserve(graph.passes.size());
    for (auto const &pass : graph.passes) {
        result.push_back(PassTiming{pass.name, 0, 0.0});
    }
    for (auto index = std::size_t{0}; index < recorded.size(); ++index) {
        auto &timing = result[recorded[index].pass];
        auto const &measured = timings.at(first + index).milliseconds;
        timing.dispatches += 1;
        timing.milliseconds = timing.milliseconds and measured ? std::optional<double>{*timing.milliseconds + *measured}
                                                               : std::nullopt;
    }
    return result;
}

void GraphRun::finish(ViewResult &result) {
    for (auto const &readback : graph.readbacks) {
        auto const &desc = graph.resources[readback.resource].desc;
        readback.handler(desc.hostData(resources[readback.resource].get()), desc.size, result);
    }
    for (auto const &finisher : graph.finishers) {
        finisher(result);
    }
    auto &stats = result.stats;
    stats.graphBytes = plan.bytes;
    stats.unaliasedBytes = 0;
    for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
        stats.unaliasedBytes += bytes[index];
        stats.graphBytes += inHeap[index] ? 0 : bytes[index];
    }
    if (graph.describe) {
        result.graph = describe(result);
    }
    releaseResources();
}

namespace {

std::string quoted(std::string const &text) {
    auto result = std::string{"\""};
    for (auto const c : text) {
        if (c == '\n') {
            result += "\\n";
            continue;
        }
        if (c == '"' or c == '\\') {
            result += '\\';
        }
        result += c;
    }
    return result + '"';
}

std::string megabytes(std::uint64_t const bytes) {
    auto stream = std::ostringstream{};
    stream << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
    return stream.str();
}

std::string shapeOf(ResourceDesc const &desc) {
    if (desc.image) {
        return std::to_string(desc.size >> 32) + "x" + std::to_string(desc.size & 0xFFFFFFFFu);
    }
    return std::to_string(desc.size) + " elements";
}

} // namespace

std::string GraphRun::describe(ViewResult const &result) const {
    auto out = std::ostringstream{};
    out << "digraph frame {\n  rankdir=LR;\n  node [fontname=monospace];\n";
    out << "  label=" << quoted("view " + std::to_string(result.view.value) + ": " + megabytes(result.stats.graphBytes)
                                + " of " + megabytes(result.stats.unaliasedBytes) + " unaliased, heap "
                                + megabytes(plan.bytes))
        << ";\n";
    for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
        auto label = std::to_string(index) + " " + graph.passes[index].name;
        if (index < result.passTimings.size()) {
            auto const &timing = result.passTimings[index];
            auto stream = std::ostringstream{};
            stream << "\n" << timing.dispatches << " dispatches";
            if (timing.milliseconds) {
                stream << ", " << std::fixed << std::setprecision(3) << *timing.milliseconds << " ms";
            }
            label += stream.str();
        }
        out << "  p" << index << " [shape=box, label=" << quoted(label) << "];\n";
    }
    auto offsets = std::vector<std::optional<std::size_t>>(graph.resources.size());
    for (auto const &item : plan.items) {
        offsets[item.resource] = item.offset;
    }
    for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
        auto const &resource = graph.resources[index];
        if (not resource.lifetime) {
            continue;
        }
        auto label = resource.desc.name + "\n" + shapeOf(resource.desc);
        if (resource.history) {
            label += ", history";
        } else if (resource.external) {
            label += ", external";
        } else if (bytes[index] != 0) {
            label += ", " + megabytes(bytes[index]);
        }
        label += "\npasses " + std::to_string(resource.lifetime->first) + " to " + std::to_string(resource.lifetime->last);
        if (offsets[index]) {
            label += "\nheap at " + megabytes(*offsets[index]);
        }
        out << "  r" << index << " [shape=ellipse, label=" << quoted(label) << "];\n";
    }
    for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
        for (auto const resource : graph.passes[index].reads) {
            out << "  r" << resource << " -> p" << index << ";\n";
        }
        for (auto const resource : graph.passes[index].writes) {
            out << "  p" << index << " -> r" << resource << ";\n";
        }
    }
    out << "}\n";
    return out.str();
}

void GraphRun::releaseResources() noexcept {
    if (pool == nullptr) {
        return;
    }
    for (auto index = std::size_t{0}; index < resources.size(); ++index) {
        auto const &resource = graph.resources[index];
        if (resources[index] != nullptr and not resource.history and not resource.external and not inHeap[index]) {
            pool->release(graph.resources[index].desc, std::move(resources[index]));
        }
    }
    resources.clear();
    if (placed.heap) {
        pool->releasePlaced(plan, graph.resources, std::move(placed));
        placed = PlacedResources{};
    }
}

} // namespace lr::render
