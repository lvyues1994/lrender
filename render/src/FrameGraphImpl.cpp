#include "FrameGraphImpl.h"

#include <algorithm>
#include <stdexcept>
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
    resources.push_back(CompiledResource{std::move(desc), std::nullopt});
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
    resources.push_back(CompiledResource{std::move(previous), pair.previous});
    resources.push_back(CompiledResource{std::move(current), pair.current});
    auto const last = static_cast<std::uint32_t>(resources.size() - 1);
    auto const declared = DeclaredHistory{last - 1, last, pair.valid};
    histories.emplace(name, declared);
    return declared;
}

void GraphBuilder::addReadback(std::uint32_t const resource, ReadbackHandler handler) {
    if (resource >= resources.size() or not resources[resource].desc.hostVisible) {
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
        needed[index] = resources[index].history.has_value();
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
        compiled.passes.push_back(CompiledPass{std::move(pass.name), std::move(pass.recorder)});
    }
    compiled.resources = std::move(resources);
    compiled.readbacks = std::move(readbacks);
    compiled.finishers = std::move(finishers);
    return compiled;
}

std::shared_ptr<void> ResourcePool::acquire(ResourceDesc const &desc, tc::Device &device) {
    auto &free = available[Key{desc.type, desc.size, desc.hostVisible}];
    if (free.empty()) {
        return desc.create(device);
    }
    auto resource = std::move(free.back());
    free.pop_back();
    return resource;
}

void ResourcePool::release(ResourceDesc const &desc, std::shared_ptr<void> resource) {
    available[Key{desc.type, desc.size, desc.hostVisible}].push_back(std::move(resource));
}

GraphRun::GraphRun(CompiledGraph graph_, ResourcePool &pool_, HistoryResources &histories, tc::Device &device)
    : graph(std::move(graph_)), pool(&pool_) {
    resources.reserve(graph.resources.size());
    for (auto const &resource : graph.resources) {
        resources.push_back(resource.history ? histories.acquire(*resource.history, resource.desc, device)
                                             : pool->acquire(resource.desc, device));
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
    releaseResources();
}

void GraphRun::releaseResources() noexcept {
    if (pool == nullptr) {
        return;
    }
    for (auto index = std::size_t{0}; index < resources.size(); ++index) {
        if (resources[index] != nullptr and not graph.resources[index].history) {
            pool->release(graph.resources[index].desc, std::move(resources[index]));
        }
    }
    resources.clear();
}

} // namespace lr::render
