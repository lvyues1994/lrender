#include "FrameGraphImpl.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lr::render {

namespace {

struct PassContextImpl final : PassContext {
    PassContextImpl(tc::Device &device_, SceneTables const &scene_, std::vector<ResourceDesc> const &descs_,
                    std::vector<std::shared_ptr<void>> const &resources_) noexcept
        : gpu(device_), tables(scene_), descs(descs_), resources(resources_) {}

    tc::Device &device() noexcept override { return gpu; }
    SceneTables const &scene() const noexcept override { return tables; }

protected:
    void *resource(std::uint32_t const index, std::type_info const &type) override {
        if (index >= resources.size()) {
            throw std::logic_error{"lr::render: a pass used a resource of another graph"};
        }
        if (descs[index].type != std::type_index{type}) {
            throw std::logic_error{"lr::render: graph resource '" + descs[index].name + "' used as another type"};
        }
        return resources[index].get();
    }

private:
    tc::Device &gpu;
    SceneTables const &tables;
    std::vector<ResourceDesc> const &descs;
    std::vector<std::shared_ptr<void>> const &resources;
};

} // namespace

PassBuilder &GraphBuilder::addPass(std::string name) {
    passes.push_back(std::make_unique<Pass>(std::move(name)));
    return *passes.back();
}

std::uint32_t GraphBuilder::declareResource(ResourceDesc desc) {
    resources.push_back(std::move(desc));
    return static_cast<std::uint32_t>(resources.size() - 1);
}

void GraphBuilder::addReadback(std::uint32_t const resource, ReadbackHandler handler) {
    if (resource >= resources.size() or not resources[resource].hostVisible) {
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
    auto needed = std::vector<bool>(resources.size(), false);
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
    auto written = std::vector<bool>(resources.size(), false);
    for (auto index = std::size_t{0}; index < passes.size(); ++index) {
        if (not runs[index]) {
            continue;
        }
        auto &pass = *passes[index];
        for (auto const resource : pass.readSet) {
            if (not written[resource]) {
                throw std::logic_error{"lr::render: pass '" + pass.name + "' reads '" + resources[resource].name
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

GraphRun::GraphRun(CompiledGraph graph_, ResourcePool &pool_, tc::Device &device)
    : graph(std::move(graph_)), pool(&pool_) {
    resources.reserve(graph.resources.size());
    for (auto const &desc : graph.resources) {
        resources.push_back(pool->acquire(desc, device));
    }
}

GraphRun::~GraphRun() { releaseResources(); }

void GraphRun::record(tc::Device &device, SceneTables const &scene) {
    auto context = PassContextImpl{device, scene, graph.resources, resources};
    for (auto const &pass : graph.passes) {
        pass.recorder(context);
    }
}

void GraphRun::finish(ViewResult &result) {
    for (auto const &readback : graph.readbacks) {
        auto const &desc = graph.resources[readback.resource];
        readback.handler(desc.hostData(resources[readback.resource].get()), desc.size, result);
    }
    releaseResources();
}

void GraphRun::releaseResources() noexcept {
    if (pool == nullptr) {
        return;
    }
    for (auto index = std::size_t{0}; index < resources.size(); ++index) {
        if (resources[index] != nullptr) {
            pool->release(graph.resources[index], std::move(resources[index]));
        }
    }
    resources.clear();
}

} // namespace lr::render
