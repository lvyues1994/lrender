#pragma once

#include <lr/render/FrameGraph.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace lr::render {

struct CompiledPass {
    std::string name;
    PassRecorder recorder;
};

struct CompiledReadback {
    std::uint32_t resource;
    std::function<void(void const *data, std::size_t count, ViewResult &)> handler;
};

struct CompiledGraph {
    std::vector<ResourceDesc> resources;
    std::vector<CompiledPass> passes; // those the outputs need, in declaration order
    std::vector<CompiledReadback> readbacks;
};

// Builds one view's graph on the render thread.
struct GraphBuilder final : FrameGraphBuilder {
    PassBuilder &addPass(std::string name) override;
    Blackboard &blackboard() noexcept override { return board; }

    // Drops the passes no output depends on. Throws std::logic_error when a pass that runs
    // reads a resource no earlier pass writes, or never called record().
    CompiledGraph compile() &&;

protected:
    std::uint32_t declareResource(ResourceDesc desc) override;
    void addReadback(std::uint32_t resource, ReadbackHandler handler) override;

private:
    struct Pass final : PassBuilder {
        explicit Pass(std::string name_) : name(std::move(name_)) {}

        void record(PassRecorder recorder_) override { recorder = std::move(recorder_); }

        std::string name;
        std::vector<std::uint32_t> readSet;
        std::vector<std::uint32_t> writeSet;
        PassRecorder recorder;

    protected:
        void declare(std::uint32_t const resource, bool const writes) override {
            (writes ? writeSet : readSet).push_back(resource);
        }
    };

    std::vector<ResourceDesc> resources;
    std::vector<std::unique_ptr<Pass>> passes;
    std::vector<CompiledReadback> readbacks;
    Blackboard board;
};

// Graph resources the GPU thread keeps between frames, reused by type and size.
struct ResourcePool {
    std::shared_ptr<void> acquire(ResourceDesc const &desc, tc::Device &device);
    void release(ResourceDesc const &desc, std::shared_ptr<void> resource);
    void clear() noexcept { available.clear(); }

private:
    using Key = std::tuple<std::type_index, std::uint64_t, bool>;
    std::map<Key, std::vector<std::shared_ptr<void>>> available;
};

// One view's graph on the GPU thread: holds its resources from record() until finish(),
// which must wait until the device has finished the frame.
struct GraphRun {
    GraphRun(CompiledGraph graph_, ResourcePool &pool_, tc::Device &device);
    GraphRun(GraphRun &&) noexcept = default;
    GraphRun &operator=(GraphRun &&) = delete;
    ~GraphRun();

    void record(tc::Device &device, SceneTables const &scene);
    // Calls the readback handlers, then returns the resources to the pool.
    void finish(ViewResult &result);

private:
    void releaseResources() noexcept;

    CompiledGraph graph;
    ResourcePool *pool;
    std::vector<std::shared_ptr<void>> resources;
};

} // namespace lr::render
