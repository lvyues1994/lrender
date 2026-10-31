#pragma once

#include "RetireList.h"

#include <lr/render/FrameGraph.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lr::render {

struct HistoryRequest {
    ViewId view;
    RenderTargetId target;
    std::string name;
    std::type_index type;
    std::uint64_t size;
    std::uint64_t frame;
};

struct HistoryPair {
    std::uint64_t previous;
    std::uint64_t current;
    bool valid;
};

// The views' history images on the render thread: which images a graph gets, and when
// they go. Images are ids the GPU thread turns into resources (HistoryResources).
struct FrameHistories {
    // Histories unused for more than `keepFrames` frames are released.
    explicit FrameHistories(std::uint64_t const keepFrames_) noexcept : keepFrames(keepFrames_) {}

    // The pair the view keeps under the name, swapped if the view's last use was in an
    // earlier frame; new ids when it has none, or one of another type or size.
    HistoryPair acquire(HistoryRequest const &request);
    // At the end of `frame`: the ids no graph will use again, those replaced since the
    // last call and those of histories unused for too long or whose target is gone.
    std::vector<std::uint64_t> expire(std::uint64_t frame, std::function<bool(RenderTargetId)> const &targetExists);
    // Every history starts over with new images, as after the device was lost; expire()
    // releases the old ids.
    void restart();

private:
    struct Record {
        RenderTargetId target;
        std::type_index type;
        std::uint64_t size;
        std::array<std::uint64_t, 2> ids;
        std::uint32_t current; // index into ids
        std::uint64_t createdFrame;
        std::uint64_t lastFrame;
    };
    using Key = std::pair<ViewId, std::string>;

    std::uint64_t keepFrames;
    std::map<Key, Record> records;
    std::vector<std::uint64_t> replaced;
    std::uint64_t nextId = 1;
};

// The GPU thread's history images, by FrameHistories id.
struct HistoryResources {
    std::shared_ptr<void> acquire(std::uint64_t id, ResourceDesc const &desc, tc::Device &device);
    void release(std::vector<std::uint64_t> const &ids, RetireList &retired);
    void clear() noexcept { resources.clear(); }
    std::size_t size() const noexcept { return resources.size(); }

private:
    std::unordered_map<std::uint64_t, std::shared_ptr<void>> resources;
};

struct CompiledResource {
    ResourceDesc desc;
    std::optional<std::uint64_t> history; // a FrameHistories id
};

struct CompiledPass {
    std::string name;
    PassRecorder recorder;
};

struct CompiledReadback {
    std::uint32_t resource;
    std::function<void(void const *data, std::size_t count, ViewResult &)> handler;
};

struct CompiledGraph {
    std::vector<CompiledResource> resources;
    std::vector<CompiledPass> passes; // those the outputs need, in declaration order
    std::vector<CompiledReadback> readbacks;
    std::vector<ResultHandler> finishers;
};

// The view a graph is built for, which owns its history.
struct GraphOwner {
    FrameHistories *histories = nullptr;
    ViewId view;
    RenderTargetId target;
    std::uint64_t frame = 0;
};

// Builds one view's graph on the render thread.
struct GraphBuilder final : FrameGraphBuilder {
    // A graph without history.
    GraphBuilder() = default;
    explicit GraphBuilder(GraphOwner const &owner_) : owner(owner_) {}

    PassBuilder &addPass(std::string name) override;
    void onFinish(ResultHandler handler) override { finishers.push_back(std::move(handler)); }
    Blackboard &blackboard() noexcept override { return board; }

    // Drops the passes no output or history depends on. Throws std::logic_error when a
    // pass that runs reads a resource no earlier pass writes, or never called record().
    CompiledGraph compile() &&;

protected:
    std::uint32_t declareResource(ResourceDesc desc) override;
    DeclaredHistory declareHistory(ResourceDesc desc) override;
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

    GraphOwner owner;
    std::vector<CompiledResource> resources;
    std::map<std::string, DeclaredHistory> histories;
    std::vector<std::unique_ptr<Pass>> passes;
    std::vector<CompiledReadback> readbacks;
    std::vector<ResultHandler> finishers;
    Blackboard board;
};

// Graph resources the GPU thread keeps between frames, reused by type and size.
struct ResourcePool {
    std::shared_ptr<void> acquire(ResourceDesc const &desc, tc::Device &device);
    void release(ResourceDesc const &desc, std::shared_ptr<void> resource);
    void clear() noexcept { available.clear(); }

private:
    using Key = std::tuple<std::type_index, std::uint64_t, BufferAccess>;
    std::map<Key, std::vector<std::shared_ptr<void>>> available;
};

struct RecordedDispatch {
    std::uint32_t pass; // into CompiledGraph::passes
    std::string program;
};

// One view's graph on the GPU thread: holds its resources from record() until finish() or
// its destruction, which must both wait until the device has finished the frame.
struct GraphRun {
    GraphRun(CompiledGraph graph_, ResourcePool &pool_, HistoryResources &histories, tc::Device &device);
    GraphRun(GraphRun &&) noexcept = default;
    GraphRun &operator=(GraphRun &&) = delete;
    ~GraphRun();

    void record(tc::Device &device, SceneTables const &scene);
    // What record() dispatched through PassContext, in order.
    std::vector<RecordedDispatch> const &dispatches() const noexcept { return recorded; }
    // Every pass with the device time of its dispatches, which are the timings from
    // `first` on: the device lists them in execution order.
    std::vector<PassTiming> passTimings(std::vector<tc::KernelTiming> const &timings, std::size_t first) const;
    // Calls the readback handlers and the finishers, then returns the resources to the pool.
    void finish(ViewResult &result);

private:
    void releaseResources() noexcept;

    CompiledGraph graph;
    ResourcePool *pool;
    std::vector<std::shared_ptr<void>> resources;
    std::vector<RecordedDispatch> recorded;
};

} // namespace lr::render
