#include "FrameGraphImpl.h"

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using namespace lr::render;

void noop(PassContext &) {}

std::vector<std::string> passNames(CompiledGraph const &graph) {
    auto names = std::vector<std::string>{};
    for (auto const &pass : graph.passes) {
        names.push_back(pass.name);
    }
    return names;
}

TEST_CASE("only the passes an output depends on are kept, in declaration order") {
    auto graph = GraphBuilder{};
    auto const a = graph.createBuffer<tc::uint>("a", 4);
    auto const unused = graph.createBuffer<tc::uint>("unused", 4);
    auto const out = graph.createReadbackBuffer<tc::uint>("out", 4);
    auto &produce = graph.addPass("produce");
    produce.write(a);
    produce.record(noop);
    auto &wasted = graph.addPass("wasted");
    wasted.read(a);
    wasted.write(unused);
    wasted.record(noop);
    auto &consume = graph.addPass("consume");
    consume.read(a);
    consume.write(out);
    consume.record(noop);
    graph.readback<tc::uint>(out, [](tc::uint const *, std::size_t, ViewResult &) {});

    auto const compiled = std::move(graph).compile();

    CHECK(passNames(compiled) == std::vector<std::string>{"produce", "consume"});
    CHECK(compiled.readbacks.size() == 1);
}

TEST_CASE("a resource lives from the first pass that runs and uses it to the last; readbacks to the end") {
    auto graph = GraphBuilder{};
    auto const a = graph.createBuffer<tc::uint>("a", 4);
    auto const b = graph.createBuffer<tc::uint>("b", 4);
    auto const unused = graph.createBuffer<tc::uint>("unused", 4);
    auto const out = graph.createReadbackBuffer<tc::uint>("out", 4);
    auto &first = graph.addPass("first");
    first.write(a);
    first.record(noop);
    auto &wasted = graph.addPass("wasted");
    wasted.read(a);
    wasted.write(unused);
    wasted.record(noop);
    auto &second = graph.addPass("second");
    second.read(a);
    second.write(b);
    second.record(noop);
    auto &third = graph.addPass("third");
    third.read(b);
    third.write(out);
    third.record(noop);
    graph.readback<tc::uint>(out, [](tc::uint const *, std::size_t, ViewResult &) {});

    auto const compiled = std::move(graph).compile();

    auto const lifetimeOf = [&compiled](std::uint32_t const index) { return compiled.resources.at(index).lifetime; };
    REQUIRE(lifetimeOf(a.index).has_value());
    CHECK(lifetimeOf(a.index)->first == 0);
    CHECK(lifetimeOf(a.index)->last == 1);
    CHECK(lifetimeOf(b.index)->first == 1);
    CHECK(lifetimeOf(b.index)->last == 2);
    CHECK(not lifetimeOf(unused.index).has_value());
    CHECK(lifetimeOf(out.index)->first == 2);
    CHECK(lifetimeOf(out.index)->last == 3);
    CHECK(compiled.passes[1].reads == std::vector<std::uint32_t>{a.index});
    CHECK(compiled.passes[1].writes == std::vector<std::uint32_t>{b.index});
}

HeapItem item(std::uint32_t const resource, std::uint32_t const first, std::uint32_t const last, std::size_t const bytes,
              std::size_t const alignment = 16) {
    return HeapItem{resource, Lifetime{first, last}, tc::PlacementSize{bytes, alignment}, 0};
}

TEST_CASE("resources that live at different times share memory, and those that live together do not") {
    // 0 and 1 live together; 2 after 0 and 1; 3 with 2 but after 1; 4 with all of them.
    auto const plan = planHeap({item(0, 0, 1, 1000), item(1, 1, 1, 300), item(2, 2, 3, 1200), item(3, 2, 2, 100, 256),
                                item(4, 0, 3, 64, 64)});

    REQUIRE(plan.items.size() == 5);
    auto const &items = plan.items;
    auto const apart = [&items](std::size_t const a, std::size_t const b) {
        return items[a].offset + items[a].size.bytes <= items[b].offset
            or items[b].offset + items[b].size.bytes <= items[a].offset;
    };
    CHECK(apart(0, 1));
    CHECK(apart(2, 3));
    CHECK(apart(4, 0));
    CHECK(apart(4, 1));
    CHECK(apart(4, 2));
    CHECK(apart(4, 3));
    CHECK(items[2].offset == 0); // the largest first, where 0 and 1 were
    for (auto const &placed : items) {
        CHECK(placed.offset % placed.size.alignment == 0);
    }
    auto ends = std::size_t{0};
    for (auto const &placed : items) {
        ends = std::max(ends, placed.offset + placed.size.bytes);
    }
    CHECK(plan.bytes == ends);
    CHECK(plan.bytes < 1000 + 300 + 1200 + 100 + 64);
}

TEST_CASE("the same heap plan gets the same heap and resources once they are back") {
    auto request = tc::DeviceRequest{};
    request.backend = "cpu";
    auto const device = tc::createDevice(request);
    auto graph = GraphBuilder{};
    graph.createBuffer<tc::uint>("a", 64);
    graph.createBuffer<tc::vec4>("b", 16);
    auto compiled = std::move(graph).compile();
    compiled.resources[0].lifetime = Lifetime{0, 0};
    compiled.resources[1].lifetime = Lifetime{1, 1};
    auto pool = ResourcePool{2};
    auto const plan = planHeap({HeapItem{0, Lifetime{0, 0}, pool.measure(compiled.resources[0].desc, *device), 0},
                                HeapItem{1, Lifetime{1, 1}, pool.measure(compiled.resources[1].desc, *device), 0}});
    CHECK(plan.items[0].offset == plan.items[1].offset);

    auto first = pool.acquirePlaced(plan, compiled.resources, *device);
    auto const *heap = &first.heap.identity();
    auto const *buffer = first.resources[0].get();
    pool.releasePlaced(plan, compiled.resources, std::move(first));
    auto again = pool.acquirePlaced(plan, compiled.resources, *device);
    CHECK(&again.heap.identity() == heap);
    CHECK(again.resources[0].get() == buffer);
    auto other = pool.acquirePlaced(plan, compiled.resources, *device);
    CHECK(&other.heap.identity() != heap); // the first set is in use
    pool.releasePlaced(plan, compiled.resources, std::move(other));
    pool.releasePlaced(plan, compiled.resources, std::move(again));
}

TEST_CASE("a pass that reads what no earlier pass wrote is an error") {
    auto graph = GraphBuilder{};
    auto const a = graph.createBuffer<tc::uint>("a", 4);
    auto const out = graph.createReadbackBuffer<tc::uint>("out", 4);
    auto &pass = graph.addPass("reader");
    pass.read(a);
    pass.write(out);
    pass.record(noop);
    graph.readback<tc::uint>(out, [](tc::uint const *, std::size_t, ViewResult &) {});

    CHECK_THROWS_AS(std::move(graph).compile(), std::logic_error);
}

TEST_CASE("a pass that never records is an error") {
    auto graph = GraphBuilder{};
    auto const out = graph.createReadbackBuffer<tc::uint>("out", 4);
    graph.addPass("silent").write(out);
    graph.readback<tc::uint>(out, [](tc::uint const *, std::size_t, ViewResult &) {});

    CHECK_THROWS_AS(std::move(graph).compile(), std::logic_error);
}

TEST_CASE("only readback buffers can be read back") {
    auto graph = GraphBuilder{};
    auto const deviceOnly = graph.createBuffer<tc::uint>("device only", 4);

    CHECK_THROWS_AS(graph.readback<tc::uint>(deviceOnly, [](tc::uint const *, std::size_t, ViewResult &) {}),
                    std::logic_error);
}

using Format = tc::gpu::RGBA32F;

struct Owner {
    FrameHistories histories{4};
    lr::core::RenderTargetId target{0};

    GraphBuilder graphFor(lr::core::ViewId const view, std::uint64_t const frame) {
        return GraphBuilder{GraphOwner{&histories, view, target, frame}};
    }
};

std::optional<std::uint64_t> historyOf(CompiledGraph const &graph, std::uint32_t const resource) {
    return graph.resources.at(resource).history;
}

TEST_CASE("a view's history images trade places every frame") {
    auto owner = Owner{};
    auto const view = lr::core::ViewId{3};
    auto first = owner.graphFor(view, 10);
    auto const a = first.history<Format>("accumulation", lr::core::ivec2{8, 8});
    auto const again = first.history<Format>("accumulation", lr::core::ivec2{8, 8});
    auto second = owner.graphFor(view, 11);
    auto const b = second.history<Format>("accumulation", lr::core::ivec2{8, 8});
    auto const firstGraph = std::move(first).compile();
    auto const secondGraph = std::move(second).compile();

    CHECK(not a.valid);
    CHECK(again.current.index == a.current.index);
    CHECK(b.valid);
    CHECK(historyOf(secondGraph, b.previous.index) == historyOf(firstGraph, a.current.index));
    CHECK(historyOf(secondGraph, b.current.index) == historyOf(firstGraph, a.previous.index));
    CHECK(historyOf(firstGraph, a.previous.index) != historyOf(firstGraph, a.current.index));
}

TEST_CASE("history goes with a new extent, after unused frames and with its target") {
    auto owner = Owner{};
    auto const view = lr::core::ViewId{0};
    auto const always = [](lr::core::RenderTargetId) { return true; };
    auto first = owner.graphFor(view, 0);
    first.history<Format>("accumulation", lr::core::ivec2{8, 8});
    auto const old = std::move(first).compile();
    CHECK(owner.histories.expire(0, always).empty());

    auto resized = owner.graphFor(view, 1);
    auto const larger = resized.history<Format>("accumulation", lr::core::ivec2{16, 8});
    std::move(resized).compile();
    auto released = owner.histories.expire(1, always);
    auto replaced = std::vector<std::uint64_t>{*historyOf(old, 0), *historyOf(old, 1)};
    std::sort(released.begin(), released.end());
    std::sort(replaced.begin(), replaced.end());
    CHECK(not larger.valid);
    CHECK(released == replaced);

    CHECK(owner.histories.expire(5, always).empty()); // unused for 4 frames: kept
    CHECK(owner.histories.expire(6, always).size() == 2);

    auto again = owner.graphFor(view, 7);
    again.history<Format>("accumulation", lr::core::ivec2{16, 8});
    std::move(again).compile();
    auto const gone = [](lr::core::RenderTargetId) { return false; };
    CHECK(owner.histories.expire(7, gone).size() == 2);
}

TEST_CASE("history counts as written, and a pass that writes it runs") {
    auto owner = Owner{};
    auto graph = owner.graphFor(lr::core::ViewId{0}, 0);
    auto const history = graph.history<Format>("accumulation", lr::core::ivec2{4, 4});
    auto &accumulate = graph.addPass("accumulate");
    accumulate.read(history.previous);
    accumulate.write(history.current);
    accumulate.record(noop);

    auto const compiled = std::move(graph).compile();

    CHECK(passNames(compiled) == std::vector<std::string>{"accumulate"});
}

TEST_CASE("a graph of no view has no history") {
    auto graph = GraphBuilder{};

    CHECK_THROWS_AS(graph.history<Format>("accumulation", lr::core::ivec2{4, 4}), std::logic_error);
}

TEST_CASE("the blackboard hands values on by type") {
    auto board = Blackboard{};
    board.put(42);

    CHECK(board.contains<int>());
    CHECK(board.get<int>() == 42);
    CHECK_THROWS_AS(board.get<float>(), std::logic_error);
}

} // namespace
