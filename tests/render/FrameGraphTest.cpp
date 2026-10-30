#include "FrameGraphImpl.h"

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
