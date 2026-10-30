#include "FrameGraphImpl.h"

#include <doctest/doctest.h>

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

TEST_CASE("the blackboard hands values on by type") {
    auto board = Blackboard{};
    board.put(42);

    CHECK(board.contains<int>());
    CHECK(board.get<int>() == 42);
    CHECK_THROWS_AS(board.get<float>(), std::logic_error);
}

} // namespace
