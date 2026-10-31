#include "Hierarchy.h"

#include <lexec/execution.hpp>

#include <algorithm>
#include <cassert>

namespace lr::scene {

namespace {

// Levels smaller than this are not worth handing to the frame pool.
constexpr auto kSmallestParallelLevel = std::size_t{256};

// The one way a world transform is computed, so that every path gives the same bits.
core::mat4 composed(core::mat4 const *const parentWorld, core::Transform const &local) noexcept {
    auto const matrix = core::matrixOf(local);
    return parentWorld == nullptr ? matrix : *parentWorld * matrix;
}

} // namespace

std::uint32_t Hierarchy::create(core::Transform const &local, std::uint32_t const parent) {
    auto entity = kNoEntity;
    if (freeSlots.empty()) {
        entity = static_cast<std::uint32_t>(alive.size());
        generations.push_back(0);
        alive.push_back(0);
        parents.push_back(kNoEntity);
        firstChildren.push_back(kNoEntity);
        nextSiblings.push_back(kNoEntity);
        previousSiblings.push_back(kNoEntity);
        locals.emplace_back();
        worlds.emplace_back(1.0f);
        stale.push_back(0);
    } else {
        entity = freeSlots.back();
        freeSlots.pop_back();
    }
    alive[entity] = 1;
    locals[entity] = local;
    worlds[entity] = core::mat4{1.0f};
    link(entity, parent);
    markStale(entity, false);
    ++count;
    return entity;
}

std::vector<std::uint32_t> Hierarchy::destroy(std::uint32_t const entity) {
    unlink(entity);
    auto subtree = std::vector<std::uint32_t>{entity};
    for (auto index = std::size_t{0}; index < subtree.size(); ++index) {
        for (auto child = firstChildren[subtree[index]]; child != kNoEntity; child = nextSiblings[child]) {
            subtree.push_back(child);
        }
    }
    for (auto const freed : subtree) {
        alive[freed] = 0;
        ++generations[freed];
        stale[freed] = 0;
        parents[freed] = kNoEntity;
        firstChildren[freed] = kNoEntity;
        nextSiblings[freed] = kNoEntity;
        previousSiblings[freed] = kNoEntity;
        freeSlots.push_back(freed);
    }
    count -= subtree.size();
    return subtree;
}

bool Hierarchy::isAlive(core::EntityId const id) const noexcept {
    return id.index < alive.size() and alive[id.index] != 0 and generations[id.index] == id.generation;
}

core::EntityId Hierarchy::idOf(std::uint32_t const entity) const noexcept {
    return core::EntityId{entity, generations[entity]};
}

bool Hierarchy::isAncestor(std::uint32_t const ancestor, std::uint32_t const entity) const noexcept {
    for (auto current = entity; current != kNoEntity; current = parents[current]) {
        if (current == ancestor) {
            return true;
        }
    }
    return false;
}

void Hierarchy::setParent(std::uint32_t const entity, std::uint32_t const parent) {
    unlink(entity);
    link(entity, parent);
    markStale(entity, true);
}

void Hierarchy::setLocal(std::uint32_t const entity, core::Transform const &local) {
    locals[entity] = local;
    markStale(entity, false);
}

core::mat4 Hierarchy::worldOf(std::uint32_t const entity) const {
    if (stale[entity] == 0) {
        return worlds[entity];
    }
    // Up to the first fresh ancestor, then down again as propagate() would go.
    auto chain = std::vector<std::uint32_t>{entity};
    while (parents[chain.back()] != kNoEntity and stale[parents[chain.back()]] != 0) {
        chain.push_back(parents[chain.back()]);
    }
    auto const top = parents[chain.back()];
    auto world = composed(top == kNoEntity ? nullptr : &worlds[top], locals[chain.back()]);
    for (auto index = chain.size() - 1; index > 0; --index) {
        world = composed(&world, locals[chain[index - 1]]);
    }
    return world;
}

Propagation Hierarchy::propagate(PropagationConfig const &config, std::vector<std::uint32_t> &moved) {
    collectStaleSubtrees();
    movedAt.assign(order.size(), 0);
    auto const parallel = config.runtime != nullptr and not order.empty() and order.size() >= config.parallelTransforms;
    auto usedPool = false;
    auto begin = std::size_t{0};
    for (auto const end : levelEnds) {
        if (parallel and end - begin >= kSmallestParallelLevel) {
            // Each entity of a level reads its parent, of the level before, and writes only itself.
            auto const done = lexec::sync_wait(lexec::schedule(config.runtime->framePool())
                                               | lexec::bulk(lexec::par, end - begin, [this, begin](std::size_t const index) noexcept {
                                                     update(begin + index);
                                                 }));
            assert(done.has_value());
            static_cast<void>(done);
            usedPool = true;
        } else {
            for (auto position = begin; position < end; ++position) {
                update(position);
            }
        }
        begin = end;
    }
    for (auto position = std::size_t{0}; position < order.size(); ++position) {
        if (movedAt[position] != 0) {
            moved.push_back(order[position]);
        }
    }
    return Propagation{order.size(), usedPool};
}

void Hierarchy::link(std::uint32_t const entity, std::uint32_t const parent) noexcept {
    parents[entity] = parent;
    previousSiblings[entity] = kNoEntity;
    nextSiblings[entity] = kNoEntity;
    if (parent == kNoEntity) {
        return;
    }
    auto const first = firstChildren[parent];
    nextSiblings[entity] = first;
    if (first != kNoEntity) {
        previousSiblings[first] = entity;
    }
    firstChildren[parent] = entity;
}

void Hierarchy::unlink(std::uint32_t const entity) noexcept {
    auto const parent = parents[entity];
    if (parent == kNoEntity) {
        return;
    }
    auto const previous = previousSiblings[entity];
    auto const next = nextSiblings[entity];
    if (previous == kNoEntity) {
        firstChildren[parent] = next;
    } else {
        nextSiblings[previous] = next;
    }
    if (next != kNoEntity) {
        previousSiblings[next] = previous;
    }
    parents[entity] = kNoEntity;
    previousSiblings[entity] = kNoEntity;
    nextSiblings[entity] = kNoEntity;
}

void Hierarchy::markStale(std::uint32_t const entity, bool const force) {
    if (stale[entity] != 0) {
        if (force) {
            staleRoots.push_back(entity);
        }
        return;
    }
    staleRoots.push_back(entity);
    stale[entity] = 1;
    auto pending = std::vector<std::uint32_t>{};
    for (auto child = firstChildren[entity]; child != kNoEntity; child = nextSiblings[child]) {
        pending.push_back(child);
    }
    while (not pending.empty()) {
        auto const next = pending.back();
        pending.pop_back();
        if (stale[next] != 0) {
            continue; // its subtree is stale already
        }
        stale[next] = 1;
        for (auto child = firstChildren[next]; child != kNoEntity; child = nextSiblings[child]) {
            pending.push_back(child);
        }
    }
}

void Hierarchy::collectStaleSubtrees() {
    order.clear();
    levelEnds.clear();
    for (auto const root : staleRoots) {
        auto const parent = parents[root];
        if (alive[root] != 0 and stale[root] != 0 and (parent == kNoEntity or stale[parent] == 0)) {
            order.push_back(root);
        }
    }
    staleRoots.clear();
    std::sort(order.begin(), order.end());
    order.erase(std::unique(order.begin(), order.end()), order.end());
    auto begin = std::size_t{0};
    while (begin < order.size()) {
        auto const end = order.size();
        levelEnds.push_back(end);
        for (auto position = begin; position < end; ++position) {
            for (auto child = firstChildren[order[position]]; child != kNoEntity; child = nextSiblings[child]) {
                order.push_back(child);
            }
        }
        begin = end;
    }
}

void Hierarchy::update(std::size_t const position) noexcept {
    auto const entity = order[position];
    auto const parent = parents[entity];
    auto const world = composed(parent == kNoEntity ? nullptr : &worlds[parent], locals[entity]);
    movedAt[position] = world == worlds[entity] ? 0 : 1;
    worlds[entity] = world;
    stale[entity] = 0;
}

} // namespace lr::scene
