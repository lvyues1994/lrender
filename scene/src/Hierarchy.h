#pragma once

#include <lr/core/Id.h>
#include <lr/core/Math.h>
#include <lr/runtime/Runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lr::scene {

inline constexpr std::uint32_t kNoEntity = core::EntityId::kInvalidIndex;

struct PropagationConfig {
    runtime::Runtime *runtime = nullptr;
    std::size_t parallelTransforms = 0;
};

struct Propagation {
    std::size_t transforms = 0; // world transforms computed
    bool parallel = false;      // some went to the frame pool
};

// Entity slots, their parents and transforms, indexed by slot. An entity's world transform
// is stale once its own or an ancestor's local transform or parent changed; all of a stale
// entity's descendants are stale too, so the stale entities form whole subtrees, which
// propagate() computes level by level from their roots.
struct Hierarchy {
    // `parent` is kNoEntity for a root.
    std::uint32_t create(core::Transform const &local, std::uint32_t parent);
    // Frees the slots of the entity and its descendants, which it returns, each before its children.
    std::vector<std::uint32_t> destroy(std::uint32_t entity);

    bool isAlive(core::EntityId id) const noexcept;
    core::EntityId idOf(std::uint32_t entity) const noexcept;
    std::size_t size() const noexcept { return count; }
    std::size_t slotCount() const noexcept { return alive.size(); }

    std::uint32_t parentOf(std::uint32_t entity) const noexcept { return parents[entity]; }
    // Whether `ancestor` is `entity` or one of its ancestors.
    bool isAncestor(std::uint32_t ancestor, std::uint32_t entity) const noexcept;
    void setParent(std::uint32_t entity, std::uint32_t parent);
    core::Transform const &localOf(std::uint32_t entity) const noexcept { return locals[entity]; }
    void setLocal(std::uint32_t entity, core::Transform const &local);
    // Computed on the way while stale; the same as propagate() gives.
    core::mat4 worldOf(std::uint32_t entity) const;

    // Brings every stale world transform up to date and appends the entities whose world
    // transform changed to `moved`.
    Propagation propagate(PropagationConfig const &config, std::vector<std::uint32_t> &moved);

private:
    void link(std::uint32_t entity, std::uint32_t parent) noexcept;
    void unlink(std::uint32_t entity) noexcept;
    // `force`: record the entity as a root to propagate from even if it was stale already,
    // as a reparented entity may have left the subtree it was stale in.
    void markStale(std::uint32_t entity, bool force);
    void collectStaleSubtrees();
    void update(std::size_t position) noexcept;

    std::vector<std::uint32_t> generations;
    std::vector<std::uint8_t> alive;
    std::vector<std::uint32_t> freeSlots;
    std::vector<std::uint32_t> parents;
    std::vector<std::uint32_t> firstChildren;
    std::vector<std::uint32_t> nextSiblings;
    std::vector<std::uint32_t> previousSiblings;
    std::vector<core::Transform> locals;
    std::vector<core::mat4> worlds;
    std::vector<std::uint8_t> stale;
    std::vector<std::uint32_t> staleRoots; // where subtrees went stale since the last propagation
    std::size_t count = 0;

    // propagate()'s: the stale entities level by level, where each level ends, and which moved.
    std::vector<std::uint32_t> order;
    std::vector<std::size_t> levelEnds;
    std::vector<std::uint8_t> movedAt;
};

} // namespace lr::scene
