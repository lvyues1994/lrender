#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace lr::core {

// A handle chosen by whoever creates the object; ids of different kinds do not mix.
template <class Tag>
struct Id {
    static constexpr std::uint32_t kInvalid = 0xFFFFFFFFu;

    constexpr bool isValid() const noexcept { return value != kInvalid; }

    friend constexpr bool operator==(Id const a, Id const b) noexcept { return a.value == b.value; }
    friend constexpr bool operator!=(Id const a, Id const b) noexcept { return a.value != b.value; }
    friend constexpr bool operator<(Id const a, Id const b) noexcept { return a.value < b.value; }

    std::uint32_t value = kInvalid;
};

using MeshId = Id<struct MeshTag>;
using TextureId = Id<struct TextureTag>;
using MaterialId = Id<struct MaterialTag>;
using InstanceId = Id<struct InstanceTag>;
using LightId = Id<struct LightTag>;
using ViewId = Id<struct ViewTag>;
using RenderTargetId = Id<struct RenderTargetTag>;
using PipelineId = Id<struct PipelineTag>;
// Given out by the renderer that shades with the model; 0 is its built-in model.
using MaterialModelId = Id<struct MaterialModelTag>;

// An entity of lr::scene's World. Its slot is reused once the entity is destroyed; the
// generation tells the old id from the new entity's.
struct EntityId {
    static constexpr std::uint32_t kInvalidIndex = 0xFFFFFFFFu;

    constexpr bool isValid() const noexcept { return index != kInvalidIndex; }

    friend constexpr bool operator==(EntityId const a, EntityId const b) noexcept {
        return a.index == b.index and a.generation == b.generation;
    }
    friend constexpr bool operator!=(EntityId const a, EntityId const b) noexcept { return not(a == b); }
    friend constexpr bool operator<(EntityId const a, EntityId const b) noexcept {
        return a.index != b.index ? a.index < b.index : a.generation < b.generation;
    }

    std::uint32_t index = kInvalidIndex;
    std::uint32_t generation = 0;
};

} // namespace lr::core

template <class Tag>
struct std::hash<lr::core::Id<Tag>> {
    std::size_t operator()(lr::core::Id<Tag> const id) const noexcept { return std::hash<std::uint32_t>{}(id.value); }
};

template <>
struct std::hash<lr::core::EntityId> {
    std::size_t operator()(lr::core::EntityId const id) const noexcept {
        return std::hash<std::uint64_t>{}(std::uint64_t{id.generation} << 32u | id.index);
    }
};
