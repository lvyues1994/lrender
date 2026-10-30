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

} // namespace lr::core

template <class Tag>
struct std::hash<lr::core::Id<Tag>> {
    std::size_t operator()(lr::core::Id<Tag> const id) const noexcept { return std::hash<std::uint32_t>{}(id.value); }
};
