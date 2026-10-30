#pragma once

#include <lr/core/Math.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace lr::core {

// A triangle list. Front faces wind counter-clockwise, as in glTF.
struct MeshData {
    std::vector<vec3> positions;
    std::vector<vec3> normals; // one per position
    std::vector<vec2> uvs;     // one per position, or none
    std::vector<std::uint32_t> indices;
};

using MeshHandle = std::shared_ptr<MeshData const>;

Aabb boundsOf(MeshData const &mesh) noexcept;

} // namespace lr::core
