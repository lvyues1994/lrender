#include <lr/core/Mesh.h>

#include <cmath>

namespace lr::core {

Aabb boundsOf(MeshData const &mesh) noexcept {
    if (mesh.positions.empty()) {
        return Aabb{};
    }
    auto result = Aabb{vec3{INFINITY}, vec3{-INFINITY}};
    for (auto const &position : mesh.positions) {
        result.min = tc::min(result.min, position);
        result.max = tc::max(result.max, position);
    }
    return result;
}

} // namespace lr::core
