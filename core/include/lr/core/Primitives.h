#pragma once

#include <lr/core/Math.h>
#include <lr/core/Mesh.h>

#include <cstdint>

namespace lr::core {

// Centred on the origin, flat-shaded.
MeshData makeBox(vec3 const &halfExtents);
// Smooth-shaded, `segments` around the y axis and `rings` from pole to pole.
MeshData makeSphere(float radius, std::uint32_t segments, std::uint32_t rings);
// In the xz plane, facing +y.
MeshData makePlane(vec2 const &halfExtents);
// In the xy plane, facing +z.
MeshData makeQuad(vec2 const &halfExtents);

} // namespace lr::core
