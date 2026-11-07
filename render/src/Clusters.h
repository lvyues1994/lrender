#pragma once

#include <lr/core/Mesh.h>
#include <lr/render/GpuTypes.h>

#include <vector>

namespace lr::render {

// The mesh's triangles cut into clusters in index order: each takes the next triangles
// while they number at most clusterMaxTriangles() over at most clusterMaxVertices()
// distinct vertices, so the triangles keep their numbers. The clusters are as compact in
// space as the order is; meshes usually come ordered for a vertex cache, which is local.
std::vector<GpuCluster> buildClusters(core::MeshData const &mesh);

} // namespace lr::render
