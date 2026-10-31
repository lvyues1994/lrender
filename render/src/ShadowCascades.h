#pragma once

#include <lr/core/Math.h>
#include <lr/core/Scene.h>
#include <lr/render/View.h>

#include <vector>

namespace lr::render {

struct ShadowCascade {
    core::mat4 viewProjection; // world to the clip space the shadow rasterizer draws
    core::mat4 worldToShadow;  // world to texture coordinates (rows down) and depth
    float texelSize = 0.0f;    // in world units
    float depthScale = 0.0f;   // depth per world unit along the light
    float farDistance = 0.0f;  // from the camera, where the cascade's slice of the view ends
};

struct CascadeRequest {
    core::CameraDesc camera;
    float aspect = 1.0f;
    core::vec3 lightDirection{0.0f, -1.0f, 0.0f}; // unit length, the way the light travels
    core::Aabb sceneBounds;                        // every caster lies within; min > max for none
    ShadowSettings settings;
};

// Orthographic cascades along the light, each around a bounding sphere of its slice of
// the view frustum: slices split the distance between practical (logarithmic and uniform,
// 3:1) split points, the sphere's size is fixed by the slice's shape alone and its centre
// snaps to whole texels, so the cascades only ever move by whole texels. Depth spans the
// scene's bounds along the light, so casters outside the slice still cast into it.
std::vector<ShadowCascade> fitShadowCascades(CascadeRequest const &request);

} // namespace lr::render
