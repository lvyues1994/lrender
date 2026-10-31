#pragma once

// Temporal anti-aliasing. The visibility pipeline renders each frame with its projection
// moved by a different sub-pixel offset; per pixel, the motion of the surface it shows
// since the previous frame finds where that surface was in the history, which is sampled
// there, clamped to the colours around the pixel in this frame (in YCoCg) and blended
// with this frame's colour.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/Material.h>
#include <lr/render/kernels/VisibleInstances.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

// The history averages its first frames evenly, up to this many; from then on the newest
// frame weighs 1 / temporalFrames() and older ones fade out exponentially.
inline float temporalFrames() { return 10.0f; }
// Screen positions of points behind the camera, which the history does not show.
inline vec2 unseenOnScreen() { return vec2(-1e9f); }

inline vec3 toYCoCg(vec3 rgb) {
    return vec3{0.25f * rgb.x + 0.5f * rgb.y + 0.25f * rgb.z, 0.5f * rgb.x - 0.5f * rgb.z,
                -0.25f * rgb.x + 0.5f * rgb.y - 0.25f * rgb.z};
}

inline vec3 fromYCoCg(vec3 ycocg) {
    return vec3{ycocg.x + ycocg.y - ycocg.z, ycocg.x + ycocg.z, ycocg.x - ycocg.y - ycocg.z};
}

// Pixels across and rows down where `viewProjection` puts `point`.
inline vec2 screenOf(mat4 viewProjection, vec3 point, ivec2 size) {
    vec4 clip = viewProjection * vec4(point, 1.0f);
    if (clip.w <= 0.0f) {
        return unseenOnScreen();
    }
    vec2 ndc = clip.xy() / clip.w;
    return vec2{(ndc.x * 0.5f + 0.5f) * float(size.x), (0.5f - ndc.y * 0.5f) * float(size.y)};
}

// motion[pixel]: where the surface the pixel shows is on the screen less where it was in
// the previous frame, both without the frames' jitters, in pixels. The point is found in
// the instance's object space and placed with each frame's transform, so a surface that
// did not move under a camera that did not move gets exactly 0. Pixels no triangle covers
// follow the far plane.
struct TC_KERNEL TemporalMotion {
    static constexpr char fileLocation[] = "lr_temporal_motion";
    uvec3 local_size{8, 8, 1};

    InstanceTable<0> instances;
    VertexTable<1> vertices;
    BufferBinding<uint, 2> indices;
    BufferBinding<uvec4, 3> visible;
    BufferBinding<uint, 4> triangles;
    // [0] the inverse of this frame's jittered view-projection, [1] this frame's
    // view-projection and [2] the previous frame's, both without jitter.
    BufferBinding<mat4, 5> matrices;
    BufferBinding<vec2, 6> motion;
    Uniform<ivec2, 0> size{};

    LR_FIND_VISIBLE_SLOT

    vec3 worldPoint(vec2 ndc, float depth) {
        vec4 point = matrices[0] * vec4(ndc, depth, 1.0f);
        return point.xyz() / point.w;
    }

    // Where the camera ray `origin` + t `direction` meets visible triangle `triangle`, in
    // its instance's object space.
    vec3 objectPointOn(uint triangle, vec3 origin, vec3 direction, mat4 objectToWorld) {
        uvec4 entry = visible[1u + findSlot(triangle)];
        uint first = entry.z + 3u * (triangle - entry.y);
        std::array<vec3, 3> corners;
        for (uint k = 0u; k < 3u; ++k) {
            corners[k] = vertices[entry.w + indices[first + k]].position;
        }
        vec3 a = (objectToWorld * vec4(corners[0], 1.0f)).xyz();
        vec3 b = (objectToWorld * vec4(corners[1], 1.0f)).xyz();
        vec3 c = (objectToWorld * vec4(corners[2], 1.0f)).xyz();
        vec3 weights = planeWeights(origin, direction, a, b - a, c - a);
        if (weights == vec3(0.0f)) {
            weights = vec3(1.0f / 3.0f);
        } else {
            weights = clamp(weights, vec3(0.0f), vec3(1.0f));
            weights = weights / (weights.x + weights.y + weights.z);
        }
        return corners[0] * weights.x + corners[1] * weights.y + corners[2] * weights.z;
    }

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint pixelIndex = uint(pixel.y * size.value.x + pixel.x);
        vec2 ndc = vec2{(float(pixel.x) + 0.5f) / float(size.value.x) * 2.0f - 1.0f,
                        1.0f - (float(pixel.y) + 0.5f) / float(size.value.y) * 2.0f};
        vec3 origin = worldPoint(ndc, 0.0f);
        vec3 far = worldPoint(ndc, 1.0f);
        vec3 now = far;
        vec3 before = far;
        uint triangle = triangles[pixelIndex];
        if (triangle != noTriangle()) {
            uint instance = visible[1u + findSlot(triangle)].x;
            mat4 objectToWorld = instances[instance].objectToWorld.value;
            mat4 previousObjectToWorld = instances[instance].previousObjectToWorld.value;
            vec3 point = objectPointOn(triangle, origin, normalize(far - origin), objectToWorld);
            now = (objectToWorld * vec4(point, 1.0f)).xyz();
            before = (previousObjectToWorld * vec4(point, 1.0f)).xyz();
        }
        motion[pixelIndex] = screenOf(matrices[1], now, size.value) - screenOf(matrices[2], before, size.value);
    }
};

// resolved[pixel]: this frame's colour blended with the history where the pixel's
// surface was, and in w the frames the history now averages. The history is sampled
// bilinearly and clamped to the box around the colours of the 3x3 pixels around this one,
// in YCoCg, so that what the pixel no longer shows does not linger. Where the history is
// invalid or the surface was off the screen, the pixel starts over.
struct TC_KERNEL TemporalResolve {
    static constexpr char fileLocation[] = "lr_temporal_resolve";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 1> resolved;
    TextureBinding<HdrFormat, Dim::D2, HdrPixel, 2, Filter::Linear, Wrap::ClampToEdge, Mip::None> history;
    BufferBinding<vec2, 3> motion;
    Uniform<ivec2, 0> size{};
    Uniform<uint, 1> historyValid{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        vec3 current = imageLoad(color, pixel).xyz();
        vec3 low = toYCoCg(current);
        vec3 high = low;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                ivec2 around = clamp(pixel + ivec2{dx, dy}, ivec2(0), size.value - ivec2(1));
                vec3 neighbour = toYCoCg(imageLoad(color, around).xyz());
                low = min(low, neighbour);
                high = max(high, neighbour);
            }
        }
        vec2 extent = vec2(size.value);
        vec2 at = vec2(pixel) + vec2(0.5f) - motion[uint(pixel.y * size.value.x + pixel.x)];
        vec3 result = current;
        float frames = 1.0f;
        if (historyValid.value != 0u and at.x >= 0.0f and at.y >= 0.0f and at.x <= extent.x and at.y <= extent.y) {
            vec4 past = textureLod(history, at / extent, 0.0f);
            vec3 clamped = fromYCoCg(clamp(toYCoCg(past.xyz()), low, high));
            frames = min(past.w + 1.0f, temporalFrames());
            result = mix(clamped, current, 1.0f / frames);
        }
        imageStore(resolved, pixel, vec4(result, frames));
    }
};

// The resolved colour, opaque, into the scene colour.
struct TC_KERNEL ShowResolved {
    static constexpr char fileLocation[] = "lr_show_resolved";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> resolved;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 1> color;
    Uniform<ivec2, 0> size{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x < size.value.x and pixel.y < size.value.y) {
            imageStore(color, pixel, vec4(imageLoad(resolved, pixel).xyz(), 1.0f));
        }
    }
};

} // namespace lr::render::kernels
