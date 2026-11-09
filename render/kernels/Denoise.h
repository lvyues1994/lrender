#pragma once

// Denoising the path tracer's mean while the view moves, as spatiotemporal variance-guided
// filtering (Schied et al. 2017) does it: the illumination (the colour less the emission,
// over the albedo of what each pixel shows) is blended with where the pixel's surface was
// in the previous frame, its variance estimated from the moments of its luminance, and
// filtered by an a-trous wavelet whose weights stop at edges of depth, normal and
// luminance; the filtered illumination times the albedo, plus the emission, is the colour.
// The guides (normal, distance, albedo, emission) come from the visibility buffer.

#include <lr/render/GpuTypes.h>
#include <lr/render/kernels/SurfaceShading.h>

#include <array>

namespace lr::render::kernels {

using namespace tc;

// Octahedral encoding of a unit vector, in [-1, 1]^2.
inline vec2 octahedralOf(vec3 n) {
    vec3 a = n / (abs(n.x) + abs(n.y) + abs(n.z));
    if (a.z >= 0.0f) {
        return a.xy();
    }
    return vec2{(1.0f - abs(a.y)) * (a.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(a.x)) * (a.y >= 0.0f ? 1.0f : -1.0f)};
}

inline vec3 normalOf(vec2 e) {
    vec3 n = vec3{e.x, e.y, 1.0f - abs(e.x) - abs(e.y)};
    if (n.z < 0.0f) {
        n = vec3{(1.0f - abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f),
                 n.z};
    }
    return normalize(n);
}

inline float luminanceOf(vec3 c) { return dot(c, vec3{0.2126f, 0.7152f, 0.0722f}); }

// The guides' distance where no surface shows: the sky, which needs no denoising.
inline float noSurface() { return -1.0f; }

// Albedo below this is taken as this, so that dividing by it does not blow noise up.
inline float leastAlbedo() { return 0.02f; }

// Frames that stop the history of a pixel growing: the illumination and its moments keep
// at least 1/maxHistory() of each new frame. Alike, so that what a bright sample leaves
// in the illumination fades no slower than the variance it adds, which lets the filter
// spread it.
inline float maxHistory() { return 10.0f; }

// The guides of each pixel the visibility buffer covers: the octahedral shading normal in
// xy and the distance along the camera ray in z (noSurface() where nothing covers), the
// base colour as the albedo and the emission, as the path tracer samples the material at
// that point.
struct TC_KERNEL PathTraceGuides {
    static constexpr char fileLocation[] = "lr_path_trace_guides";
    uvec3 local_size{8, 8, 1};

    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 0> normalDepth;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 1> albedo;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 2> emission;
    LR_SURFACE_SHADING(3u)
    BufferBinding<uint, 23> triangles;

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint triangle = triangles[uint(pixel.y * size.value.x + pixel.x)];
        if (triangle == noTriangle()) {
            imageStore(normalDepth, pixel, vec4{0.0f, 0.0f, noSurface(), 0.0f});
            imageStore(albedo, pixel, vec4(1.0f));
            imageStore(emission, pixel, vec4(0.0f));
            return;
        }
        PixelRays rays = raysThrough(pixel);
        ShadingInputs inputs = surfaceAt(triangle, rays);
        float distance = length(inputs.position - rays.origin);
        imageStore(normalDepth, pixel, vec4(octahedralOf(inputs.normal), distance, 0.0f));
        imageStore(albedo, pixel, vec4(max(inputs.textures.baseColor.xyz(), vec3(leastAlbedo())), 1.0f));
        imageStore(emission, pixel, vec4(inputs.textures.emissive, 0.0f));
    }
};

// illumination[pixel] = (colour - emission) / albedo; on the sky, whose albedo is 1 and
// emission 0, the colour. A pixel that shows a surface and is brighter than every one
// around it that does is dimmed to the brightest of them, and one darker than all of them
// takes the darkest one's: a lone extreme sample, which the filter could not tell from a
// feature.
struct TC_KERNEL DemodulateIllumination {
    static constexpr char fileLocation[] = "lr_demodulate_illumination";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 1> albedo;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 2> emission;
    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 3> normalDepth;
    BufferBinding<vec4, 4> illumination;
    Uniform<ivec2, 0> size{};

    vec3 litAt(ivec2 at) {
        vec3 lit = (imageLoad(color, at).xyz() - imageLoad(emission, at).xyz()) / imageLoad(albedo, at).xyz();
        return max(lit, vec3(0.0f));
    }

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        vec3 lit = litAt(pixel);
        if (imageLoad(normalDepth, pixel).z >= 0.0f) {
            float brightest = -1.0f;
            vec3 darkest = vec3(-1.0f);
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    ivec2 at = pixel + ivec2{i, j};
                    if ((i == 0 and j == 0) or at.x < 0 or at.y < 0 or at.x >= size.value.x or at.y >= size.value.y
                        or imageLoad(normalDepth, at).z < 0.0f) {
                        continue;
                    }
                    vec3 other = litAt(at);
                    brightest = max(brightest, luminanceOf(other));
                    if (darkest.x < 0.0f or luminanceOf(other) < luminanceOf(darkest)) {
                        darkest = other;
                    }
                }
            }
            float luminance = luminanceOf(lit);
            if (brightest >= 0.0f and luminance > brightest) {
                lit = lit * (brightest / luminance);
            } else if (darkest.x >= 0.0f and luminance < luminanceOf(darkest)) {
                lit = darkest;
            }
        }
        illumination[uint(pixel.y * size.value.x + pixel.x)] = vec4(lit, 0.0f);
    }
};

// Each pixel's illumination blended with the history where its surface was in the
// previous frame (`motion` back from it), sampled bilinearly from the previous pixels that
// show a surface of about the same distance (within a tenth) and normal: color = (rgb,
// frames the history holds), moments = (luminance, its square). Without any such pixel, or
// without a valid history, the pixel starts over.
struct TC_KERNEL AccumulateIllumination {
    static constexpr char fileLocation[] = "lr_accumulate_illumination";
    uvec3 local_size{8, 8, 1};

    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 0> normalDepth;
    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 1> previousNormalDepth;
    BufferBinding<vec4, 2> illumination;
    BufferBinding<vec2, 3> motion;
    BufferBinding<vec4, 4> previousColor;
    BufferBinding<vec2, 5> previousMoments;
    BufferBinding<vec4, 6> color;
    BufferBinding<vec2, 7> moments;
    Uniform<ivec2, 0> size{};
    Uniform<uint, 1> historyValid{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint index = uint(pixel.y * size.value.x + pixel.x);
        vec3 lit = illumination[index].xyz();
        float luminance = luminanceOf(lit);
        vec4 guide = imageLoad(normalDepth, pixel);
        if (guide.z < 0.0f) {
            color[index] = vec4(lit, 1.0f);
            moments[index] = vec2{luminance, luminance * luminance};
            return;
        }
        vec3 normal = normalOf(guide.xy());
        vec2 before = vec2(pixel) + vec2(0.5f) - motion[index] - vec2(0.5f);
        ivec2 corner = ivec2(floor(before));
        vec2 fraction = before - vec2(corner);
        vec4 heldColor = vec4(0.0f);
        vec2 heldMoments = vec2(0.0f);
        float weights = 0.0f;
        if (historyValid.value != 0u) {
            for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                    ivec2 at = corner + ivec2{i, j};
                    if (at.x < 0 or at.y < 0 or at.x >= size.value.x or at.y >= size.value.y) {
                        continue;
                    }
                    vec4 previous = imageLoad(previousNormalDepth, at);
                    if (previous.z < 0.0f or abs(previous.z - guide.z) > 0.1f * guide.z
                        or dot(normalOf(previous.xy()), normal) < 0.9f) {
                        continue;
                    }
                    float weight = (i == 0 ? 1.0f - fraction.x : fraction.x) * (j == 0 ? 1.0f - fraction.y : fraction.y);
                    uint from = uint(at.y * size.value.x + at.x);
                    heldColor += previousColor[from] * weight;
                    heldMoments += previousMoments[from] * weight;
                    weights += weight;
                }
            }
        }
        if (weights < 0.01f) {
            color[index] = vec4(lit, 1.0f);
            moments[index] = vec2{luminance, luminance * luminance};
            return;
        }
        heldColor = heldColor / weights;
        heldMoments = heldMoments / weights;
        float frames = min(heldColor.w + 1.0f, 1000.0f);
        float alpha = max(1.0f / frames, 1.0f / maxHistory());
        color[index] = vec4(mix(heldColor.xyz(), lit, alpha), frames);
        moments[index] = mix(heldMoments, vec2{luminance, luminance * luminance}, alpha);
    }
};

// Whether two pixels' guides show about the same surface, as a weight in [0, 1]: by their
// distance, against how fast it changes across the screen at the first, `offset` pixels
// apart, and by their normals.
inline float guideWeight(vec4 centre, vec4 other, float gradient, float offset) {
    if (other.z < 0.0f) {
        return 0.0f;
    }
    float depth = exp(-abs(centre.z - other.z) / (gradient * offset + 1.0e-3f * centre.z + 1.0e-6f));
    float facing = max(dot(normalOf(centre.xy()), normalOf(other.xy())), 0.0f);
    float facing2 = facing * facing;
    float facing4 = facing2 * facing2;
    float facing16 = facing4 * facing4 * facing4 * facing4;
    float facing128 = facing16 * facing16 * facing16 * facing16 * facing16 * facing16 * facing16 * facing16;
    return depth * facing128;
}

// How fast the distance changes across the screen at `pixel`, from the pixels to its right
// and below that show a surface.
#define LR_DEPTH_GRADIENT                                                                                              \
    float depthAt(ivec2 at) { return imageLoad(normalDepth, clamp(at, ivec2(0), size.value - ivec2(1))).z; }         \
                                                                                                                       \
    float depthGradient(ivec2 pixel, float depth) {                                                                    \
        float right = depthAt(pixel + ivec2{1, 0});                                                                    \
        float down = depthAt(pixel + ivec2{0, 1});                                                                     \
        return max(right >= 0.0f ? abs(right - depth) : 0.0f, down >= 0.0f ? abs(down - depth) : 0.0f);                \
    }

// How many samples the history of `frames` frames averages as well as: all of them, until
// each new one keeps 1/maxHistory(), which averages as well as 2 maxHistory() - 1.
inline float historySamples(float frames) { return min(frames, 2.0f * maxHistory() - 1.0f); }

// filtered = (illumination, variance of it): the variance of one sample from the moments
// where the history holds four frames or more, else from the luminance of the 7x7 pixels
// around, weighted by their guides; over the samples the history averages.
struct TC_KERNEL EstimateVariance {
    static constexpr char fileLocation[] = "lr_estimate_variance";
    uvec3 local_size{8, 8, 1};

    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 0> normalDepth;
    BufferBinding<vec4, 1> color;
    BufferBinding<vec2, 2> moments;
    BufferBinding<vec4, 3> filtered;
    Uniform<ivec2, 0> size{};

    LR_DEPTH_GRADIENT

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint index = uint(pixel.y * size.value.x + pixel.x);
        vec4 held = color[index];
        vec4 guide = imageLoad(normalDepth, pixel);
        if (guide.z < 0.0f or held.w >= 4.0f) {
            vec2 moment = moments[index];
            float variance = max(moment.y - moment.x * moment.x, 0.0f) / historySamples(held.w);
            filtered[index] = vec4(held.xyz(), guide.z < 0.0f ? 0.0f : variance);
            return;
        }
        vec2 sum = vec2(0.0f);
        float weights = 0.0f;
        float gradient = depthGradient(pixel, guide.z);
        for (int j = -3; j <= 3; ++j) {
            for (int i = -3; i <= 3; ++i) {
                ivec2 at = clamp(pixel + ivec2{i, j}, ivec2(0), size.value - ivec2(1));
                float offset = length(vec2(ivec2{i, j}));
                float weight =
                    (i == 0 and j == 0) ? 1.0f : guideWeight(guide, imageLoad(normalDepth, at), gradient, offset);
                sum += moments[uint(at.y * size.value.x + at.x)] * weight;
                weights += weight;
            }
        }
        sum = sum / weights;
        // Few frames: the estimate is unsure, so it errs high.
        filtered[index] = vec4(held.xyz(), max(sum.y - sum.x * sum.x, 0.0f) * (4.0f / held.w) / held.w);
    }
};

// One a-trous iteration over (illumination, variance), the 5x5 B3 spline kernel `step`
// pixels apart, weighted by guideWeight() and by how far the luminance is from the centre's
// against its standard deviation (the 3x3 blur of the variance). With `feedback` the
// filtered illumination also goes into the history `color`, as the next frame's.
struct TC_KERNEL FilterIllumination {
    static constexpr char fileLocation[] = "lr_filter_illumination";
    uvec3 local_size{8, 8, 1};

    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 0> normalDepth;
    BufferBinding<vec4, 1> source;
    BufferBinding<vec4, 2> target;
    BufferBinding<vec4, 3> color;
    Uniform<ivec2, 0> size{};
    Uniform<int, 1> step{};
    Uniform<uint, 2> feedback{};

    LR_DEPTH_GRADIENT

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        uint index = uint(pixel.y * size.value.x + pixel.x);
        vec4 centre = source[index];
        vec4 guide = imageLoad(normalDepth, pixel);
        vec4 result = centre;
        if (guide.z >= 0.0f) {
            float gradient = depthGradient(pixel, guide.z);
            float blurred = 0.0f;
            std::array<float, 3> gauss{0.25f, 0.5f, 0.25f};
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    ivec2 at = clamp(pixel + ivec2{i, j}, ivec2(0), size.value - ivec2(1));
                    blurred += source[uint(at.y * size.value.x + at.x)].w * gauss[uint(i + 1)] * gauss[uint(j + 1)];
                }
            }
            float luminance = luminanceOf(centre.xyz());
            float spread = 4.0f * sqrt(max(blurred, 0.0f)) + 1.0e-4f;
            std::array<float, 5> kernel{1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};
            vec3 sum = vec3(0.0f);
            float variance = 0.0f;
            float weights = 0.0f;
            for (int j = -2; j <= 2; ++j) {
                for (int i = -2; i <= 2; ++i) {
                    ivec2 at = pixel + ivec2{i, j} * step.value;
                    if (at.x < 0 or at.y < 0 or at.x >= size.value.x or at.y >= size.value.y) {
                        continue;
                    }
                    vec4 other = source[uint(at.y * size.value.x + at.x)];
                    float offset = length(vec2(ivec2{i, j} * step.value));
                    float weight = (i == 0 and j == 0)
                                     ? 1.0f
                                     : guideWeight(guide, imageLoad(normalDepth, at), gradient, offset)
                                           * exp(-abs(luminanceOf(other.xyz()) - luminance) / spread);
                    weight *= kernel[uint(i + 2)] * kernel[uint(j + 2)];
                    sum += other.xyz() * weight;
                    variance += other.w * weight * weight;
                    weights += weight;
                }
            }
            result = vec4(sum / weights, variance / (weights * weights));
        }
        target[index] = result;
        if (feedback.value != 0u) {
            color[index] = vec4(result.xyz(), color[index].w);
        }
    }
};

// colour = albedo * filtered + emission, blended towards the path tracer's `mean` by
// `reference`; the sky, and every pixel once `reference` is 1, keep the colour the path
// tracer wrote.
struct TC_KERNEL RemodulateIllumination {
    static constexpr char fileLocation[] = "lr_remodulate_illumination";
    uvec3 local_size{8, 8, 1};

    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 0> color;
    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 1> mean;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 2> albedo;
    ImageBinding<HdrFormat, Dim::D2, HdrPixel, 3> emission;
    ImageBinding<AccumulationFormat, Dim::D2, AccumulationPixel, 4> normalDepth;
    BufferBinding<vec4, 5> filtered;
    Uniform<ivec2, 0> size{};
    Uniform<float, 1> reference{};

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        if (pixel.x >= size.value.x or pixel.y >= size.value.y) {
            return;
        }
        if (imageLoad(normalDepth, pixel).z < 0.0f or reference.value >= 1.0f) {
            return;
        }
        vec3 denoised = imageLoad(albedo, pixel).xyz() * filtered[uint(pixel.y * size.value.x + pixel.x)].xyz()
                      + imageLoad(emission, pixel).xyz();
        imageStore(color, pixel, vec4(mix(denoised, imageLoad(mean, pixel).xyz(), reference.value), 1.0f));
    }
};

} // namespace lr::render::kernels
