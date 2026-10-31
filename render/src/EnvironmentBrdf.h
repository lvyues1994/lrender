#pragma once

#include <lr/core/Math.h>

#include <vector>

namespace lr::render {

// What the shading kernels' GGX lobe with Schlick's Fresnel reflects of a uniform
// environment of radiance 1: F0 A + B. (A, B) at the centres of environmentBrdfSize()^2
// texels, row r and column c at roughness r / (size - 1) and n . v = c / (size - 1),
// integrated by importance sampling the lobe. Computed once per process.
std::vector<core::vec2> const &environmentBrdfTable();

// (A, B) as the kernels read the table: bilinear between texel centres.
core::vec2 environmentBrdf(float nDotV, float roughness);

} // namespace lr::render
