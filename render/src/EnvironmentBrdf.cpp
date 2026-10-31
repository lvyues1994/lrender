#include "EnvironmentBrdf.h"

#include <lr/render/GpuTypes.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lr::render {

namespace {

constexpr auto kSamples = 2048u;
constexpr double kPi = 3.14159265358979323846;

double radicalInverse(std::uint32_t bits) noexcept {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<double>(bits) / 4294967296.0;
}

// The integral over the hemisphere of reflectedLight()'s specular term (Lighting.h) with
// F0 = 0 and the Schlick factor split off, by importance sampling D(h) (n . h) on a
// Hammersley set: the weight of a direction is D V (n . l) / pdf = 4 V (n . l)(v . h) / (n . h).
core::vec2 integrate(double const nDotV, double const roughness) {
    auto const alpha = std::max(roughness * roughness, 0.001);
    auto const alpha2 = alpha * alpha;
    auto const nv = std::max(nDotV, 0.0001);
    auto const vx = std::sqrt(std::max(1.0 - nDotV * nDotV, 0.0));
    auto const vz = nDotV;
    auto scale = 0.0;
    auto bias = 0.0;
    for (auto sample = 0u; sample < kSamples; ++sample) {
        auto const u1 = (sample + 0.5) / kSamples;
        auto const u2 = radicalInverse(sample);
        auto const cos2 = (1.0 - u1) / (1.0 + (alpha2 - 1.0) * u1);
        auto const nh = std::sqrt(cos2);
        auto const sinTheta = std::sqrt(std::max(1.0 - cos2, 0.0));
        auto const hx = sinTheta * std::cos(2.0 * kPi * u2);
        auto const hz = nh;
        auto const vh = vx * hx + vz * hz;
        auto const nl = 2.0 * vh * hz - vz;
        if (nl <= 0.0 or vh <= 0.0) {
            continue;
        }
        auto const visibility =
            0.5 / (nl * std::sqrt(nv * nv * (1.0 - alpha2) + alpha2) + nv * std::sqrt(nl * nl * (1.0 - alpha2) + alpha2));
        auto const weight = 4.0 * visibility * nl * vh / nh;
        auto const schlick = std::pow(1.0 - std::clamp(vh, 0.0, 1.0), 5.0);
        scale += weight * (1.0 - schlick);
        bias += weight * schlick;
    }
    return core::vec2{static_cast<float>(scale / kSamples), static_cast<float>(bias / kSamples)};
}

std::vector<core::vec2> buildTable() {
    auto const size = environmentBrdfSize();
    auto table = std::vector<core::vec2>(static_cast<std::size_t>(size) * size);
    for (auto row = 0u; row < size; ++row) {
        for (auto column = 0u; column < size; ++column) {
            table[static_cast<std::size_t>(row) * size + column] =
                integrate(static_cast<double>(column) / (size - 1), static_cast<double>(row) / (size - 1));
        }
    }
    return table;
}

} // namespace

std::vector<core::vec2> const &environmentBrdfTable() {
    static auto const table = buildTable();
    return table;
}

core::vec2 environmentBrdf(float const nDotV, float const roughness) {
    auto const &table = environmentBrdfTable();
    auto const last = static_cast<float>(environmentBrdfSize() - 1);
    auto const x = std::clamp(nDotV, 0.0f, 1.0f) * last;
    auto const y = std::clamp(roughness, 0.0f, 1.0f) * last;
    auto const x0 = std::min(static_cast<std::uint32_t>(x), environmentBrdfSize() - 2);
    auto const y0 = std::min(static_cast<std::uint32_t>(y), environmentBrdfSize() - 2);
    auto const fx = x - static_cast<float>(x0);
    auto const fy = y - static_cast<float>(y0);
    auto const at = [&table](std::uint32_t const column, std::uint32_t const row) {
        return table[static_cast<std::size_t>(row) * environmentBrdfSize() + column];
    };
    auto const top = at(x0, y0) * (1.0f - fx) + at(x0 + 1, y0) * fx;
    auto const bottom = at(x0, y0 + 1) * (1.0f - fx) + at(x0 + 1, y0 + 1) * fx;
    return top * (1.0f - fy) + bottom * fy;
}

} // namespace lr::render
