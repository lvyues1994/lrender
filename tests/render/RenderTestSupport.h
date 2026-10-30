#pragma once

#include <lr/core/Primitives.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace lr::test {

// A renderer with its own small runtime.
struct RenderHarness {
    explicit RenderHarness(std::string const &backend = "cpu", std::string const &device = "") {
        auto runtimeConfig = runtime::RuntimeConfig{};
        runtimeConfig.framePoolThreads = 2;
        runtimeInstance = runtime::makeRuntime(runtimeConfig);
        auto config = render::RendererConfig{};
        config.backend = backend;
        config.deviceNameContains = device;
        config.runtime = runtimeInstance.get();
        renderer = render::makeRenderer(config);
    }

    // Throws what the frame failed with.
    render::FrameResult render(render::FrameDesc frame) {
        auto result = lexec::sync_wait(renderer->submit(std::move(frame)));
        if (not result) {
            throw std::runtime_error{"the frame was stopped"};
        }
        return std::get<0>(std::move(*result));
    }

    std::unique_ptr<runtime::Runtime> runtimeInstance;
    std::unique_ptr<render::Renderer> renderer;
};

inline render::ViewDesc viewOf(core::RenderTargetId const target, core::CameraDesc const &camera,
                               core::ViewId const id = core::ViewId{0}) {
    auto view = render::ViewDesc{};
    view.id = id;
    view.target = target;
    view.camera = camera;
    view.settings.toneMapping = render::ToneMapping::Clamp;
    return view;
}

inline core::CameraDesc cameraAt(core::vec3 const &position, core::vec3 const &target,
                                 core::vec3 const &up = core::vec3{0.0f, 1.0f, 0.0f}) {
    auto camera = core::CameraDesc{};
    camera.position = position;
    camera.target = target;
    camera.up = up;
    return camera;
}

inline core::MeshHandle share(core::MeshData mesh) { return std::make_shared<core::MeshData const>(std::move(mesh)); }

inline core::MaterialDesc emissive(core::vec3 const &radiance) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    material.emissive = radiance;
    return material;
}

inline core::mat4 translation(core::vec3 const &offset) {
    auto transform = core::Transform{};
    transform.translation = offset;
    return core::matrixOf(transform);
}

// What the tonemap pass writes for a linear value with ToneMapping::Clamp and exposure 1.
inline std::uint32_t srgb8(float const linear) {
    auto const value = std::clamp(linear, 0.0f, 1.0f);
    auto const encoded = value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint32_t>(std::lround(encoded * 255.0f));
}

inline std::uint32_t rgba8(core::vec3 const &linear) {
    return srgb8(linear.x) | srgb8(linear.y) << 8u | srgb8(linear.z) << 16u | 255u << 24u;
}

inline std::uint32_t channel(std::uint32_t const pixel, std::uint32_t const index) {
    return (pixel >> (8u * index)) & 0xFFu;
}

inline std::uint32_t pixelAt(render::ImageRgba8 const &image, int const x, int const y) {
    return image.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(image.extent.x)
                        + static_cast<std::size_t>(x)];
}

// The largest difference of any channel of any pixel.
inline std::uint32_t maxChannelDifference(render::ImageRgba8 const &a, render::ImageRgba8 const &b) {
    auto largest = std::uint32_t{0};
    for (auto index = std::size_t{0}; index < a.pixels.size(); ++index) {
        for (auto c = 0u; c < 4u; ++c) {
            auto const x = channel(a.pixels[index], c);
            auto const y = channel(b.pixels[index], c);
            largest = std::max(largest, x > y ? x - y : y - x);
        }
    }
    return largest;
}

} // namespace lr::test
