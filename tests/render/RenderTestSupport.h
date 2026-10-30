#pragma once

#include "TestKernels.h"

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
#include <vector>

namespace lr::test {

struct HarnessOptions {
    std::string backend = "cpu";
    std::string device;
    bool gpuTimers = false;
    std::uint32_t framePoolThreads = 2;
};

// A renderer with its own small runtime.
struct RenderHarness {
    explicit RenderHarness(HarnessOptions const &options) {
        auto runtimeConfig = runtime::RuntimeConfig{};
        runtimeConfig.framePoolThreads = options.framePoolThreads;
        runtimeInstance = runtime::makeRuntime(runtimeConfig);
        auto config = render::RendererConfig{};
        config.backend = options.backend;
        config.deviceNameContains = options.device;
        config.runtime = runtimeInstance.get();
        config.gpuTimers = options.gpuTimers;
        renderer = render::makeRenderer(config);
    }

    explicit RenderHarness(std::string const &backend = "cpu", std::string const &device = "")
        : RenderHarness(HarnessOptions{backend, device, false, 2}) {}

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

inline std::uint32_t channelDifference(std::uint32_t const a, std::uint32_t const b, std::uint32_t const index) {
    auto const x = channel(a, index);
    auto const y = channel(b, index);
    return x > y ? x - y : y - x;
}

// The largest difference of any channel of any pixel.
inline std::uint32_t maxChannelDifference(render::ImageRgba8 const &a, render::ImageRgba8 const &b) {
    auto largest = std::uint32_t{0};
    for (auto index = std::size_t{0}; index < a.pixels.size(); ++index) {
        for (auto c = 0u; c < 4u; ++c) {
            largest = std::max(largest, channelDifference(a.pixels[index], b.pixels[index], c));
        }
    }
    return largest;
}

// A post-processing feature that reads the view's HDR colour back, for tests that need
// more than the 8 bits of the output.
struct HdrProbe final : render::RenderFeature {
    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto const count = static_cast<std::size_t>(view.extent.x) * static_cast<std::size_t>(view.extent.y);
        auto const target = graph.createReadbackBuffer<core::vec4>("hdr probe", count);
        auto &pass = graph.addPass("read hdr");
        pass.read(view.sceneColor);
        pass.write(target);
        pass.record([color = view.sceneColor, target, extent = view.extent](render::PassContext &context) {
            auto &device = context.device();
            auto kernel = ReadHdr{};
            device.useKernel(kernel);
            kernel.color.attach(context.image(color));
            device.bindImage(kernel.color);
            kernel.pixels.attach(context.buffer(target));
            device.bindBuffer(kernel.pixels);
            kernel.size = extent;
            device.setUniform(kernel.size);
            context.dispatch(kernel, tc::uvec3{static_cast<tc::uint>((extent.x + 7) / 8),
                                               static_cast<tc::uint>((extent.y + 7) / 8), 1});
        });
        graph.readback<core::vec4>(target, [sink = pixels](core::vec4 const *data, std::size_t const size,
                                                           render::ViewResult &) { sink->assign(data, data + size); });
    }

    // The last frame's, rows from the top; written on the GPU thread before the frame's
    // sender completes.
    std::shared_ptr<std::vector<core::vec4>> pixels = std::make_shared<std::vector<core::vec4>>();
};

} // namespace lr::test
