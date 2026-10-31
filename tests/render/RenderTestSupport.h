#pragma once

#include "TestKernels.h"

#include <lr/core/Primitives.h>
#include <lr/render/Features.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
    std::uint32_t framesInFlight = 2;
    std::size_t textureStreamingBytes = std::size_t{8} << 20;
    std::uint32_t maxTextureSize = 2048;
    render::VariantUse subgroups = render::VariantUse::Auto;
    render::VariantUse singlePassVisibility = render::VariantUse::Auto;
    render::VariantUse rayQueries = render::VariantUse::Auto;
    // So that tests know which frame shows a converted texture.
    render::ConversionDelivery textureConversions = render::ConversionDelivery::Deterministic;
    bool aliasGraphResources = true;
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
        config.framesInFlight = options.framesInFlight;
        config.textureStreamingBytes = options.textureStreamingBytes;
        config.maxTextureSize = options.maxTextureSize;
        config.subgroups = options.subgroups;
        config.singlePassVisibility = options.singlePassVisibility;
        config.rayQueries = options.rayQueries;
        config.textureConversions = options.textureConversions;
        config.aliasGraphResources = options.aliasGraphResources;
        renderer = render::makeRenderer(config);
    }

    explicit RenderHarness(std::string const &backend = "cpu", std::string const &device = "")
        : RenderHarness(withBackend(backend, device)) {}

    static HarnessOptions withBackend(std::string const &backend, std::string const &device) {
        auto options = HarnessOptions{};
        options.backend = backend;
        options.device = device;
        return options;
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

inline std::uint32_t channelDifference(std::uint32_t const a, std::uint32_t const b, std::uint32_t const index) {
    auto const x = channel(a, index);
    auto const y = channel(b, index);
    return x > y ? x - y : y - x;
}

// Whether the colour channels of two pixels are at most `tolerance` apart.
inline bool nearPixel(std::uint32_t const a, std::uint32_t const b, std::uint32_t const tolerance = 1) {
    return channelDifference(a, b, 0) <= tolerance and channelDifference(a, b, 1) <= tolerance
       and channelDifference(a, b, 2) <= tolerance;
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

// Copies the first `count` elements of `source` into `sink` once the frame is done.
template <class T>
void readBack(render::FrameGraphBuilder &graph, std::string const &name, render::BufferRef<T> const source,
              std::size_t const count, std::shared_ptr<std::vector<T>> sink) {
    auto const target = graph.createReadbackBuffer<T>(name, count);
    auto &pass = graph.addPass("read " + name);
    pass.read(source);
    pass.write(target);
    pass.record([source, target, count](render::PassContext &context) {
        auto &device = context.device();
        auto kernel = CopyBuffer<T>{};
        device.useKernel(kernel);
        kernel.source.attach(context.buffer(source));
        device.bindBuffer(kernel.source);
        kernel.target.attach(context.buffer(target));
        device.bindBuffer(kernel.target);
        kernel.count = static_cast<tc::uint>(count);
        device.setUniform(kernel.count);
        context.dispatch(kernel, tc::uvec3{static_cast<tc::uint>((count + 63) / 64), 1, 1});
    });
    graph.readback<T>(target, [sink = std::move(sink)](T const *data, std::size_t const size, render::ViewResult &) {
        sink->assign(data, data + size);
    });
}

// Reads back what the visibility feature leaves for the features after it.
struct VisibilityProbe final : render::RenderFeature {
    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto const outputs = graph.blackboard().get<render::VisibilityOutputs>();
        auto const count = static_cast<std::size_t>(view.extent.x) * static_cast<std::size_t>(view.extent.y);
        readBack(graph, "probed triangles", outputs.triangles, count, triangles);
        readBack(graph, "probed depth", outputs.depth, count, depth);
        readBack(graph, "probed visible instances", outputs.visible, std::size_t{view.instanceCount} + 1, visible);
    }

    // The last frame's: per pixel, rows from the top, and the visible instances.
    std::shared_ptr<std::vector<tc::uint>> triangles = std::make_shared<std::vector<tc::uint>>();
    std::shared_ptr<std::vector<tc::uint>> depth = std::make_shared<std::vector<tc::uint>>();
    std::shared_ptr<std::vector<tc::uvec4>> visible = std::make_shared<std::vector<tc::uvec4>>();
};

// Reads back the temporal anti-aliasing feature's motion of the last frame where it ran.
struct MotionProbe final : render::RenderFeature {
    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        if (graph.blackboard().contains<render::TemporalOutputs>()) {
            auto const count = static_cast<std::size_t>(view.extent.x) * static_cast<std::size_t>(view.extent.y);
            readBack(graph, "probed motion", graph.blackboard().get<render::TemporalOutputs>().motion, count, motion);
        }
    }

    std::shared_ptr<std::vector<core::vec2>> motion = std::make_shared<std::vector<core::vec2>>();
};

// The device GPU backends test on: Mesa's software devices, which have no sync files, or
// the one whose name contains LRENDER_TEST_DEVICE.
inline std::string deviceFor(std::string const &backend) {
    if (backend == "cpu") {
        return "";
    }
    auto const *const device = std::getenv("LRENDER_TEST_DEVICE");
    return device != nullptr ? device : "llvmpipe";
}

// The CPU backend, then each GPU backend on deviceFor() its name. GPU backends that cannot
// start are skipped, unless LRENDER_TEST_REQUIRE_GPU is set.
template <class Fn>
void forEachBackend(HarnessOptions options, Fn const &fn) {
    auto const requireGpu = std::getenv("LRENDER_TEST_REQUIRE_GPU") != nullptr;
    for (auto const &backend : tc::availableBackends()) {
        CAPTURE(backend);
        options.backend = backend;
        options.device = deviceFor(backend);
        auto harness = std::unique_ptr<RenderHarness>{};
        try {
            harness = std::make_unique<RenderHarness>(options);
        } catch (tc::DeviceError const &error) {
            if (requireGpu) {
                FAIL(error.what());
            }
            MESSAGE("skipped: " << error.what());
            continue;
        }
        fn(*harness);
    }
}

// Throws tc::DeviceError when its pass is recorded for the `failingCall`th time, as tc
// does when the device is lost while a frame is recorded.
struct DeviceLossInjector final : render::RenderFeature {
    explicit DeviceLossInjector(int const failingCall) : callsLeft(std::make_shared<std::atomic<int>>(failingCall)) {}

    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto &pass = graph.addPass("lose the device");
        pass.read(view.sceneColor);
        pass.write(view.sceneColor);
        pass.record([callsLeft = callsLeft](render::PassContext &) {
            if (callsLeft->fetch_sub(1) == 1) {
                throw tc::DeviceError{"lr::test: the device is lost"};
            }
        });
    }

    std::shared_ptr<std::atomic<int>> callsLeft;
};

inline render::FrameDesc frameOf(core::RenderWorldDelta delta, render::ViewDesc const &view) {
    return render::FrameDesc{std::move(delta), {view}};
}

inline core::ImageHandle share(core::ImageData image) {
    return std::make_shared<core::ImageData const>(std::move(image));
}

// An RGBA8 image whose texel (x, y) is texel(x, y), a tc::ivec4 of channels in [0, 255].
template <class Texel>
core::ImageHandle rgba8Image(core::ivec2 const extent, core::ColorEncoding const encoding, Texel const &texel) {
    auto image = core::ImageData{};
    image.extent = extent;
    image.encoding = encoding;
    auto &level = image.levels.emplace_back(static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y) * 4);
    for (auto y = 0; y < extent.y; ++y) {
        for (auto x = 0; x < extent.x; ++x) {
            auto const value = texel(x, y);
            auto const index =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(extent.x) + static_cast<std::size_t>(x)) * 4;
            for (auto c = 0; c < 4; ++c) {
                level[index + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(value[c]);
            }
        }
    }
    return share(std::move(image));
}

inline float srgbToLinear(float const encoded) {
    return encoded <= 0.04045f ? encoded / 12.92f : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

// A camera on +z, `distance` from a quad of half size `half` facing it, that fills the
// middle half of a square view: pixel i of `pixels` across the quad sees u = (i + 0.5) / pixels.
inline core::CameraDesc halfFillingCamera(float const half = 1.0f, float const distance = 4.0f) {
    auto camera = cameraAt(core::vec3{0.0f, 0.0f, distance}, core::vec3{0.0f});
    camera.verticalFov = 2.0f * std::atan(2.0f * half / distance);
    return camera;
}

// A black, unlit-looking material whose texture-scaled emission shows the texture.
inline core::MaterialDesc emissiveTexture(core::TextureId const texture, core::vec3 const &emission = core::vec3{1.0f}) {
    auto material = core::MaterialDesc{};
    material.baseColor = core::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    material.emissive = emission;
    material.emissiveTexture = texture;
    return material;
}

} // namespace lr::test
