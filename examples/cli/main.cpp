// lrender_cli: renders the demo scene offline and writes it to a PPM file.
//
//   lrender_cli [--backend cpu|opengl|gles|vulkan] [--device <name part>] [--size <w>x<h>]
//               [--pipeline visibility|pathtrace] [--taa] [--samples <n>] [--timers on|off]
//               [--out <file> | --sequence <directory>]
//
// The path tracer renders --samples frames, one sample per pixel each, and writes the last
// to --out; with --sequence every frame goes to <directory>/frame<index>.ppm instead.
// --taa turns on the visibility pipeline's temporal anti-aliasing, which then renders
// --samples frames (16 unless given) of the still scene.

#include <lr/examples/CommandLine.h>
#include <lr/examples/DemoScene.h>
#include <lr/present/ImageSequencePresenter.h>
#include <lr/present/Ppm.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <tc/device.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>

namespace {

constexpr auto kTaaFrames = std::uint32_t{16};

struct Options {
    std::string backend = "cpu";
    std::string device;
    lr::core::ivec2 size{960, 540};
    bool pathTrace = false;
    bool taa = false;
    std::optional<std::uint32_t> samples;
    bool timers = false;
    std::string out = "lrender.ppm";
    std::string sequence; // a directory: every frame through a window target
};

std::uint32_t framesOf(Options const &options) {
    if (options.pathTrace) {
        return options.samples.value_or(1u);
    }
    return options.taa ? options.samples.value_or(kTaaFrames) : 1u;
}

// Applies one flag and its value; false when either is not understood.
bool apply(Options &options, std::string const &flag, std::string const &value) {
    if (flag == "--backend") {
        options.backend = value;
    } else if (flag == "--device") {
        options.device = value;
    } else if (flag == "--size") {
        auto const size = lr::examples::parseSize(value);
        options.size = size.value_or(lr::core::ivec2{0});
        return size.has_value();
    } else if (flag == "--pipeline") {
        options.pathTrace = value == "pathtrace";
        return value == "pathtrace" or value == "visibility";
    } else if (flag == "--samples") {
        options.samples = lr::examples::parseCount(value);
        return options.samples.has_value();
    } else if (flag == "--timers") {
        options.timers = value == "on";
        return value == "on" or value == "off";
    } else if (flag == "--out") {
        options.out = value;
    } else if (flag == "--sequence") {
        options.sequence = value;
    } else {
        return false;
    }
    return true;
}

std::optional<Options> parse(int const argc, char **const argv) {
    auto options = Options{};
    for (auto index = 1; index < argc; ++index) {
        auto const flag = std::string{argv[index]};
        if (flag == "--taa") {
            options.taa = true;
            continue;
        }
        if (index + 1 == argc or not apply(options, flag, argv[index + 1])) {
            return std::nullopt;
        }
        ++index;
    }
    return options;
}

void printTimings(lr::render::ViewResult const &view) {
    for (auto const &pass : view.passTimings) {
        if (pass.milliseconds) {
            std::printf("  %-28s %2u dispatches %9.3f ms\n", pass.pass.c_str(), pass.dispatches, *pass.milliseconds);
        } else {
            std::printf("  %-28s %2u dispatches   (no time)\n", pass.pass.c_str(), pass.dispatches);
        }
    }
}

std::shared_ptr<lr::present::ImageSequencePresenter> sequenceOf(Options const &options, lr::runtime::Runtime &runtime) {
    if (options.sequence.empty()) {
        return nullptr;
    }
    auto config = lr::present::ImageSequenceConfig{};
    config.directory = options.sequence;
    config.runtime = &runtime;
    return lr::present::makeImageSequencePresenter(config);
}

lr::render::RenderTargetId targetOf(lr::render::Renderer &renderer, Options const &options,
                                    std::shared_ptr<lr::present::ImageSequencePresenter> const &sequence) {
    if (sequence != nullptr) {
        return renderer.targets().createWindow(lr::render::WindowTargetDesc{options.size, sequence});
    }
    return renderer.targets().createImage(lr::render::ImageTargetDesc{options.size});
}

// Where the frames went: the last one into --out, or all of them into --sequence.
bool writeOutput(Options const &options, lr::render::ViewResult const &view,
                 lr::present::ImageSequencePresenter *const sequence) {
    if (sequence != nullptr) {
        auto const progress = sequence->flush();
        std::printf(" -> %s/frame*.ppm, %llu written", options.sequence.c_str(),
                    static_cast<unsigned long long>(progress.written));
        return progress.failed == 0;
    }
    std::printf(" -> %s", options.out.c_str());
    return view.image and lr::present::writePpm(lr::render::imageView(*view.image), options.out);
}

int run(Options const &options) {
    auto runtime = lr::runtime::makeRuntime(lr::runtime::RuntimeConfig{});
    auto const sequence = sequenceOf(options, *runtime);
    auto config = lr::render::RendererConfig{};
    config.backend = options.backend;
    config.deviceNameContains = options.device;
    config.runtime = runtime.get();
    config.gpuTimers = options.timers;
    auto renderer = lr::render::makeRenderer(config);
    auto const &capabilities = renderer->capabilities();
    std::printf("backend %s: %s; subgroups %s, single-pass visibility %s\n", renderer->deviceInfo().backend.c_str(),
                renderer->deviceInfo().deviceName.c_str(), capabilities.subgroups ? "on" : "off",
                capabilities.singlePassVisibility ? "on" : "off");

    auto scene = lr::examples::makeDemoScene();
    auto view = lr::render::ViewDesc{};
    view.id = lr::core::ViewId{0};
    view.target = targetOf(*renderer, options, sequence);
    view.pipeline = options.pathTrace ? renderer->pipelines().pathTraceReference() : renderer->pipelines().visibility();
    view.camera = scene.camera;
    view.settings.antiAliasing = options.taa ? lr::render::AntiAliasing::Taa : lr::render::AntiAliasing::None;
    auto delta = std::move(scene.delta);

    auto const frames = framesOf(options);
    auto last = lr::render::FrameResult{};
    auto const start = std::chrono::steady_clock::now();
    for (auto frame = 0u; frame < frames; ++frame) {
        auto result = lexec::sync_wait(renderer->submit(lr::render::FrameDesc{std::move(delta), {view}}));
        delta = lr::core::RenderWorldDelta{};
        if (not result) {
            std::fprintf(stderr, "the frame was stopped\n");
            return EXIT_FAILURE;
        }
        last = std::get<0>(std::move(*result));
    }
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
    auto const &result = last.views.at(0);
    std::printf("%dx%d, %u frames in %.1f ms", options.size.x, options.size.y, frames, elapsed.count());
    if (options.pathTrace) {
        std::printf(", %u samples per pixel", result.stats.accumulatedSamples);
    } else {
        std::printf(", %u instances and %u triangles visible%s", result.stats.visibleInstances,
                    result.stats.visibleTriangles, options.taa ? ", temporally anti-aliased" : "");
    }
    auto const written = writeOutput(options, result, sequence.get());
    std::printf("\n");
    if (not written) {
        std::fprintf(stderr, "cannot write the output\n");
        return EXIT_FAILURE;
    }
    printTimings(result);
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr, "usage: lrender_cli [--backend cpu|opengl|gles|vulkan] [--device <name part>] "
                             "[--size <w>x<h>]\n"
                             "                   [--pipeline visibility|pathtrace] [--taa] [--samples <n>] "
                             "[--timers on|off]\n"
                             "                   [--out <file> | --sequence <directory>]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "lrender_cli: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
