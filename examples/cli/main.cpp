// lrender_cli: renders the demo scene offline and writes it to a PPM file.
//
//   lrender_cli [--backend cpu|opengl|gles|vulkan] [--device <name part>] [--size <w>x<h>]
//               [--pipeline visibility|pathtrace] [--samples <n>] [--timers on|off] [--out <file>]
//
// The path tracer renders --samples frames, one sample per pixel each, and writes the last.

#include <lr/examples/DemoScene.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <tc/device.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>

namespace {

struct Options {
    std::string backend = "cpu";
    std::string device;
    lr::core::ivec2 size{960, 540};
    bool pathTrace = false;
    std::uint32_t samples = 1;
    bool timers = false;
    std::string out = "lrender.ppm";
};

std::optional<lr::core::ivec2> parseSize(std::string const &text) {
    auto width = 0;
    auto height = 0;
    if (std::sscanf(text.c_str(), "%dx%d", &width, &height) != 2 or width <= 0 or height <= 0) {
        return std::nullopt;
    }
    return lr::core::ivec2{width, height};
}

std::optional<std::uint32_t> parseCount(std::string const &text) {
    auto count = 0u;
    auto end = 0;
    if (std::sscanf(text.c_str(), "%u%n", &count, &end) != 1 or static_cast<std::size_t>(end) != text.size()
        or count == 0) {
        return std::nullopt;
    }
    return count;
}

// Applies one flag and its value; false when either is not understood.
bool apply(Options &options, std::string const &flag, std::string const &value) {
    if (flag == "--backend") {
        options.backend = value;
    } else if (flag == "--device") {
        options.device = value;
    } else if (flag == "--size") {
        auto const size = parseSize(value);
        options.size = size.value_or(lr::core::ivec2{0});
        return size.has_value();
    } else if (flag == "--pipeline") {
        options.pathTrace = value == "pathtrace";
        return value == "pathtrace" or value == "visibility";
    } else if (flag == "--samples") {
        auto const samples = parseCount(value);
        options.samples = samples.value_or(0u);
        return samples.has_value();
    } else if (flag == "--timers") {
        options.timers = value == "on";
        return value == "on" or value == "off";
    } else if (flag == "--out") {
        options.out = value;
    } else {
        return false;
    }
    return true;
}

std::optional<Options> parse(int const argc, char **const argv) {
    if (argc % 2 == 0) {
        return std::nullopt;
    }
    auto options = Options{};
    for (auto index = 1; index + 1 < argc; index += 2) {
        if (not apply(options, argv[index], argv[index + 1])) {
            return std::nullopt;
        }
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

int run(Options const &options) {
    auto runtime = lr::runtime::makeRuntime(lr::runtime::RuntimeConfig{});
    auto config = lr::render::RendererConfig{};
    config.backend = options.backend;
    config.deviceNameContains = options.device;
    config.runtime = runtime.get();
    config.gpuTimers = options.timers;
    auto renderer = lr::render::makeRenderer(config);
    std::printf("backend %s: %s\n", renderer->deviceInfo().backend.c_str(), renderer->deviceInfo().deviceName.c_str());

    auto scene = lr::examples::makeDemoScene();
    auto view = lr::render::ViewDesc{};
    view.id = lr::core::ViewId{0};
    view.target = renderer->targets().createImage(lr::render::ImageTargetDesc{options.size});
    view.pipeline = options.pathTrace ? renderer->pipelines().pathTraceReference() : renderer->pipelines().visibility();
    view.camera = scene.camera;
    auto delta = std::move(scene.delta);

    auto const frames = options.pathTrace ? options.samples : 1u;
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
    auto const &views = last.views;
    if (views.empty() or not views[0].image or not lr::examples::writePpm(*views[0].image, options.out)) {
        std::fprintf(stderr, "cannot write %s\n", options.out.c_str());
        return EXIT_FAILURE;
    }
    auto const &stats = views[0].stats;
    std::printf("%dx%d, %u frames in %.1f ms", options.size.x, options.size.y, frames, elapsed.count());
    if (options.pathTrace) {
        std::printf(", %u samples per pixel", stats.accumulatedSamples);
    } else {
        std::printf(", %u instances and %u triangles visible", stats.visibleInstances, stats.visibleTriangles);
    }
    std::printf(" -> %s\n", options.out.c_str());
    printTimings(views[0]);
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr, "usage: lrender_cli [--backend cpu|opengl|gles|vulkan] [--device <name part>] "
                             "[--size <w>x<h>]\n"
                             "                   [--pipeline visibility|pathtrace] [--samples <n>] "
                             "[--timers on|off] [--out <file>]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "lrender_cli: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
