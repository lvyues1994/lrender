// lrender_cli: renders the demo scene offline and writes it to a PPM file.
//
//   lrender_cli [--backend cpu|opengl|gles|vulkan] [--device <name part>] [--size <w>x<h>] [--out <file>]

#include <lr/examples/DemoScene.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <tc/device.hpp>

#include <chrono>
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

std::optional<Options> parse(int const argc, char **const argv) {
    auto options = Options{};
    for (auto index = 1; index + 1 < argc; index += 2) {
        auto const flag = std::string{argv[index]};
        auto const value = std::string{argv[index + 1]};
        if (flag == "--backend") {
            options.backend = value;
        } else if (flag == "--device") {
            options.device = value;
        } else if (flag == "--size") {
            auto const size = parseSize(value);
            if (not size) {
                return std::nullopt;
            }
            options.size = *size;
        } else if (flag == "--out") {
            options.out = value;
        } else {
            return std::nullopt;
        }
    }
    if (argc % 2 == 0) {
        return std::nullopt;
    }
    return options;
}

int run(Options const &options) {
    auto runtime = lr::runtime::makeRuntime(lr::runtime::RuntimeConfig{});
    auto config = lr::render::RendererConfig{};
    config.backend = options.backend;
    config.deviceNameContains = options.device;
    config.runtime = runtime.get();
    auto renderer = lr::render::makeRenderer(config);
    std::printf("backend %s: %s\n", renderer->deviceInfo().backend.c_str(), renderer->deviceInfo().deviceName.c_str());

    auto scene = lr::examples::makeDemoScene();
    auto view = lr::render::ViewDesc{};
    view.id = lr::core::ViewId{0};
    view.target = renderer->targets().createImage(lr::render::ImageTargetDesc{options.size});
    view.camera = scene.camera;
    auto frame = lr::render::FrameDesc{std::move(scene.delta), {view}};

    auto const start = std::chrono::steady_clock::now();
    auto result = lexec::sync_wait(renderer->submit(std::move(frame)));
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
    if (not result) {
        std::fprintf(stderr, "the frame was stopped\n");
        return EXIT_FAILURE;
    }
    auto const &views = std::get<0>(*result).views;
    if (views.empty() or not views[0].image or not lr::examples::writePpm(*views[0].image, options.out)) {
        std::fprintf(stderr, "cannot write %s\n", options.out.c_str());
        return EXIT_FAILURE;
    }
    std::printf("%dx%d, %u instances and %u triangles visible, %.1f ms -> %s\n", options.size.x, options.size.y,
                views[0].stats.visibleInstances, views[0].stats.visibleTriangles, elapsed.count(), options.out.c_str());
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr, "usage: lrender_cli [--backend cpu|opengl|gles|vulkan] [--device <name part>] "
                             "[--size <w>x<h>] [--out <file>]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "lrender_cli: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
