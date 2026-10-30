#include "PresentTestSupport.h"

#include <lr/present/ImageSequencePresenter.h>
#include <lr/present/Ppm.h>
#include <lr/render/Renderer.h>

#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace lr;

// The pixels of a binary PPM as RGBA8 with opaque alpha; nullopt when it is not one.
std::optional<render::ImageRgba8> readPpm(std::filesystem::path const &path) {
    auto file = std::ifstream{path, std::ios::binary};
    auto magic = std::string{};
    auto width = 0;
    auto height = 0;
    auto maximum = 0;
    file >> magic >> width >> height >> maximum;
    file.get();
    if (not file or magic != "P6" or maximum != 255 or width <= 0 or height <= 0) {
        return std::nullopt;
    }
    auto const bytes = std::vector<unsigned char>(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    auto const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (bytes.size() != 3 * count) {
        return std::nullopt;
    }
    auto image = render::ImageRgba8{core::ivec2{width, height}, std::vector<std::uint32_t>(count)};
    for (auto index = std::size_t{0}; index < count; ++index) {
        image.pixels[index] = std::uint32_t{bytes[3 * index]} | std::uint32_t{bytes[3 * index + 1]} << 8u
                            | std::uint32_t{bytes[3 * index + 2]} << 16u | 0xFF000000u;
    }
    return image;
}

std::unique_ptr<runtime::Runtime> smallRuntime() {
    auto config = runtime::RuntimeConfig{};
    config.framePoolThreads = 2;
    return runtime::makeRuntime(config);
}

std::shared_ptr<present::ImageSequencePresenter> sequenceIn(std::filesystem::path const &directory,
                                                            runtime::Runtime &runtime) {
    auto config = present::ImageSequenceConfig{};
    config.directory = directory;
    config.prefix = "frame";
    config.runtime = &runtime;
    return present::makeImageSequencePresenter(config);
}

TEST_CASE("an image sequence writes each frame it is given to a file named after the frame") {
    auto const directory = test::freshOutputDirectory("image_sequence_frames");
    auto runtime = smallRuntime();
    auto const sequence = sequenceIn(directory, *runtime);
    auto const first = render::ImageRgba8{core::ivec2{3, 2}, {0xFF0000FFu, 0xFF00FF00u, 0xFFFF0000u,
                                                              0xFF000000u, 0xFFFFFFFFu, 0xFF804020u}};
    auto const second = render::ImageRgba8{core::ivec2{1, 1}, {0xFF123456u}};

    sequence->present(render::PresentFrame{7, core::ViewId{0}, core::RenderTargetId{0}, render::imageView(first)});
    sequence->present(render::PresentFrame{8, core::ViewId{0}, core::RenderTargetId{0}, render::imageView(second)});
    auto const progress = sequence->flush();

    CHECK(progress.written == 2);
    CHECK(progress.failed == 0);
    auto const seven = readPpm(directory / "frame7.ppm");
    REQUIRE(seven.has_value());
    CHECK(seven->extent == first.extent);
    CHECK(seven->pixels == first.pixels);
    auto const eight = readPpm(directory / "frame8.ppm");
    REQUIRE(eight.has_value());
    CHECK(eight->pixels == second.pixels);
}

TEST_CASE("an image sequence counts the frames it cannot write") {
    auto runtime = smallRuntime();
    auto const sequence = sequenceIn(std::filesystem::path{LR_TEST_OUTPUT_DIR} / "no such directory", *runtime);
    auto const image = render::ImageRgba8{core::ivec2{1, 1}, {0xFFFFFFFFu}};

    sequence->present(render::PresentFrame{0, core::ViewId{0}, core::RenderTargetId{0}, render::imageView(image)});

    auto const progress = sequence->flush();
    CHECK(progress.written == 0);
    CHECK(progress.failed == 1);
}

TEST_CASE("the frames of a window target with an image sequence end up as files") {
    constexpr auto kFrames = 3u;
    auto const directory = test::freshOutputDirectory("image_sequence_window");
    auto harness = test::RenderHarness{};
    auto const sequence = sequenceIn(directory, *harness.runtimeInstance);
    auto &targets = harness.renderer->targets();
    auto const extent = core::ivec2{48, 32};
    auto const window = targets.createWindow(render::WindowTargetDesc{extent, sequence});
    auto const image = targets.createImage(render::ImageTargetDesc{extent});
    auto const camera = test::presentedCamera();
    auto const views = std::vector<render::ViewDesc>{test::viewOf(window, camera, core::ViewId{0}),
                                                     test::viewOf(image, camera, core::ViewId{1})};

    auto results = std::vector<render::FrameResult>{};
    for (auto frame = 0u; frame < kFrames; ++frame) {
        results.push_back(harness.render(render::FrameDesc{frame == 0 ? test::presentedScene() : core::RenderWorldDelta{}, views}));
    }
    auto const progress = sequence->flush();

    CHECK(progress.written == kFrames);
    for (auto const &result : results) {
        CAPTURE(result.frameIndex);
        auto const written = readPpm(directory / ("frame" + std::to_string(result.frameIndex) + ".ppm"));
        REQUIRE(written.has_value());
        CHECK(written->extent == extent);
        CHECK(written->pixels == result.views.at(1).image->pixels);
    }
}

} // namespace
