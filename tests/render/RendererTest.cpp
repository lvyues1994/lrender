#include "RenderTestSupport.h"

#include <lr/core/Primitives.h>
#include <lr/render/Presenter.h>
#include <lr/render/Renderer.h>

#include <lexec/execution.hpp>
#include <lexec/scopes/counting_scope.hpp>
#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace lr;
using test::RenderHarness;

constexpr auto kSize = 16;

core::RenderWorldDelta quadScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, test::share(core::makeQuad(core::vec2{0.5f, 0.5f}))}};
    delta.materials = {{core::MaterialId{0}, test::emissive(core::vec3{0.6f, 0.1f, 0.05f})}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.1f, 0.2f, 0.3f}};
    return delta;
}

core::CameraDesc frontCamera() { return test::cameraAt(core::vec3{0.0f, 0.0f, 3.0f}, core::vec3{0.0f}); }

render::RenderTargetId imageTarget(RenderHarness &harness, core::ivec2 const extent = core::ivec2{kSize, kSize}) {
    return harness.renderer->targets().createImage(render::ImageTargetDesc{extent});
}

render::ViewDesc pathTraced(RenderHarness &harness, core::RenderTargetId const target,
                            core::ViewId const id = core::ViewId{0}) {
    auto view = test::viewOf(target, frontCamera(), id);
    view.pipeline = harness.renderer->pipelines().pathTraceReference();
    return view;
}

struct Outcome {
    std::size_t submitted = 0; // the position of the frame among those submitted together
    std::optional<render::FrameResult> result;
    std::exception_ptr failure;
};

// Submits every frame before waiting for any, and lists what their senders complete with
// in the order they complete.
std::vector<Outcome> renderBackToBack(render::Renderer &renderer, std::vector<render::FrameDesc> frames) {
    auto mutex = std::mutex{};
    auto outcomes = std::vector<Outcome>{};
    auto scope = lexec::counting_scope{};
    auto const record = [&mutex, &outcomes](Outcome outcome) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        outcomes.push_back(std::move(outcome));
    };
    for (auto index = std::size_t{0}; index < frames.size(); ++index) {
        lexec::spawn(renderer.submit(std::move(frames[index]))
                         | lexec::then([&record, index](render::FrameResult result) noexcept {
                               record(Outcome{index, std::move(result), nullptr});
                           })
                         | lexec::upon_error([&record, index](std::exception_ptr failure) noexcept {
                               record(Outcome{index, std::nullopt, std::move(failure)});
                           })
                         | lexec::upon_stopped([&record, index]() noexcept {
                               record(Outcome{index, std::nullopt, std::make_exception_ptr(std::runtime_error{"stopped"})});
                           }),
                     scope.get_token());
    }
    static_cast<void>(lexec::sync_wait(scope.join()));
    return outcomes;
}

template <class E>
bool failedWith(std::exception_ptr const &failure) {
    try {
        std::rethrow_exception(failure);
    } catch (E const &) {
        return true;
    } catch (...) {
        return false;
    }
}

constexpr auto kOrderedFrames = 12u;

TEST_CASE("frames submitted back to back are rendered and completed in order, on every backend") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &harness) {
        auto const view = pathTraced(harness, imageTarget(harness, core::ivec2{8, 8}));
        auto frames = std::vector<render::FrameDesc>{};
        for (auto frame = 0u; frame < kOrderedFrames; ++frame) {
            frames.push_back(render::FrameDesc{frame == 0 ? quadScene() : core::RenderWorldDelta{}, {view}});
        }

        auto const outcomes = renderBackToBack(*harness.renderer, std::move(frames));

        REQUIRE(outcomes.size() == kOrderedFrames);
        for (auto index = 0u; index < kOrderedFrames; ++index) {
            CAPTURE(index);
            CHECK(outcomes[index].submitted == index);
            REQUIRE(outcomes[index].result.has_value());
            auto const &result = *outcomes[index].result;
            CHECK(result.frameIndex == index);
            CHECK(result.views.at(0).stats.accumulatedSamples == index + 1);
            CHECK(result.wallTime.count() > 0.0);
        }
    });
}

TEST_CASE("accumulating frames back to back gives the image accumulating them one at a time does") {
    constexpr auto kFrames = 8u;
    auto const finalImage = [](RenderHarness &harness, bool const backToBack) {
        auto const view = pathTraced(harness, imageTarget(harness, core::ivec2{8, 8}));
        if (backToBack) {
            auto frames = std::vector<render::FrameDesc>{};
            for (auto frame = 0u; frame < kFrames; ++frame) {
                frames.push_back(render::FrameDesc{frame == 0 ? quadScene() : core::RenderWorldDelta{}, {view}});
            }
            return *renderBackToBack(*harness.renderer, std::move(frames)).back().result->views[0].image;
        }
        auto last = harness.render(render::FrameDesc{quadScene(), {view}});
        for (auto frame = 1u; frame < kFrames; ++frame) {
            last = harness.render(render::FrameDesc{{}, {view}});
        }
        return *last.views[0].image;
    };
    test::forEachBackend(test::HarnessOptions{}, [&finalImage](RenderHarness &harness) {
        auto options = test::HarnessOptions{};
        options.backend = harness.renderer->deviceInfo().backend;
        options.device = options.backend == "cpu" ? "" : "llvmpipe";
        options.framesInFlight = 1;
        auto oneAtATime = RenderHarness{options};

        CHECK(test::maxChannelDifference(finalImage(harness, true), finalImage(oneAtATime, false)) == 0);
    });
}

// Counts on the GPU thread how many of the frames it is in were recorded and not yet
// finished when its pass was recorded.
struct InFlightProbe final : render::RenderFeature {
    struct Counts {
        std::uint32_t recorded = 0;
        std::uint32_t finished = 0;
        std::uint32_t most = 0;
    };

    render::FeatureStage stage() const noexcept override { return render::FeatureStage::PostProcess; }

    void setup(render::FrameGraphBuilder &graph, render::ViewContext const &view) override {
        auto &pass = graph.addPass("count frames in flight");
        pass.read(view.sceneColor);
        pass.write(view.sceneColor);
        pass.record([counts = counts](render::PassContext &) {
            ++counts->recorded;
            counts->most = std::max(counts->most, counts->recorded - counts->finished);
        });
        graph.onFinish([counts = counts](render::ViewResult &) { ++counts->finished; });
    }

    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
};

constexpr auto kBoundedFrames = 10u;

TEST_CASE("no more than framesInFlight frames are recorded and not yet finished, on every backend") {
    for (auto const limit : {1u, 2u, 3u}) {
        CAPTURE(limit);
        auto options = test::HarnessOptions{};
        options.framesInFlight = limit;
        test::forEachBackend(options, [limit](RenderHarness &harness) {
            auto &pipelines = harness.renderer->pipelines();
            auto const probe = std::make_shared<InFlightProbe>();
            auto view = test::viewOf(imageTarget(harness), frontCamera());
            view.pipeline = pipelines.create(render::PipelineDesc{pipelines.visibility(), {probe}});
            auto frames = std::vector<render::FrameDesc>{};
            for (auto frame = 0u; frame < kBoundedFrames; ++frame) {
                frames.push_back(render::FrameDesc{frame == 0 ? quadScene() : core::RenderWorldDelta{}, {view}});
            }

            auto const outcomes = renderBackToBack(*harness.renderer, std::move(frames));

            auto const succeeded = std::count_if(outcomes.begin(), outcomes.end(),
                                                 [](Outcome const &outcome) { return outcome.result.has_value(); });
            CHECK(succeeded == kBoundedFrames);
            CHECK(probe->counts->recorded == kBoundedFrames);
            CHECK(probe->counts->finished == kBoundedFrames);
            CAPTURE(probe->counts->most);
            CHECK(probe->counts->most >= 1);
            CHECK(probe->counts->most <= limit);
        });
    }
}

TEST_CASE("a renderer needs at least one frame in flight") {
    auto options = test::HarnessOptions{};
    options.framesInFlight = 0;
    CHECK_THROWS_AS(RenderHarness{options}, std::invalid_argument);
}

TEST_CASE("a frame that fails keeps its place among the frames submitted around it") {
    auto harness = RenderHarness{};
    auto const view = test::viewOf(imageTarget(harness), frontCamera());
    auto broken = core::RenderWorldDelta{};
    broken.instances = {{core::InstanceId{1}, core::MeshId{99}, core::MaterialId{0}, core::mat4{1.0f}}};
    auto frames = std::vector<render::FrameDesc>{};
    frames.push_back(render::FrameDesc{quadScene(), {view}});
    frames.push_back(render::FrameDesc{std::move(broken), {view}});
    frames.push_back(render::FrameDesc{{}, {view}});

    auto const outcomes = renderBackToBack(*harness.renderer, std::move(frames));

    REQUIRE(outcomes.size() == 3);
    for (auto index = std::size_t{0}; index < outcomes.size(); ++index) {
        CHECK(outcomes[index].submitted == index);
    }
    REQUIRE(outcomes[0].result.has_value());
    CHECK(failedWith<std::invalid_argument>(outcomes[1].failure));
    REQUIRE(outcomes[2].result.has_value());
    CHECK(outcomes[2].result->views[0].image->pixels == outcomes[0].result->views[0].image->pixels);
}

struct Presented {
    std::uint64_t frameIndex = 0;
    core::ViewId view;
    core::RenderTargetId target;
    render::ImageRgba8 image;
    std::thread::id thread;
};

struct RecordingPresenter final : render::Presenter {
    void present(render::PresentFrame const &frame) override {
        auto const &pixels = frame.pixels;
        auto const count = static_cast<std::size_t>(pixels.extent.x) * static_cast<std::size_t>(pixels.extent.y);
        auto image = render::ImageRgba8{pixels.extent, std::vector<std::uint32_t>(pixels.pixels, pixels.pixels + count)};
        auto const lock = std::lock_guard<std::mutex>{mutex};
        frames.push_back(Presented{frame.frameIndex, frame.view, frame.target, std::move(image), std::this_thread::get_id()});
    }

    std::vector<Presented> presented() {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return frames;
    }

private:
    std::mutex mutex;
    std::vector<Presented> frames;
};

TEST_CASE("a window target hands each frame to its presenter on the GPU thread instead of returning it") {
    auto harness = RenderHarness{};
    auto &targets = harness.renderer->targets();
    auto const presenter = std::make_shared<RecordingPresenter>();
    auto const window = targets.createWindow(render::WindowTargetDesc{core::ivec2{kSize, kSize}, presenter});
    auto const image = imageTarget(harness);

    auto const result = harness.render(render::FrameDesc{
        quadScene(), {test::viewOf(window, frontCamera(), core::ViewId{0}), test::viewOf(image, frontCamera(), core::ViewId{1})}});

    auto const presented = presenter->presented();
    REQUIRE(presented.size() == 1);
    CHECK(presented[0].frameIndex == result.frameIndex);
    CHECK(presented[0].view == core::ViewId{0});
    CHECK(presented[0].target == window);
    auto const onAnotherThread = presented[0].thread != std::this_thread::get_id();
    CHECK(onAnotherThread);
    REQUIRE(result.views.size() == 2);
    CHECK(not result.views[0].image.has_value());
    REQUIRE(result.views[1].image.has_value());
    CHECK(presented[0].image.extent == result.views[1].image->extent);
    CHECK(presented[0].image.pixels == result.views[1].image->pixels);
    CHECK_THROWS_AS(targets.createWindow(render::WindowTargetDesc{core::ivec2{kSize, kSize}, nullptr}),
                    std::invalid_argument);
}

TEST_CASE("resizing a window target resizes the frames its presenter gets and starts the view's accumulation over") {
    auto harness = RenderHarness{};
    auto const presenter = std::make_shared<RecordingPresenter>();
    auto const window = harness.renderer->targets().createWindow(render::WindowTargetDesc{core::ivec2{kSize, kSize}, presenter});
    auto const view = pathTraced(harness, window);
    auto const samples = [&harness, &view](core::RenderWorldDelta delta) {
        return harness.render(render::FrameDesc{std::move(delta), {view}}).views.at(0).stats.accumulatedSamples;
    };

    CHECK(samples(quadScene()) == 1);
    CHECK(samples({}) == 2);
    harness.renderer->targets().resize(window, core::ivec2{8, 4});
    CHECK(samples({}) == 1);

    auto const presented = presenter->presented();
    REQUIRE(presented.size() == 3);
    CHECK(presented[1].image.extent == core::ivec2{kSize, kSize});
    CHECK(presented[2].image.extent == core::ivec2{8, 4});
    CHECK(presented[2].image.pixels.size() == 32);
}

TEST_CASE("frames submitted before a window target is destroyed still reach its presenter") {
    auto harness = RenderHarness{};
    auto &targets = harness.renderer->targets();
    auto const presenter = std::make_shared<RecordingPresenter>();
    auto const window = targets.createWindow(render::WindowTargetDesc{core::ivec2{kSize, kSize}, presenter});
    auto const view = test::viewOf(window, frontCamera());

    auto sender = harness.renderer->submit(render::FrameDesc{quadScene(), {view}});
    targets.destroy(window);
    CHECK(lexec::sync_wait(std::move(sender)).has_value());

    CHECK(presenter->presented().size() == 1);
    CHECK_THROWS_AS(harness.renderer->submit(render::FrameDesc{{}, {view}}), std::invalid_argument);
}

TEST_CASE("a lost device fails its frame, and the next frame renders the whole scene on a new device, on every backend") {
    test::forEachBackend(test::HarnessOptions{}, [](RenderHarness &harness) {
        auto &pipelines = harness.renderer->pipelines();
        auto const visibility = test::viewOf(imageTarget(harness), frontCamera(), core::ViewId{0});
        auto const traced = pathTraced(harness, imageTarget(harness), core::ViewId{1});
        auto lossy = visibility;
        lossy.pipeline = pipelines.create(
            render::PipelineDesc{pipelines.visibility(), {std::make_shared<test::DeviceLossInjector>(1)}});

        auto const first = harness.render(render::FrameDesc{quadScene(), {visibility, traced}});
        CHECK(harness.render(render::FrameDesc{{}, {visibility, traced}}).views.at(1).stats.accumulatedSamples == 2);
        CHECK_THROWS_AS(harness.render(render::FrameDesc{{}, {lossy, traced}}), tc::DeviceError);
        auto const after = harness.render(render::FrameDesc{{}, {lossy, traced}});

        // Nothing in the last delta: the scene, hierarchies included, came from the render
        // thread's copy, and the path tracer starts over with the same first sample.
        CHECK(test::maxChannelDifference(*after.views.at(0).image, *first.views.at(0).image) == 0);
        CHECK(after.views.at(1).stats.accumulatedSamples == 1);
        CHECK(test::maxChannelDifference(*after.views.at(1).image, *first.views.at(1).image) == 0);
    });
}

TEST_CASE("the frames in flight when the device is lost fail with it, and the frames after it render the scene") {
    constexpr auto kFrames = 8u;
    constexpr auto kLostIn = 3u;
    auto harness = RenderHarness{};
    auto &pipelines = harness.renderer->pipelines();
    auto view = test::viewOf(imageTarget(harness), frontCamera());
    view.pipeline = pipelines.create(
        render::PipelineDesc{pipelines.visibility(),
                             {std::make_shared<test::DeviceLossInjector>(static_cast<int>(kLostIn) + 1)}});
    auto frames = std::vector<render::FrameDesc>{};
    for (auto frame = 0u; frame < kFrames; ++frame) {
        frames.push_back(render::FrameDesc{frame == 0 ? quadScene() : core::RenderWorldDelta{}, {view}});
    }

    auto const outcomes = renderBackToBack(*harness.renderer, std::move(frames));
    auto const expected = *harness.render(render::FrameDesc{{}, {view}}).views.at(0).image;

    REQUIRE(outcomes.size() == kFrames);
    for (auto index = std::size_t{0}; index < kFrames; ++index) {
        CAPTURE(index);
        auto const &outcome = outcomes[index];
        CHECK(outcome.submitted == index);
        if (outcome.result) {
            CHECK(index != kLostIn);
            CHECK(outcome.result->views.at(0).image->pixels == expected.pixels);
        } else {
            CHECK(failedWith<tc::DeviceError>(outcome.failure));
            // With two frames in flight, only the one before can be in flight with it.
            CHECK(index + 1 >= kLostIn);
        }
    }
}

} // namespace
