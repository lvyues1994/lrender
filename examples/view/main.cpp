// lrender_view: shows the demo scene, or a glTF file's, in an SDL window.
//
//   lrender_view [--backend cpu|opengl|gles|vulkan] [--device <name part>] [--size <w>x<h>]
//                [--pipeline visibility|pathtrace] [--frames <n>] [--gltf <file>]
//
// Drag with the left mouse button to orbit and scroll to zoom. Tab switches between the
// visibility and the path tracing pipelines (1 and 2 pick one); T turns the visibility
// pipeline's temporal anti-aliasing off and on (on to begin with); Esc or Q quits. With
// --frames it quits after that many frames, for runs nobody watches
// (SDL_VIDEODRIVER=offscreen). A --gltf scene is loaded again when its files change;
// development builds on a GPU backend also reload lrender's kernels when the build
// rewrites them.

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <lr/asset/AssetLibrary.h>
#include <lr/asset/AssetSource.h>
#include <lr/asset/FileWatcher.h>
#include <lr/asset/SceneDelta.h>
#include <lr/examples/CommandLine.h>
#include <lr/examples/DemoScene.h>
#include <lr/examples/GltfScene.h>
#include <lr/present/SdlPresenter.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <lexec/execution.hpp>
#include <lexec/scopes/counting_scope.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using lr::core::ivec2;
using lr::core::vec3;

constexpr auto kFramesInFlight = std::uint32_t{2};

struct Options {
    std::string backend = "cpu";
    std::string device;
    ivec2 size{960, 540};
    bool pathTrace = false;
    std::uint32_t frames = 0; // 0: until the window is closed
    std::string gltf;
};

bool apply(Options &options, std::string const &flag, std::string const &value) {
    if (flag == "--gltf") {
        options.gltf = value;
    } else if (flag == "--backend") {
        options.backend = value;
    } else if (flag == "--device") {
        options.device = value;
    } else if (flag == "--size") {
        auto const size = lr::examples::parseSize(value);
        options.size = size.value_or(ivec2{0});
        return size.has_value();
    } else if (flag == "--pipeline") {
        options.pathTrace = value == "pathtrace";
        return value == "pathtrace" or value == "visibility";
    } else if (flag == "--frames") {
        auto const frames = lr::examples::parseCount(value);
        options.frames = frames.value_or(0u);
        return frames.has_value();
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

// A camera on a sphere around its target.
struct Orbit {
    vec3 target{0.0f};
    float yaw = 0.0f;   // about +y, from +z
    float pitch = 0.0f; // above the target's horizon
    float distance = 5.0f;
};

Orbit orbitOf(lr::core::CameraDesc const &camera) {
    auto const offset = camera.position - camera.target;
    auto const distance = tc::length(offset);
    return Orbit{camera.target, std::atan2(offset.x, offset.z), std::asin(offset.y / distance), distance};
}

lr::core::CameraDesc cameraOf(Orbit const &orbit, lr::core::CameraDesc camera) {
    auto const flat = std::cos(orbit.pitch);
    auto const direction = vec3{flat * std::sin(orbit.yaw), std::sin(orbit.pitch), flat * std::cos(orbit.yaw)};
    camera.target = orbit.target;
    camera.position = orbit.target + direction * orbit.distance;
    return camera;
}

struct ViewerInput {
    Orbit orbit;
    bool pathTrace = false;
    bool taa = true;
    bool quit = false;
    bool resized = false;
};

void handleKey(SDL_Keycode const key, ViewerInput &input) {
    if (key == SDLK_ESCAPE or key == SDLK_q) {
        input.quit = true;
    } else if (key == SDLK_TAB) {
        input.pathTrace = not input.pathTrace;
    } else if (key == SDLK_t) {
        input.taa = not input.taa;
    } else if (key == SDLK_1) {
        input.pathTrace = false;
    } else if (key == SDLK_2) {
        input.pathTrace = true;
    }
}

void handle(SDL_Event const &event, ViewerInput &input) {
    constexpr auto kRadiansPerPixel = 0.01f;
    constexpr auto kPitchLimit = 1.5f;
    if (event.type == SDL_QUIT) {
        input.quit = true;
    } else if (event.type == SDL_KEYDOWN) {
        handleKey(event.key.keysym.sym, input);
    } else if (event.type == SDL_MOUSEMOTION and (event.motion.state & SDL_BUTTON_LMASK) != 0) {
        input.orbit.yaw -= static_cast<float>(event.motion.xrel) * kRadiansPerPixel;
        input.orbit.pitch = std::clamp(input.orbit.pitch + static_cast<float>(event.motion.yrel) * kRadiansPerPixel,
                                       -kPitchLimit, kPitchLimit);
    } else if (event.type == SDL_MOUSEWHEEL) {
        input.orbit.distance = std::clamp(input.orbit.distance * std::pow(0.9f, event.wheel.preciseY), 0.5f, 50.0f);
    } else if (event.type == SDL_WINDOWEVENT and event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        input.resized = true;
    }
}

// What the frames report back, from the GPU thread.
struct FrameReports {
    void finished(lr::render::FrameResult const &result) noexcept {
        if (not result.views.empty()) {
            samples.store(result.views[0].stats.accumulatedSamples, std::memory_order_relaxed);
        }
        count.fetch_add(1, std::memory_order_release);
    }

    void failed(std::exception_ptr const &error) noexcept {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            if (not failure) {
                failure = error;
            }
        }
        count.fetch_add(1, std::memory_order_release);
    }

    std::exception_ptr firstFailure() {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return failure;
    }

    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint32_t> samples{0};

private:
    std::mutex mutex;
    std::exception_ptr failure;
};

struct SdlVideo {
    SdlVideo() {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            throw std::runtime_error{std::string{"SDL_Init: "} + SDL_GetError()};
        }
    }
    SdlVideo(SdlVideo const &) = delete;
    SdlVideo &operator=(SdlVideo const &) = delete;
    ~SdlVideo() { SDL_Quit(); }
};

struct WindowDeleter {
    void operator()(SDL_Window *const window) const noexcept { SDL_DestroyWindow(window); }
};
struct RendererDeleter {
    void operator()(SDL_Renderer *const renderer) const noexcept { SDL_DestroyRenderer(renderer); }
};
using WindowHandle = std::unique_ptr<SDL_Window, WindowDeleter>;
using SdlRendererHandle = std::unique_ptr<SDL_Renderer, RendererDeleter>;

// An accelerated renderer synchronised to the display where there is one, else SDL's
// software renderer.
SdlRendererHandle makeSdlRenderer(SDL_Window *const window) {
    auto renderer = SdlRendererHandle{SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC)};
    if (renderer == nullptr) {
        renderer.reset(SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE));
    }
    if (renderer == nullptr) {
        throw std::runtime_error{std::string{"SDL_CreateRenderer: "} + SDL_GetError()};
    }
    return renderer;
}

ivec2 outputSize(SDL_Renderer *const renderer) {
    auto width = 0;
    auto height = 0;
    if (SDL_GetRendererOutputSize(renderer, &width, &height) != 0 or width <= 0 or height <= 0) {
        throw std::runtime_error{std::string{"SDL_GetRendererOutputSize: "} + SDL_GetError()};
    }
    return ivec2{width, height};
}

struct Session {
    lr::render::Renderer &renderer;
    lr::present::SdlPresenter &presenter;
    SDL_Window *window;
    SDL_Renderer *sdlRenderer;
    lr::render::RenderTargetId target;
    lr::core::CameraDesc camera; // the demo scene's, for everything but the orbit
};

struct LoopResult {
    std::uint64_t submitted = 0;
    std::uint64_t shown = 0;
};

using Clock = std::chrono::steady_clock;

// The build writes several files for each kernel; they are reloaded once it has been
// quiet this long.
constexpr auto kKernelsQuiet = std::chrono::milliseconds{300};

// A --gltf scene, loaded again when the asset library reports that one of its files changed.
struct SceneReloads {
    lr::asset::AssetLibrary *assets = nullptr; // null without --gltf
    std::string path;
    lr::asset::IdAllocator *ids = nullptr;
    lr::core::Removals shown; // removes the scene now in the world
    bool loading = false;
    std::vector<lr::asset::AssetChange> changes;
    std::mutex mutex; // guards `arrived`, which an IO worker fills
    std::optional<lr::asset::AssetResult<lr::asset::SceneHandle>> arrived;
};

void watchScene(SceneReloads &reloads, lexec::counting_scope &scope) {
    if (reloads.assets == nullptr or reloads.loading) {
        return;
    }
    reloads.assets->drainChanges(reloads.changes);
    if (reloads.changes.empty()) {
        return;
    }
    reloads.loading = true;
    lexec::spawn(reloads.assets->loadScene(reloads.path)
                     | lexec::then([&reloads](lr::asset::AssetResult<lr::asset::SceneHandle> scene) noexcept {
                           auto const lock = std::lock_guard<std::mutex>{reloads.mutex};
                           reloads.arrived = std::move(scene);
                       }),
                 scope.get_token());
}

// The change that swaps the scene in the world for the one just loaded, if one was.
std::optional<lr::core::RenderWorldDelta> reloadedScene(SceneReloads &reloads) {
    auto arrived = std::optional<lr::asset::AssetResult<lr::asset::SceneHandle>>{};
    {
        auto const lock = std::lock_guard<std::mutex>{reloads.mutex};
        arrived = std::exchange(reloads.arrived, std::nullopt);
    }
    if (not arrived) {
        return std::nullopt;
    }
    reloads.loading = false;
    if (not *arrived) {
        std::fprintf(stderr, "reloading %s: %s\n", reloads.path.c_str(), arrived->error().message.c_str());
        return std::nullopt;
    }
    auto scene = lr::examples::viewableScene(***arrived, *reloads.ids);
    scene.delta.removed = std::exchange(reloads.shown, scene.removal);
    std::printf("reloaded %s\n", reloads.path.c_str());
    return std::move(scene.delta);
}

// lrender's kernels, reloaded on GPU backends of development builds when the build
// rewrites them.
struct KernelReloads {
    std::unique_ptr<lr::asset::FileWatcher> watcher; // null where kernels are not reloaded
    std::string directory;
    std::optional<Clock::time_point> changedAt;
    std::vector<lr::asset::FileChange> changes;
};

KernelReloads kernelReloadsFor(lr::render::Renderer &renderer, lr::runtime::Runtime &runtime) {
    auto reloads = KernelReloads{};
    auto const directory = lr::render::builtInKernelDirectory();
    if (not directory or renderer.deviceInfo().backend == "cpu") {
        return reloads;
    }
    auto config = lr::asset::FileWatcherConfig{};
    config.runtime = &runtime;
    reloads.watcher = lr::asset::makeFileWatcher(config);
    if (auto const error = reloads.watcher->watchDirectory(*directory)) {
        std::fprintf(stderr, "cannot watch %s: %s\n", directory->c_str(), error.message().c_str());
        reloads.watcher.reset();
        return reloads;
    }
    reloads.directory = *directory;
    std::printf("kernels reload from %s when the build rewrites them\n", directory->c_str());
    return reloads;
}

void printReport(tc::ReloadReport const &report) noexcept {
    std::printf("reloaded %zu kernels\n", report.reloaded.size());
    for (auto const &rejection : report.rejected) {
        std::fprintf(stderr, "kernel %s not reloaded: %s\n", rejection.program.c_str(), rejection.reason.c_str());
    }
}

void watchKernels(KernelReloads &reloads, lr::render::Renderer &renderer, lexec::counting_scope &scope) {
    if (reloads.watcher == nullptr) {
        return;
    }
    reloads.watcher->drainChanges(reloads.changes);
    if (std::any_of(reloads.changes.begin(), reloads.changes.end(), [](lr::asset::FileChange const &change) {
            return change.kind == lr::asset::FileChangeKind::Modified;
        })) {
        reloads.changedAt = Clock::now();
    }
    if (not reloads.changedAt or Clock::now() - *reloads.changedAt < kKernelsQuiet) {
        return;
    }
    reloads.changedAt.reset();
    lexec::spawn(renderer.reloadKernels(reloads.directory)
                     | lexec::then([](tc::ReloadReport const &report) noexcept { printReport(report); })
                     | lexec::upon_error([](std::exception_ptr const &) noexcept {
                           std::fprintf(stderr, "reloading the kernels failed\n");
                       }),
                 scope.get_token());
}

struct Reloads {
    SceneReloads &scene;
    KernelReloads &kernels;
};

void submitFrame(Session const &session, ViewerInput const &input, lr::core::RenderWorldDelta delta,
                 FrameReports &reports, lexec::counting_scope &scope) {
    auto &pipelines = session.renderer.pipelines();
    auto view = lr::render::ViewDesc{};
    view.id = lr::core::ViewId{0};
    view.target = session.target;
    view.pipeline = input.pathTrace ? pipelines.pathTraceReference() : pipelines.visibility();
    view.camera = cameraOf(input.orbit, session.camera);
    view.settings.antiAliasing = input.taa ? lr::render::AntiAliasing::Taa : lr::render::AntiAliasing::None;
    lexec::spawn(session.renderer.submit(lr::render::FrameDesc{std::move(delta), {view}})
                     | lexec::then([&reports](lr::render::FrameResult const &result) noexcept { reports.finished(result); })
                     | lexec::upon_error([&reports](std::exception_ptr const &error) noexcept { reports.failed(error); })
                     | lexec::upon_stopped([&reports]() noexcept {
                           reports.failed(std::make_exception_ptr(std::runtime_error{"a frame was stopped"}));
                       }),
                 scope.get_token());
}

void showStatus(Session const &session, ViewerInput const &input, FrameReports const &reports) {
    auto const title = input.pathTrace ? "lrender: path tracing, " + std::to_string(reports.samples.load()) + " samples"
                                       : std::string{"lrender: visibility buffer, TAA "} + (input.taa ? "on" : "off");
    SDL_SetWindowTitle(session.window, title.c_str());
}

// Keeps at most kFramesInFlight frames outstanding and shows the newest each turn, never
// waiting for a frame.
LoopResult runLoop(Session const &session, Options const &options, lr::core::RenderWorldDelta scene,
                   Reloads const &reloads, FrameReports &reports, lexec::counting_scope &scope) {
    auto input = ViewerInput{orbitOf(session.camera), options.pathTrace, true, false, false};
    auto result = LoopResult{};
    auto delta = std::move(scene);
    while (not input.quit and not reports.firstFailure()) {
        watchScene(reloads.scene, scope);
        watchKernels(reloads.kernels, session.renderer, scope);
        auto event = SDL_Event{};
        while (SDL_PollEvent(&event) != 0) {
            handle(event, input);
        }
        if (input.resized) {
            session.renderer.targets().resize(session.target, outputSize(session.sdlRenderer));
            input.resized = false;
        }
        auto const finished = reports.count.load(std::memory_order_acquire);
        if (options.frames != 0 and finished >= options.frames) {
            break;
        }
        auto const wanted = options.frames == 0 or result.submitted < options.frames;
        auto const submitting = wanted and result.submitted - finished < kFramesInFlight;
        if (submitting and result.submitted != 0) {
            delta = reloadedScene(reloads.scene).value_or(lr::core::RenderWorldDelta{});
        }
        if (submitting) {
            submitFrame(session, input, std::move(delta), reports, scope);
            delta = lr::core::RenderWorldDelta{};
            ++result.submitted;
        }
        auto const shown = session.presenter.blitLatest().has_value();
        if (shown) {
            ++result.shown;
            showStatus(session, input, reports);
        }
        if (not submitting and not shown) {
            SDL_Delay(1);
        }
    }
    return result;
}

int run(Options const &options) {
    auto const video = SdlVideo{};
    auto const window = WindowHandle{SDL_CreateWindow("lrender", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                                      options.size.x, options.size.y, SDL_WINDOW_RESIZABLE)};
    if (window == nullptr) {
        throw std::runtime_error{std::string{"SDL_CreateWindow: "} + SDL_GetError()};
    }
    auto const sdlRenderer = makeSdlRenderer(window.get());
    auto presenter = lr::present::makeSdlPresenter(lr::present::SdlPresenterConfig{sdlRenderer.get()});
    auto runtime = lr::runtime::makeRuntime(lr::runtime::RuntimeConfig{});
    auto config = lr::render::RendererConfig{};
    config.backend = options.backend;
    config.deviceNameContains = options.device;
    config.runtime = runtime.get();
    config.framesInFlight = kFramesInFlight;
    auto renderer = lr::render::makeRenderer(config);
    auto sdlInfo = SDL_RendererInfo{};
    SDL_GetRendererInfo(sdlRenderer.get(), &sdlInfo);
    std::printf("backend %s: %s; SDL %s on %s\n", renderer->deviceInfo().backend.c_str(),
                renderer->deviceInfo().deviceName.c_str(), sdlInfo.name, SDL_GetCurrentVideoDriver());

    auto const source = lr::asset::makeFileSource(lr::asset::FileSourceConfig{});
    auto assetConfig = lr::asset::AssetLibraryConfig{};
    assetConfig.source = source.get();
    assetConfig.runtime = runtime.get();
    assetConfig.compressedFormats = renderer->deviceInfo().limits.compressedArrayFormats;
    assetConfig.watchFiles = true;
    auto const assets = options.gltf.empty() ? nullptr : lr::asset::makeAssetLibrary(assetConfig);
    auto const ids = lr::asset::makeSequentialIds();
    auto sceneReloads = SceneReloads{};
    auto scene = lr::examples::makeDemoScene();
    if (assets != nullptr) {
        auto loaded = lr::examples::viewableScene(*lr::examples::loadSceneNow(*assets, options.gltf), *ids);
        scene = lr::examples::DemoScene{std::move(loaded.delta), loaded.camera};
        sceneReloads.assets = assets.get();
        sceneReloads.path = options.gltf;
        sceneReloads.ids = ids.get();
        sceneReloads.shown = std::move(loaded.removal);
    }
    auto kernelReloads = kernelReloadsFor(*renderer, *runtime);
    auto const target = renderer->targets().createWindow(
        lr::render::WindowTargetDesc{outputSize(sdlRenderer.get()), presenter});
    auto const session = Session{*renderer, *presenter, window.get(), sdlRenderer.get(), target, scene.camera};
    auto reports = FrameReports{};
    auto scope = lexec::counting_scope{};
    auto const start = std::chrono::steady_clock::now();
    auto loop = LoopResult{};
    auto failure = std::exception_ptr{};
    try {
        loop = runLoop(session, options, std::move(scene.delta), Reloads{sceneReloads, kernelReloads}, reports, scope);
    } catch (...) {
        failure = std::current_exception();
    }
    static_cast<void>(lexec::sync_wait(scope.join()));
    if (presenter->blitLatest()) {
        ++loop.shown;
    }
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
    if (failure == nullptr) {
        failure = reports.firstFailure();
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    auto const frames = reports.count.load();
    std::printf("%llu frames in %.1f ms (%.1f per second), %llu shown\n", static_cast<unsigned long long>(frames),
                elapsed.count(), static_cast<double>(frames) * 1000.0 / elapsed.count(),
                static_cast<unsigned long long>(loop.shown));
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    SDL_SetMainReady();
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr, "usage: lrender_view [--backend cpu|opengl|gles|vulkan] [--device <name part>] "
                             "[--size <w>x<h>]\n"
                             "                    [--pipeline visibility|pathtrace] [--frames <n>] [--gltf <file>]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "lrender_view: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
