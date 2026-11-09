// lrender_view: shows the demo scene, a glTF file's or a mesh in an SDL window, built in a
// lr::scene World that scripts change every frame.
//
//   lrender_view [--backend cpu|opengl|gles|vulkan] [--device <name part>] [--size <w>x<h>]
//                [--pipeline visibility|pathtrace|pathtrace-interactive] [--frames <n>]
//                [--gltf <file> | --mesh <file>] [--present sdl|vulkan]
//
// --present vulkan shows the frames through lr::present::VulkanPresenter on the window's
// Vulkan surface, on the SharedBuffer path where the device is the same GPU; sdl (the
// default) blits them through an SDL renderer, on the Readback path.
// The camera circles slowly; drag with the left mouse button to turn it and scroll to zoom.
// Tab steps through the visibility, the path tracing and the denoised path tracing
// pipelines (1, 2 and 3 pick one);
// T turns the visibility pipeline's temporal anti-aliasing off and on (on to begin with);
// Esc or Q quits. With --frames it quits after that many frames, simulating 60 frames a
// second, for runs nobody watches (SDL_VIDEODRIVER=offscreen). A --gltf scene is loaded
// again when its files change; --mesh runs docs/architecture.md's orbitCamera script.
// Development builds on a GPU backend also reload lrender's kernels when the build
// rewrites them.

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <lr/asset/AssetLibrary.h>
#include <lr/asset/AssetSource.h>
#include <lr/asset/FileWatcher.h>
#include <lr/examples/CommandLine.h>
#include <lr/examples/DemoScene.h>
#include <lr/examples/GltfScene.h>
#include <lr/examples/Scripts.h>
#include <lr/present/SdlPresenter.h>
#ifdef LR_VIEW_VULKAN_PRESENTER
#include <lr/present/VulkanPresenter.h>

#include <SDL_vulkan.h>
#endif
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>
#include <lr/scene/ScriptHost.h>
#include <lr/scene/World.h>

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
#include <utility>
#include <vector>

namespace {

using lr::core::ivec2;
using lr::core::vec3;

constexpr auto kFramesInFlight = std::uint32_t{2};

// The built-in pipelines, in the order the number keys pick them.
enum class Pipeline { Visibility, PathTrace, PathTraceInteractive };

std::optional<Pipeline> pipelineNamed(std::string const &name) {
    if (name == "visibility") {
        return Pipeline::Visibility;
    } else if (name == "pathtrace") {
        return Pipeline::PathTrace;
    } else if (name == "pathtrace-interactive") {
        return Pipeline::PathTraceInteractive;
    }
    return std::nullopt;
}

struct Options {
    std::string backend = "cpu";
    std::string device;
    ivec2 size{960, 540};
    Pipeline pipeline = Pipeline::Visibility;
    std::uint32_t frames = 0; // 0: until the window is closed
    std::string gltf;
    std::string mesh;
    bool vulkanPresenter = false;
};

bool apply(Options &options, std::string const &flag, std::string const &value) {
    if (flag == "--present") {
        options.vulkanPresenter = value == "vulkan";
        return value == "vulkan" or value == "sdl";
    } else if (flag == "--gltf") {
        options.gltf = value;
    } else if (flag == "--mesh") {
        options.mesh = value;
    } else if (flag == "--backend") {
        options.backend = value;
    } else if (flag == "--device") {
        options.device = value;
    } else if (flag == "--size") {
        auto const size = lr::examples::parseSize(value);
        options.size = size.value_or(ivec2{0});
        return size.has_value();
    } else if (flag == "--pipeline") {
        auto const pipeline = pipelineNamed(value);
        options.pipeline = pipeline.value_or(Pipeline::Visibility);
        return pipeline.has_value();
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
    if (not options.gltf.empty() and not options.mesh.empty()) {
        return std::nullopt;
    }
    return options;
}

struct ViewerInput {
    Pipeline pipeline = Pipeline::Visibility;
    bool taa = true;
    bool quit = false;
    bool resized = false;
};

void handleKey(SDL_Keycode const key, ViewerInput &input) {
    if (key == SDLK_ESCAPE or key == SDLK_q) {
        input.quit = true;
    } else if (key == SDLK_TAB) {
        input.pipeline = static_cast<Pipeline>((static_cast<int>(input.pipeline) + 1) % 3);
    } else if (key == SDLK_t) {
        input.taa = not input.taa;
    } else if (key == SDLK_1) {
        input.pipeline = Pipeline::Visibility;
    } else if (key == SDLK_2) {
        input.pipeline = Pipeline::PathTrace;
    } else if (key == SDLK_3) {
        input.pipeline = Pipeline::PathTraceInteractive;
    }
}

void handle(SDL_Event const &event, ViewerInput &input, lr::examples::Orbit &orbit) {
    constexpr auto kRadiansPerPixel = 0.01f;
    constexpr auto kPitchLimit = 1.5f;
    if (event.type == SDL_QUIT) {
        input.quit = true;
    } else if (event.type == SDL_KEYDOWN) {
        handleKey(event.key.keysym.sym, input);
    } else if (event.type == SDL_MOUSEMOTION and (event.motion.state & SDL_BUTTON_LMASK) != 0) {
        orbit.yaw -= static_cast<float>(event.motion.xrel) * kRadiansPerPixel;
        orbit.pitch = std::clamp(orbit.pitch + static_cast<float>(event.motion.yrel) * kRadiansPerPixel, -kPitchLimit,
                                 kPitchLimit);
    } else if (event.type == SDL_MOUSEWHEEL) {
        orbit.distance = std::clamp(orbit.distance * std::pow(0.9f, event.wheel.preciseY), 0.5f, 50.0f);
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
        if (not result.lostTextures.empty()) {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            lostTextures.insert(lostTextures.end(), result.lostTextures.begin(), result.lostTextures.end());
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

    // The textures a lost device held, for the world to upsert again.
    std::vector<lr::core::TextureId> takeLostTextures() {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return std::exchange(lostTextures, {});
    }

    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint32_t> samples{0};

private:
    std::mutex mutex;
    std::exception_ptr failure;
    std::vector<lr::core::TextureId> lostTextures;
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

// How the window shows the frames.
struct WindowOutput {
    virtual ~WindowOutput() = default;

    virtual std::shared_ptr<lr::render::Presenter> presenter() const = 0;
    // The window's size in pixels.
    virtual ivec2 size() const = 0;
    // On the main thread: whether a frame came to the window since the last call.
    virtual bool show() = 0;
    virtual std::string describe() const = 0;
};

// Through an SDL renderer, which the main thread blits the newest frame with.
struct SdlOutput final : WindowOutput {
    explicit SdlOutput(SDL_Window *const window)
        : renderer(makeSdlRenderer(window)),
          frames(lr::present::makeSdlPresenter(lr::present::SdlPresenterConfig{renderer.get()})) {}

    std::shared_ptr<lr::render::Presenter> presenter() const override { return frames; }

    ivec2 size() const override {
        auto width = 0;
        auto height = 0;
        if (SDL_GetRendererOutputSize(renderer.get(), &width, &height) != 0 or width <= 0 or height <= 0) {
            throw std::runtime_error{std::string{"SDL_GetRendererOutputSize: "} + SDL_GetError()};
        }
        return ivec2{width, height};
    }

    bool show() override { return frames->blitLatest().has_value(); }

    std::string describe() const override {
        auto info = SDL_RendererInfo{};
        SDL_GetRendererInfo(renderer.get(), &info);
        return std::string{"SDL "} + info.name + " on " + SDL_GetCurrentVideoDriver();
    }

private:
    SdlRendererHandle renderer;
    std::shared_ptr<lr::present::SdlPresenter> frames;
};

#ifdef LR_VIEW_VULKAN_PRESENTER
// Through a Vulkan swapchain of the window's surface, presented on the presenter's thread.
struct VulkanOutput final : WindowOutput {
    VulkanOutput(SDL_Window *const window_, std::string const &device) : window(window_) {
        auto count = 0u;
        if (SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr) != SDL_TRUE) {
            throw std::runtime_error{std::string{"SDL_Vulkan_GetInstanceExtensions: "} + SDL_GetError()};
        }
        auto names = std::vector<char const *>(count);
        SDL_Vulkan_GetInstanceExtensions(window, &count, names.data());
        auto config = lr::present::VulkanPresenterConfig{};
        config.instanceExtensions.assign(names.begin(), names.end());
        config.createSurface = [window = window](VkInstance const instance) {
            auto surface = VkSurfaceKHR{VK_NULL_HANDLE};
            return SDL_Vulkan_CreateSurface(window, instance, &surface) == SDL_TRUE ? surface : VK_NULL_HANDLE;
        };
        config.deviceNameContains = device;
        frames = lr::present::makeVulkanPresenter(config);
    }

    std::shared_ptr<lr::render::Presenter> presenter() const override { return frames; }

    ivec2 size() const override {
        auto width = 0;
        auto height = 0;
        SDL_Vulkan_GetDrawableSize(window, &width, &height);
        return ivec2{std::max(width, 1), std::max(height, 1)};
    }

    bool show() override {
        auto const shown = frames->stats().shown;
        return std::exchange(lastShown, shown) != shown;
    }

    std::string describe() const override { return "Vulkan presenter on " + frames->deviceName(); }

private:
    SDL_Window *window;
    std::shared_ptr<lr::present::VulkanPresenter> frames;
    std::uint64_t lastShown = 0;
};
#endif

std::unique_ptr<WindowOutput> outputFor(Options const &options, SDL_Window *const window) {
#ifdef LR_VIEW_VULKAN_PRESENTER
    if (options.vulkanPresenter) {
        return std::make_unique<VulkanOutput>(window, options.device);
    }
#else
    if (options.vulkanPresenter) {
        throw std::runtime_error{"this build has no lr::present_vulkan"};
    }
#endif
    return std::make_unique<SdlOutput>(window);
}

struct Session {
    lr::render::Renderer &renderer;
    WindowOutput &output;
    SDL_Window *window;
    lr::render::RenderTargetId target;
};

// The world the scripts change and the frames show, seen through its camera entity.
struct Simulation {
    lr::scene::World &world;
    lr::scene::ScriptHost &scripts;
    lr::examples::Orbit &orbit;
    lr::scene::EntityId camera;
};

using Clock = std::chrono::steady_clock;

// Simulation time for each tick: a 60th of a second for counted runs, otherwise the time
// since the previous tick, at most a tenth of a second.
struct SimulationClock {
    lr::scene::SimDuration step() {
        auto const now = Clock::now();
        auto const elapsed = std::chrono::duration_cast<lr::scene::SimDuration>(now - last);
        last = now;
        return fixed ? lr::scene::SimDuration{16'666'667} : std::min(elapsed, lr::scene::SimDuration{100'000'000});
    }

    bool fixed;
    Clock::time_point last = Clock::now();
};

struct LoopResult {
    std::uint64_t submitted = 0;
    std::uint64_t shown = 0;
    std::optional<vec3> firstCamera;
    vec3 lastCamera{0.0f};
};

// The build writes several files for each kernel; they are reloaded once it has been
// quiet this long.
constexpr auto kKernelsQuiet = std::chrono::milliseconds{300};

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

// The frame's one view has its camera; the rest comes from the input.
void submitFrame(Session const &session, ViewerInput const &input, lr::render::FrameDesc frame, FrameReports &reports,
                 lexec::counting_scope &scope) {
    auto &pipelines = session.renderer.pipelines();
    auto &view = frame.views.at(0);
    view.id = lr::core::ViewId{0};
    view.target = session.target;
    switch (input.pipeline) {
    case Pipeline::Visibility:
        view.pipeline = pipelines.visibility();
        break;
    case Pipeline::PathTrace:
        view.pipeline = pipelines.pathTraceReference();
        break;
    case Pipeline::PathTraceInteractive:
        view.pipeline = pipelines.pathTraceInteractive();
        break;
    }
    view.settings.antiAliasing = input.taa ? lr::render::AntiAliasing::Taa : lr::render::AntiAliasing::None;
    lexec::spawn(session.renderer.submit(std::move(frame))
                     | lexec::then([&reports](lr::render::FrameResult const &result) noexcept { reports.finished(result); })
                     | lexec::upon_error([&reports](std::exception_ptr const &error) noexcept { reports.failed(error); })
                     | lexec::upon_stopped([&reports]() noexcept {
                           reports.failed(std::make_exception_ptr(std::runtime_error{"a frame was stopped"}));
                       }),
                 scope.get_token());
}

// Ticks the scripts and takes what they changed in the world, seen through its camera.
lr::render::FrameDesc simulate(Simulation const &simulation, SimulationClock &clock, FrameReports &reports,
                               LoopResult &result) {
    simulation.world.resources().resubmitTextures(reports.takeLostTextures());
    simulation.scripts.tick(clock.step());
    auto frame = lr::render::FrameDesc{};
    frame.delta = simulation.world.extractDelta();
    auto &view = frame.views.emplace_back();
    view.camera = simulation.world.cameraOf(simulation.camera);
    if (not result.firstCamera) {
        result.firstCamera = view.camera.position;
    }
    result.lastCamera = view.camera.position;
    return frame;
}

void showStatus(Session const &session, ViewerInput const &input, FrameReports const &reports) {
    auto const samples = std::to_string(reports.samples.load()) + " samples";
    auto const title = input.pipeline == Pipeline::PathTrace ? "lrender: path tracing, " + samples
                     : input.pipeline == Pipeline::PathTraceInteractive
                         ? "lrender: path tracing, denoised, " + samples
                         : std::string{"lrender: visibility buffer, TAA "} + (input.taa ? "on" : "off");
    SDL_SetWindowTitle(session.window, title.c_str());
}

// Keeps at most kFramesInFlight frames outstanding and shows the newest each turn, never
// waiting for a frame. Each frame submitted is a tick of the scripts.
LoopResult runLoop(Session const &session, Options const &options, Simulation const &simulation,
                   KernelReloads &kernels, FrameReports &reports, lexec::counting_scope &scope) {
    auto input = ViewerInput{options.pipeline, true, false, false};
    auto clock = SimulationClock{options.frames != 0};
    auto result = LoopResult{};
    while (not input.quit and not reports.firstFailure()) {
        watchKernels(kernels, session.renderer, scope);
        auto event = SDL_Event{};
        while (SDL_PollEvent(&event) != 0) {
            handle(event, input, simulation.orbit);
        }
        if (input.resized) {
            session.renderer.targets().resize(session.target, session.output.size());
            input.resized = false;
        }
        auto const finished = reports.count.load(std::memory_order_acquire);
        if (options.frames != 0 and finished >= options.frames) {
            break;
        }
        auto const wanted = options.frames == 0 or result.submitted < options.frames;
        auto const submitting = wanted and result.submitted - finished < kFramesInFlight;
        if (submitting) {
            submitFrame(session, input, simulate(simulation, clock, reports, result), reports, scope);
            ++result.submitted;
        }
        auto const shown = session.output.show();
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

// The world's scene and the scripts that run it: the demo scene with a camera steered
// around it and a bobbing lantern, a glTF file's scene, or a mesh the camera circles.
lr::scene::EntityId startScene(Options const &options, lr::scene::World &world, lr::scene::ScriptHost &scripts,
                               lr::asset::AssetLibrary &assets, lr::examples::Orbit &orbit) {
    if (options.gltf.empty() and options.mesh.empty()) {
        auto const demo = lr::examples::addDemoScene(world);
        orbit = lr::examples::orbitOf(demo.view);
        scripts.start(lr::examples::steerCamera(world, orbit, demo.camera));
        scripts.start(lr::examples::bob(world, demo.lantern));
        return demo.camera;
    }
    auto const camera = world.createEntity(lr::core::Transform::identity(), std::nullopt);
    world.setCamera(camera, lr::scene::CameraLens{});
    if (not options.gltf.empty()) {
        scripts.start(lr::examples::showGltf(world, assets, options.gltf, orbit, camera));
        scripts.start(lr::examples::steerCamera(world, orbit, camera));
        return camera;
    }
    auto sun = lr::core::DirectionalLight{};
    sun.direction = vec3{-0.4f, -1.0f, -0.6f};
    sun.illuminance = 2.5f;
    sun.castsShadows = true;
    world.setLight(world.createEntity(lr::core::Transform::identity(), std::nullopt), sun);
    world.setEnvironment(lr::examples::skyForScenes());
    scripts.start(lr::examples::orbitCamera(world, assets, camera, options.mesh));
    return camera;
}

// Script failures are bugs in the scripts: reported, and the program fails.
bool reportScriptFailures(lr::scene::ScriptHost &scripts) {
    auto const failures = scripts.takeFailures();
    for (auto const &failure : failures) {
        try {
            std::rethrow_exception(failure.error);
        } catch (std::exception const &error) {
            std::fprintf(stderr, "script %u failed: %s\n", failure.script.value, error.what());
        } catch (...) {
            std::fprintf(stderr, "script %u failed\n", failure.script.value);
        }
    }
    return failures.empty();
}

int run(Options const &options) {
    auto const video = SdlVideo{};
    auto const flags = std::uint32_t{SDL_WINDOW_RESIZABLE}
                     | (options.vulkanPresenter ? std::uint32_t{SDL_WINDOW_VULKAN} : std::uint32_t{0});
    auto const window = WindowHandle{SDL_CreateWindow("lrender", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                                      options.size.x, options.size.y, flags)};
    if (window == nullptr) {
        throw std::runtime_error{std::string{"SDL_CreateWindow: "} + SDL_GetError()};
    }
    auto const output = outputFor(options, window.get());
    auto runtime = lr::runtime::makeRuntime(lr::runtime::RuntimeConfig{});
    auto config = lr::render::RendererConfig{};
    config.backend = options.backend;
    config.deviceNameContains = options.device;
    config.runtime = runtime.get();
    config.framesInFlight = kFramesInFlight;
    auto renderer = lr::render::makeRenderer(config);
    std::printf("backend %s: %s; %s\n", renderer->deviceInfo().backend.c_str(),
                renderer->deviceInfo().deviceName.c_str(), output->describe().c_str());

    auto const source = lr::asset::makeFileSource(lr::asset::FileSourceConfig{});
    auto assetConfig = lr::asset::AssetLibraryConfig{};
    assetConfig.source = source.get();
    assetConfig.runtime = runtime.get();
    assetConfig.compressedFormats = renderer->deviceInfo().limits.compressedArrayFormats;
    assetConfig.watchFiles = true;
    auto const assets = lr::asset::makeAssetLibrary(assetConfig);
    auto worldConfig = lr::scene::WorldConfig{};
    worldConfig.runtime = runtime.get();
    auto const world = lr::scene::makeWorld(worldConfig);
    auto orbit = lr::examples::Orbit{};
    // Destroyed first: it stops the scripts, which use the world, the orbit and the assets.
    auto const scripts = lr::scene::makeScriptHost(lr::scene::ScriptHostConfig{});
    auto const camera = startScene(options, *world, *scripts, *assets, orbit);

    auto kernelReloads = kernelReloadsFor(*renderer, *runtime);
    auto const target =
        renderer->targets().createWindow(lr::render::WindowTargetDesc{output->size(), output->presenter()});
    auto const session = Session{*renderer, *output, window.get(), target};
    auto reports = FrameReports{};
    auto scope = lexec::counting_scope{};
    auto const start = Clock::now();
    auto loop = LoopResult{};
    auto failure = std::exception_ptr{};
    try {
        loop = runLoop(session, options, Simulation{*world, *scripts, orbit, camera}, kernelReloads, reports, scope);
    } catch (...) {
        failure = std::current_exception();
    }
    static_cast<void>(lexec::sync_wait(scope.join()));
    if (output->show()) {
        ++loop.shown;
    }
    auto const elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start);
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
    auto const from = loop.firstCamera.value_or(loop.lastCamera);
    std::printf("%llu ticks, %.2f s simulated, %zu entities; the camera went from (%.2f, %.2f, %.2f) to (%.2f, %.2f, "
                "%.2f)\n",
                static_cast<unsigned long long>(scripts->tickCount()),
                std::chrono::duration<double>{scripts->now()}.count(), world->entityCount(),
                static_cast<double>(from.x), static_cast<double>(from.y), static_cast<double>(from.z),
                static_cast<double>(loop.lastCamera.x), static_cast<double>(loop.lastCamera.y),
                static_cast<double>(loop.lastCamera.z));
    return reportScriptFailures(*scripts) ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
    SDL_SetMainReady();
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr, "usage: lrender_view [--backend cpu|opengl|gles|vulkan] [--device <name part>] "
                             "[--size <w>x<h>]\n"
                             "                    [--pipeline visibility|pathtrace|pathtrace-interactive] "
                             "[--frames <n>] [--gltf <file> | --mesh <file>]\n"
                             "                    [--present sdl|vulkan]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "lrender_view: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
