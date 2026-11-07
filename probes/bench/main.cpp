// lrender_bench: measures the renderer on a scene, in one or more configurations side by side.
//
//   lrender_bench [--backend <name>] [--device <name part>] [--size <w>x<h>]
//                 [--scene scale|demo|sponza|game|<file.gltf>] [--scenes <directory>]
//                 [--instances <n>] [--lights <n>] [--segments <n>]
//                 [--config <name>[:<setting>=<value>,...]]... [--camera orbit|still]
//                 [--warmup <frames>] [--frames <n>] [--timers on|off] [--json <file>]
//
// Every --config gets a renderer of its own on the same device. Their frames alternate, each
// waited for before the next starts, so that whatever the GPU's clocks do affects them alike.
// Settings: aliasing=on|off, subgroups=auto|off, singlepass=auto|off, rayqueries=auto|off,
// taa=on|off, pipeline=visibility|pathtrace. Before the --frames that count (a full turn of
// the orbit unless given), every configuration renders --warmup frames from where the orbit
// starts, then more until the medians of its last two windows of 30 frames agree within 2%
// (at most 600 more): the GPU's clocks have risen and the textures have streamed in. Frames
// go to a window target whose presenter drops them: the Readback path windows take. The
// scale scene is lr::examples::makeScaleScene's; sponza and game are the glTF sample
// scenes tools/fetch-scenes.sh downloads into --scenes (build/scenes unless given).

#include <lr/asset/AssetLibrary.h>
#include <lr/asset/AssetSource.h>
#include <lr/examples/CommandLine.h>
#include <lr/examples/DemoScene.h>
#include <lr/examples/GltfScene.h>
#include <lr/examples/ScaleScene.h>
#include <lr/render/Renderer.h>
#include <lr/runtime/Runtime.h>

#include <tc/device.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using lr::render::VariantUse;

constexpr auto kWindow = std::size_t{30};
constexpr auto kStableWithin = 0.02;
constexpr auto kMaxExtraWarmup = 600u;
constexpr auto kOrbitFrames = 240u;

struct Settings {
    bool aliasing = true;
    VariantUse subgroups = VariantUse::Auto;
    VariantUse singlePass = VariantUse::Auto;
    VariantUse rayQueries = VariantUse::Auto;
    bool taa = false;
    bool pathTrace = false;
};

struct ConfigSpec {
    std::string name;
    std::string text; // the settings as given
    Settings settings;
};

struct Options {
    std::string backend = "vulkan";
    std::string device;
    lr::core::ivec2 size{1920, 1080};
    std::string scene = "scale";
    std::string scenes = "build/scenes";
    lr::examples::ScaleSceneDesc scale;
    std::vector<ConfigSpec> configs;
    bool orbit = true;
    std::uint32_t warmup = 60;
    std::uint32_t frames = kOrbitFrames; // a full turn
    bool timers = true;
    std::string json;
};

// A decimal number, zero allowed.
std::optional<std::uint32_t> parseNumber(std::string const &text) {
    return text == "0" ? std::optional<std::uint32_t>{0u} : lr::examples::parseCount(text);
}

std::optional<VariantUse> parseUse(std::string const &value) {
    if (value == "auto") {
        return VariantUse::Auto;
    }
    if (value == "off") {
        return VariantUse::Off;
    }
    return std::nullopt;
}

std::optional<bool> parseSwitch(std::string const &value) {
    if (value == "on" or value == "off") {
        return value == "on";
    }
    return std::nullopt;
}

bool applySetting(Settings &settings, std::string const &key, std::string const &value) {
    auto const use = parseUse(value);
    auto const on = parseSwitch(value);
    if (key == "aliasing" and on) {
        settings.aliasing = *on;
    } else if (key == "subgroups" and use) {
        settings.subgroups = *use;
    } else if (key == "singlepass" and use) {
        settings.singlePass = *use;
    } else if (key == "rayqueries" and use) {
        settings.rayQueries = *use;
    } else if (key == "taa" and on) {
        settings.taa = *on;
    } else if (key == "pipeline" and (value == "visibility" or value == "pathtrace")) {
        settings.pathTrace = value == "pathtrace";
    } else {
        return false;
    }
    return true;
}

// "<name>[:<setting>=<value>,...]"
std::optional<ConfigSpec> parseConfig(std::string const &text) {
    auto const colon = text.find(':');
    auto spec = ConfigSpec{text.substr(0, colon), colon == std::string::npos ? "" : text.substr(colon + 1), {}};
    if (spec.name.empty()) {
        return std::nullopt;
    }
    auto settings = std::istringstream{spec.text};
    for (auto item = std::string{}; std::getline(settings, item, ',');) {
        auto const equals = item.find('=');
        if (equals == std::string::npos
            or not applySetting(spec.settings, item.substr(0, equals), item.substr(equals + 1))) {
            return std::nullopt;
        }
    }
    return spec;
}

bool applyCount(std::uint32_t &target, std::string const &value) {
    auto const number = parseNumber(value);
    target = number.value_or(target);
    return number.has_value();
}

bool apply(Options &options, std::string const &flag, std::string const &value) {
    if (flag == "--backend") {
        options.backend = value;
    } else if (flag == "--device") {
        options.device = value;
    } else if (flag == "--size") {
        auto const size = lr::examples::parseSize(value);
        options.size = size.value_or(options.size);
        return size.has_value();
    } else if (flag == "--scene") {
        options.scene = value;
    } else if (flag == "--scenes") {
        options.scenes = value;
    } else if (flag == "--instances") {
        return applyCount(options.scale.instances, value);
    } else if (flag == "--lights") {
        return applyCount(options.scale.lights, value);
    } else if (flag == "--segments") {
        return applyCount(options.scale.sphereSegments, value) and options.scale.sphereSegments >= 3;
    } else if (flag == "--config") {
        auto spec = parseConfig(value);
        if (spec) {
            options.configs.push_back(std::move(*spec));
        }
        return spec.has_value();
    } else if (flag == "--camera") {
        options.orbit = value == "orbit";
        return value == "orbit" or value == "still";
    } else if (flag == "--warmup") {
        return applyCount(options.warmup, value);
    } else if (flag == "--frames") {
        return applyCount(options.frames, value) and options.frames > 0;
    } else if (flag == "--timers") {
        auto const on = parseSwitch(value);
        options.timers = on.value_or(options.timers);
        return on.has_value();
    } else if (flag == "--json") {
        options.json = value;
    } else {
        return false;
    }
    return true;
}

std::optional<Options> parse(int const argc, char **const argv) {
    auto options = Options{};
    for (auto index = 1; index + 1 < argc; index += 2) {
        if (not apply(options, argv[index], argv[index + 1])) {
            return std::nullopt;
        }
    }
    if (argc % 2 == 0) {
        return std::nullopt;
    }
    if (options.configs.empty()) {
        options.configs.push_back(ConfigSpec{"default", "", Settings{}});
    }
    return options;
}

struct Summary {
    double median = 0.0;
    double p10 = 0.0;
    double p90 = 0.0;
    double mean = 0.0;
    std::size_t count = 0;
};

// Samples of one quantity, one per frame that had it.
class Series {
public:
    void add(double const value) { values.push_back(value); }
    void clear() noexcept { values.clear(); }
    std::size_t size() const noexcept { return values.size(); }

    std::optional<Summary> summary() const {
        if (values.empty()) {
            return std::nullopt;
        }
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        auto mean = 0.0;
        for (auto const value : sorted) {
            mean += value;
        }
        return Summary{quantile(sorted, 0.5), quantile(sorted, 0.1), quantile(sorted, 0.9),
                       mean / static_cast<double>(sorted.size()), sorted.size()};
    }

    // Whether the last two windows of `window` samples have medians within `within` of each other.
    bool settled(std::size_t const window, double const within) const {
        if (values.size() < 2 * window) {
            return false;
        }
        auto const end = values.end();
        auto const last = medianOf(std::vector<double>(end - static_cast<std::ptrdiff_t>(window), end));
        auto const before = medianOf(std::vector<double>(end - static_cast<std::ptrdiff_t>(2 * window),
                                                         end - static_cast<std::ptrdiff_t>(window)));
        return std::abs(last - before) <= within * before;
    }

private:
    static double quantile(std::vector<double> const &sorted, double const q) {
        auto const position = q * static_cast<double>(sorted.size() - 1);
        auto const low = static_cast<std::size_t>(position);
        auto const high = std::min(low + 1, sorted.size() - 1);
        auto const fraction = position - static_cast<double>(low);
        return sorted[low] + (sorted[high] - sorted[low]) * fraction;
    }

    static double medianOf(std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        return quantile(samples, 0.5);
    }

    std::vector<double> values;
};

struct DroppingPresenter final : lr::render::Presenter {
    void present(lr::render::PresentFrame const &) override {}
};

struct PassSeries {
    std::string pass;
    std::uint32_t dispatches = 0;
    Series milliseconds;
};

// What the counted frames of a configuration measured.
struct Measurements {
    Series gpu; // the sum of the pass times, of frames where every pass had one
    Series wall;
    Series prepare;
    Series record;
    Series finish;
    std::vector<PassSeries> passes; // in the order they first ran
    std::uint32_t clockChanges = 0; // frames left out of `gpu`: a pass had no time
    lr::render::ViewStats stats;    // of the last frame
};

// The device time of the frame: the sum of its passes, when each had a time.
std::optional<double> gpuTimeOf(lr::render::ViewResult const &view) {
    if (view.passTimings.empty()) {
        return std::nullopt;
    }
    auto total = 0.0;
    for (auto const &pass : view.passTimings) {
        if (not pass.milliseconds) {
            return std::nullopt;
        }
        total += *pass.milliseconds;
    }
    return total;
}

void measure(Measurements &into, lr::render::FrameResult const &frame) {
    auto const &view = frame.views.at(0);
    auto const gpu = gpuTimeOf(view);
    if (gpu) {
        into.gpu.add(*gpu);
    } else if (not view.passTimings.empty()) {
        ++into.clockChanges;
    }
    into.wall.add(frame.wallTime.count());
    into.prepare.add(frame.prepareTime.count());
    into.record.add(frame.recordTime.count());
    into.finish.add(frame.finishTime.count());
    for (auto const &timing : view.passTimings) {
        auto found = std::find_if(into.passes.begin(), into.passes.end(),
                                  [&timing](PassSeries const &pass) { return pass.pass == timing.pass; });
        if (found == into.passes.end()) {
            found = into.passes.insert(into.passes.end(), PassSeries{timing.pass, timing.dispatches, {}});
        }
        if (timing.milliseconds) {
            found->milliseconds.add(*timing.milliseconds);
        }
    }
    into.stats = view.stats;
}

// One configuration: its renderer, the view it renders and what it measured.
struct Run {
    ConfigSpec spec;
    std::unique_ptr<lr::render::Renderer> renderer;
    lr::render::ViewDesc view;
    Series warmup; // the frame time the warm-up waits to settle
    Measurements measured;
};

Run makeRun(ConfigSpec spec, Options const &options, lr::runtime::Runtime &runtime) {
    auto config = lr::render::RendererConfig{};
    config.backend = options.backend;
    config.deviceNameContains = options.device;
    config.runtime = &runtime;
    config.gpuTimers = options.timers;
    config.textureConversions = lr::render::ConversionDelivery::Deterministic;
    config.aliasGraphResources = spec.settings.aliasing;
    config.subgroups = spec.settings.subgroups;
    config.singlePassVisibility = spec.settings.singlePass;
    config.rayQueries = spec.settings.rayQueries;
    auto renderer = lr::render::makeRenderer(config);
    auto view = lr::render::ViewDesc{};
    view.id = lr::core::ViewId{0};
    view.target = renderer->targets().createWindow(
        lr::render::WindowTargetDesc{options.size, std::make_shared<DroppingPresenter>()});
    view.pipeline = spec.settings.pathTrace ? renderer->pipelines().pathTraceReference()
                                            : renderer->pipelines().visibility();
    view.settings.antiAliasing = spec.settings.taa ? lr::render::AntiAliasing::Taa : lr::render::AntiAliasing::None;
    return Run{std::move(spec), std::move(renderer), view, {}, {}};
}

lr::render::FrameResult render(Run &run, lr::core::RenderWorldDelta delta, lr::core::CameraDesc const &camera,
                               bool const cut) {
    auto view = run.view;
    view.camera = camera;
    view.cameraCut = cut;
    auto result = lexec::sync_wait(run.renderer->submit(lr::render::FrameDesc{std::move(delta), {view}}));
    if (not result) {
        throw std::runtime_error{"the frame was stopped"};
    }
    return std::get<0>(std::move(*result));
}

struct BenchScene {
    lr::core::RenderWorldDelta delta;
    lr::core::CameraDesc camera;
    std::string description;
};

std::string gltfPathOf(Options const &options) {
    if (options.scene == "sponza") {
        return options.scenes + "/khronos/Models/Sponza/glTF/Sponza.gltf";
    }
    if (options.scene == "game") {
        return options.scenes + "/khronos/Models/ABeautifulGame/glTF/ABeautifulGame.gltf";
    }
    return options.scene;
}

std::string described(lr::core::RenderWorldDelta const &delta) {
    auto triangles = std::uint64_t{0};
    auto trianglesOf = [&delta](lr::core::MeshId const id) {
        auto const found = std::find_if(delta.meshes.begin(), delta.meshes.end(),
                                        [id](lr::core::MeshUpsert const &mesh) { return mesh.id == id; });
        return found == delta.meshes.end() ? std::uint64_t{0} : found->mesh->indices.size() / 3u;
    };
    auto perMesh = std::vector<std::pair<lr::core::MeshId, std::uint64_t>>{};
    for (auto const &instance : delta.instances) {
        auto found = std::find_if(perMesh.begin(), perMesh.end(),
                                  [&instance](auto const &known) { return known.first == instance.mesh; });
        if (found == perMesh.end()) {
            found = perMesh.insert(perMesh.end(), {instance.mesh, trianglesOf(instance.mesh)});
        }
        triangles += found->second;
    }
    auto text = std::ostringstream{};
    text << delta.meshes.size() << " meshes, " << delta.instances.size() << " instances, " << triangles
         << " triangles, " << delta.materials.size() << " materials, " << delta.textures.size() << " textures, "
         << delta.lights.size() << " lights";
    return text.str();
}

BenchScene sceneOf(Options const &options, lr::render::Renderer &renderer, lr::runtime::Runtime &runtime) {
    if (options.scene == "scale") {
        auto scene = lr::examples::makeScaleScene(options.scale);
        auto description = described(scene.delta);
        return BenchScene{std::move(scene.delta), scene.camera, std::move(description)};
    }
    if (options.scene == "demo") {
        auto scene = lr::examples::makeDemoScene();
        auto description = described(scene.delta);
        return BenchScene{std::move(scene.delta), scene.camera, std::move(description)};
    }
    auto const path = gltfPathOf(options);
    auto const source = lr::asset::makeFileSource(lr::asset::FileSourceConfig{});
    auto config = lr::asset::AssetLibraryConfig{};
    config.source = source.get();
    config.runtime = &runtime;
    config.compressedFormats = renderer.deviceInfo().limits.compressedArrayFormats;
    auto const assets = lr::asset::makeAssetLibrary(config);
    auto const loaded = lr::examples::loadSceneNow(*assets, path);
    for (auto const &warning : loaded->warnings) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), warning.c_str());
    }
    auto const ids = lr::asset::makeSequentialIds();
    auto scene = lr::examples::viewableScene(*loaded, *ids);
    auto description = described(scene.delta);
    return BenchScene{std::move(scene.delta), scene.camera, std::move(description)};
}

// Around the camera's target, a full turn every kOrbitFrames frames.
lr::core::CameraDesc cameraAt(lr::core::CameraDesc const &start, std::uint32_t const frame, bool const orbit) {
    if (not orbit) {
        return start;
    }
    auto const angle = 6.283185307179586 * static_cast<double>(frame % kOrbitFrames) / kOrbitFrames;
    auto const offset = start.position - start.target;
    auto const c = static_cast<float>(std::cos(angle));
    auto const s = static_cast<float>(std::sin(angle));
    auto camera = start;
    camera.position = start.target + lr::core::vec3{c * offset.x + s * offset.z, offset.y, c * offset.z - s * offset.x};
    return camera;
}

// The GPU's performance state and clocks, where nvidia-smi reports them.
std::string gpuClocks(lr::render::Renderer const &renderer) {
    if (renderer.deviceInfo().deviceName.find("NVIDIA") == std::string::npos) {
        return "";
    }
    auto *const pipe = popen("nvidia-smi --query-gpu=pstate,clocks.gr,clocks.mem --format=csv,noheader 2>/dev/null",
                             "r");
    if (pipe == nullptr) {
        return "";
    }
    auto line = std::string(256, '\0');
    auto const read = std::fgets(line.data(), static_cast<int>(line.size()), pipe) != nullptr;
    pclose(pipe);
    line.resize(read ? line.find_first_of("\r\n\0", 0, 3) : 0);
    return line;
}

std::string quoted(std::string const &text) {
    auto out = std::string{"\""};
    for (auto const c : text) {
        if (c == '"' or c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out + "\"";
}

std::string number(double const value) {
    char text[32];
    std::snprintf(text, sizeof text, "%.4f", value);
    return text;
}

std::string json(std::optional<Summary> const &summary) {
    if (not summary) {
        return "null";
    }
    return "{\"median\": " + number(summary->median) + ", \"p10\": " + number(summary->p10)
         + ", \"p90\": " + number(summary->p90) + ", \"mean\": " + number(summary->mean)
         + ", \"count\": " + std::to_string(summary->count) + "}";
}

struct Report {
    Options const &options;
    std::string device;
    std::string scene;
    std::uint32_t warmupFrames = 0;
    bool settled = false;
    std::string clocksBefore;
    std::string clocksAfter;
};

std::string json(Report const &report, std::vector<Run> const &runs) {
    auto out = std::ostringstream{};
    out << "{\n  \"device\": " << quoted(report.device) << ",\n  \"backend\": " << quoted(report.options.backend)
        << ",\n  \"size\": [" << report.options.size.x << ", " << report.options.size.y << "]"
        << ",\n  \"scene\": " << quoted(report.options.scene) << ",\n  \"sceneContents\": " << quoted(report.scene)
        << ",\n  \"camera\": " << quoted(report.options.orbit ? "orbit" : "still")
        << ",\n  \"timers\": " << (report.options.timers ? "true" : "false")
        << ",\n  \"warmupFrames\": " << report.warmupFrames << ",\n  \"settled\": " << (report.settled ? "true" : "false")
        << ",\n  \"frames\": " << report.options.frames << ",\n  \"gpuClocks\": {\"before\": "
        << quoted(report.clocksBefore) << ", \"after\": " << quoted(report.clocksAfter) << "},\n  \"configs\": [";
    for (auto index = std::size_t{0}; index < runs.size(); ++index) {
        auto const &run = runs[index];
        auto const &measured = run.measured;
        out << (index == 0 ? "\n" : ",\n") << "    {\"name\": " << quoted(run.spec.name)
            << ", \"settings\": " << quoted(run.spec.text) << ",\n     \"frame\": {\"gpu\": " << json(measured.gpu.summary())
            << ",\n               \"wall\": " << json(measured.wall.summary())
            << ",\n               \"prepare\": " << json(measured.prepare.summary())
            << ",\n               \"record\": " << json(measured.record.summary())
            << ",\n               \"finish\": " << json(measured.finish.summary()) << "},\n     \"clockChanges\": "
            << measured.clockChanges << ", \"graphBytes\": " << measured.stats.graphBytes
            << ", \"unaliasedBytes\": " << measured.stats.unaliasedBytes
            << ", \"visibleInstances\": " << measured.stats.visibleInstances
            << ", \"visibleTriangles\": " << measured.stats.visibleTriangles << ",\n     \"passes\": [";
        for (auto pass = std::size_t{0}; pass < measured.passes.size(); ++pass) {
            auto const &series = measured.passes[pass];
            out << (pass == 0 ? "\n" : ",\n") << "       {\"pass\": " << quoted(series.pass)
                << ", \"dispatches\": " << series.dispatches << ", \"ms\": " << json(series.milliseconds.summary())
                << "}";
        }
        out << "]}";
    }
    out << "\n  ]\n}\n";
    return out.str();
}

void printSummary(char const *const label, std::optional<Summary> const &summary) {
    if (summary) {
        std::printf("  %-8s %9.3f ms  (p10 %.3f, p90 %.3f)\n", label, summary->median, summary->p10, summary->p90);
    }
}

void print(Report const &report, std::vector<Run> const &runs) {
    std::printf("%s, %dx%d: %s\n", report.device.c_str(), report.options.size.x, report.options.size.y,
                report.scene.c_str());
    std::printf("warm-up %u frames%s; %u frames counted; GPU %s -> %s\n", report.warmupFrames,
                report.settled ? "" : " (not settled)", report.options.frames, report.clocksBefore.c_str(),
                report.clocksAfter.c_str());
    for (auto const &run : runs) {
        auto const &measured = run.measured;
        std::printf("%s%s%s: %u instances and %u triangles visible, frame graph %.1f MiB\n", run.spec.name.c_str(),
                    run.spec.text.empty() ? "" : " ", run.spec.text.c_str(), measured.stats.visibleInstances,
                    measured.stats.visibleTriangles, static_cast<double>(measured.stats.graphBytes) / 1048576.0);
        printSummary("gpu", measured.gpu.summary());
        printSummary("wall", measured.wall.summary());
        printSummary("prepare", measured.prepare.summary());
        printSummary("record", measured.record.summary());
        printSummary("finish", measured.finish.summary());
        for (auto const &pass : measured.passes) {
            auto const summary = pass.milliseconds.summary();
            std::printf("    %-34s %2u %9.3f ms\n", pass.pass.c_str(), pass.dispatches, summary ? summary->median : 0.0);
        }
    }
}

// What the warm-up waits to settle: the device time where there are timers, else the wall time.
double frameTimeOf(lr::render::FrameResult const &frame) {
    return gpuTimeOf(frame.views.at(0)).value_or(frame.wallTime.count());
}

int run(Options const &options) {
    auto runtime = lr::runtime::makeRuntime(lr::runtime::RuntimeConfig{});
    auto runs = std::vector<Run>{};
    for (auto const &spec : options.configs) {
        runs.push_back(makeRun(spec, options, *runtime));
    }
    auto scene = sceneOf(options, *runs.front().renderer, *runtime);
    auto report = Report{options, runs.front().renderer->deviceInfo().deviceName, scene.description, 0, false, "", ""};

    auto settled = [&runs] {
        return std::all_of(runs.begin(), runs.end(),
                           [](Run const &run) { return run.warmup.settled(kWindow, kStableWithin); });
    };
    for (auto frame = 0u; frame < options.warmup + kMaxExtraWarmup; ++frame) {
        for (auto &run : runs) {
            auto delta = frame == 0 ? scene.delta : lr::core::RenderWorldDelta{};
            run.warmup.add(frameTimeOf(render(run, std::move(delta), scene.camera, false)));
        }
        report.warmupFrames = frame + 1;
        if (report.warmupFrames >= options.warmup and settled()) {
            report.settled = true;
            break;
        }
    }

    report.clocksBefore = gpuClocks(*runs.front().renderer);
    for (auto frame = 0u; frame < options.frames; ++frame) {
        for (auto &run : runs) {
            measure(run.measured, render(run, {}, cameraAt(scene.camera, frame, options.orbit), frame == 0));
        }
    }
    report.clocksAfter = gpuClocks(*runs.front().renderer);

    print(report, runs);
    if (not options.json.empty()) {
        auto *const file = std::fopen(options.json.c_str(), "w");
        auto const text = json(report, runs);
        if (file == nullptr or std::fputs(text.c_str(), file) < 0 or std::fclose(file) != 0) {
            std::fprintf(stderr, "cannot write %s\n", options.json.c_str());
            return EXIT_FAILURE;
        }
        std::printf("-> %s\n", options.json.c_str());
    }
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr,
                     "usage: lrender_bench [--backend <name>] [--device <name part>] [--size <w>x<h>]\n"
                     "                     [--scene scale|demo|sponza|game|<file.gltf>] [--scenes <directory>]\n"
                     "                     [--instances <n>] [--lights <n>] [--segments <n>]\n"
                     "                     [--config <name>[:<setting>=<value>,...]]... [--camera orbit|still]\n"
                     "                     [--warmup <frames>] [--frames <n>] [--timers on|off] [--json <file>]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "lrender_bench: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
