// layout_probe: times the scene tables' read patterns with each candidate layout on one
// device, and prints the device times.
//
//   layout_probe [--backend cpu|opengl|gles|vulkan] [--device <name part>] [--rounds <n>] [--scale <s>]
//
// --scale multiplies the table sizes (1: 2^18 instances, a sphere of 2^20 triangles).
// The layouts of one read pattern take turns, dispatch by dispatch, so clock changes of
// the device affect them alike.

#include "LayoutKernels.h"

#include <lr/core/Math.h>
#include <lr/core/Primitives.h>

#include <tc/devices.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace lr;

struct Options {
    std::string backend = "cpu";
    std::string device;
    int rounds = 20;
    double scale = 1.0;
};

std::optional<Options> parse(int const argc, char **const argv) {
    auto options = Options{};
    if (argc % 2 == 0) {
        return std::nullopt;
    }
    for (auto index = 1; index + 1 < argc; index += 2) {
        auto const flag = std::string{argv[index]};
        auto const value = std::string{argv[index + 1]};
        if (flag == "--backend") {
            options.backend = value;
        } else if (flag == "--device") {
            options.device = value;
        } else if (flag == "--rounds") {
            options.rounds = std::atoi(value.c_str());
        } else if (flag == "--scale") {
            options.scale = std::atof(value.c_str());
        } else {
            return std::nullopt;
        }
    }
    if (options.rounds <= 0 or options.scale <= 0.0) {
        return std::nullopt;
    }
    return options;
}

// One layout of a read pattern: binds its kernel and resources and dispatches it once.
struct Case {
    std::string layout;
    std::function<void()> dispatch;
};

struct Timing {
    double median = 0.0;
    double minimum = 0.0;
};

// Some drivers (llvmpipe's OpenGL) answer every timer query with 0.
bool measured(Timing const &timing) { return timing.minimum > 0.0; }

// The device times of each case over `rounds` rounds, after as many rounds again that let
// the device's clocks and caches settle. Timings come in execution order, so the k-th
// belongs to case k modulo the number of cases.
std::vector<Timing> timeInTurns(tc::Device &device, std::vector<Case> const &cases, int const rounds) {
    auto const runRounds = [&cases](int const count) {
        for (auto round = 0; round < count; ++round) {
            for (auto const &entry : cases) {
                entry.dispatch();
            }
        }
    };
    runRounds(std::max(rounds, 10));
    static_cast<void>(device.takeTimings());
    runRounds(rounds);
    auto const timings = device.takeTimings();
    auto times = std::vector<std::vector<double>>(cases.size());
    for (auto index = std::size_t{0}; index < timings.size(); ++index) {
        if (timings[index].milliseconds) {
            times[index % cases.size()].push_back(*timings[index].milliseconds);
        }
    }
    auto result = std::vector<Timing>{};
    for (auto &values : times) {
        std::sort(values.begin(), values.end());
        result.push_back(values.empty() ? Timing{} : Timing{values[values.size() / 2], values.front()});
    }
    return result;
}

void report(tc::Device &device, std::string const &title, std::vector<Case> const &cases, int const rounds) {
    auto const timings = timeInTurns(device, cases, rounds);
    std::printf("%-58s median ms     min ms\n", title.c_str());
    for (auto index = std::size_t{0}; index < cases.size(); ++index) {
        if (measured(timings[index])) {
            std::printf("  %-56s %9.3f %10.3f\n", cases[index].layout.c_str(), timings[index].median,
                        timings[index].minimum);
        } else {
            std::printf("  %-56s   the device reports no time\n", cases[index].layout.c_str());
        }
    }
}

tc::uvec3 groupsFor(std::size_t const items) { return tc::uvec3{static_cast<tc::uint>((items + 63) / 64), 1, 1}; }

core::mat4 randomPlacement(std::mt19937 &random) {
    auto place = std::uniform_real_distribution<float>{-60.0f, 60.0f};
    auto angle = std::uniform_real_distribution<float>{0.0f, 6.28f};
    auto transform = core::Transform{};
    transform.translation = core::vec3{place(random), place(random), place(random)};
    transform.rotation = core::Rotation::axisAngle(core::vec3{place(random), place(random), 1.0f}, angle(random));
    return core::matrixOf(transform);
}

// What the culling pass reads.
struct InstanceData {
    std::vector<render::GpuInstance> instances;
    std::shared_ptr<tc::TableResource<render::GpuMesh>> meshes;
    std::shared_ptr<tc::BufferResource<tc::uint>> results;
};

template <class T, class Layout>
std::shared_ptr<tc::TableResource<T, Layout>> uploadedTable(tc::Device &device, std::vector<T> const &values) {
    auto table = std::make_shared<tc::TableResource<T, Layout>>(values.size());
    for (auto index = std::size_t{0}; index < values.size(); ++index) {
        table->set(index, values[index]);
    }
    device.uploadTable(*table);
    return table;
}

template <class Layout>
Case cullCase(std::string layout, tc::Device &device, InstanceData const &data) {
    auto const table = uploadedTable<render::GpuInstance, Layout>(device, data.instances);
    auto const kernel = std::make_shared<probes::CullReads<Layout>>();
    kernel->instances.attach(*table);
    kernel->meshes.attach(*data.meshes);
    kernel->results.attach(*data.results);
    kernel->viewProjection = core::perspective(1.0f, 1.0f, 0.1f, 200.0f)
                           * core::lookAt(core::vec3{0.0f, 0.0f, 80.0f}, core::vec3{0.0f}, core::vec3{0.0f, 1.0f, 0.0f});
    kernel->count = static_cast<tc::uint>(data.instances.size());
    auto const groups = groupsFor(data.instances.size());
    return Case{std::move(layout), [&device, table, kernel, meshes = data.meshes, results = data.results, groups] {
                    device.useKernel(*kernel);
                    device.bindTable(kernel->instances);
                    device.bindTable(kernel->meshes);
                    device.bindBuffer(kernel->results);
                    device.setUniform(kernel->viewProjection);
                    device.setUniform(kernel->count);
                    device.execute(*kernel, groups);
                }};
}

void probeInstances(tc::Device &device, Options const &options) {
    auto random = std::mt19937{1};
    auto data = InstanceData{{}, std::make_shared<tc::TableResource<render::GpuMesh>>(64),
                             std::make_shared<tc::BufferResource<tc::uint>>(1)};
    auto extent = std::uniform_real_distribution<float>{0.2f, 3.0f};
    for (auto index = std::size_t{0}; index < data.meshes->size(); ++index) {
        auto const half = core::vec3{extent(random), extent(random), extent(random)};
        data.meshes->set(index, render::GpuMesh{0, 0, static_cast<tc::uint>(index + 1), -half, half, 0, 0, 0});
    }
    device.uploadTable(*data.meshes);
    auto const count = static_cast<std::size_t>(262144.0 * options.scale);
    for (auto index = std::size_t{0}; index < count; ++index) {
        auto const toWorld = randomPlacement(random);
        data.instances.push_back(
            render::GpuInstance{toWorld, tc::inverse(toWorld), static_cast<tc::uint>(index % data.meshes->size()), 0, toWorld});
    }
    data.results = std::make_shared<tc::BufferResource<tc::uint>>(static_cast<std::int32_t>(count));
    device.uploadBuffer(*data.results);
    auto const cases = std::vector<Case>{cullCase<tc::Aos>("Aos", device, data), cullCase<tc::Soa<>>("Soa<>", device, data)};
    report(device, "instance culling, " + std::to_string(count) + " instances of 25 words", cases, options.rounds);
}

// What the vertex read patterns read: a sphere's triangles, whose neighbours share
// vertices as in real meshes.
struct VertexData {
    std::vector<render::GpuVertex> vertices;
    std::shared_ptr<tc::BufferResource<tc::uint>> indices;
    std::shared_ptr<tc::BufferResource<float>> results;
    std::size_t triangles = 0;
};

template <template <class> class Reads, class Layout>
Case vertexCase(std::string layout, tc::Device &device, VertexData const &data) {
    auto const table = uploadedTable<render::GpuVertex, Layout>(device, data.vertices);
    auto const kernel = std::make_shared<Reads<Layout>>();
    kernel->vertices.attach(*table);
    kernel->indices.attach(*data.indices);
    kernel->results.attach(*data.results);
    kernel->count = static_cast<tc::uint>(data.triangles);
    auto const groups = groupsFor(data.triangles);
    return Case{std::move(layout), [&device, table, kernel, indices = data.indices, results = data.results, groups] {
                    device.useKernel(*kernel);
                    device.bindTable(kernel->vertices);
                    device.bindBuffer(kernel->indices);
                    device.bindBuffer(kernel->results);
                    device.setUniform(kernel->count);
                    device.execute(*kernel, groups);
                }};
}

VertexData sphereData(tc::Device &device, Options const &options) {
    auto const segments = static_cast<std::uint32_t>(1024.0 * std::sqrt(options.scale));
    auto const mesh = core::makeSphere(1.0f, segments, segments / 2);
    auto data = VertexData{};
    for (auto index = std::size_t{0}; index < mesh.positions.size(); ++index) {
        data.vertices.push_back(render::GpuVertex{mesh.positions[index], mesh.normals[index], mesh.uvs[index]});
    }
    data.triangles = mesh.indices.size() / 3;
    data.indices = std::make_shared<tc::BufferResource<tc::uint>>(static_cast<std::int32_t>(mesh.indices.size()));
    std::copy(mesh.indices.begin(), mesh.indices.end(), data.indices->begin());
    device.uploadBuffer(*data.indices);
    data.results = std::make_shared<tc::BufferResource<float>>(static_cast<std::int32_t>(data.triangles));
    device.uploadBuffer(*data.results);
    return data;
}

template <template <class> class Reads>
std::vector<Case> vertexCases(tc::Device &device, VertexData const &data) {
    return std::vector<Case>{vertexCase<Reads, tc::Aos>("Aos (5 words)", device, data),
                             vertexCase<Reads, tc::Soa<>>("Soa<>", device, data),
                             vertexCase<Reads, probes::PaddedAos>("PaddedAos (8 words)", device, data)};
}

void probeVertices(tc::Device &device, Options const &options) {
    auto const data = sphereData(device, options);
    auto const size = std::to_string(data.triangles) + " triangles, " + std::to_string(data.vertices.size()) + " vertices";
    report(device, "corner positions, " + size, vertexCases<probes::PositionReads>(device, data), options.rounds);
    report(device, "whole corners, " + size, vertexCases<probes::VertexReads>(device, data), options.rounds);
}

int run(Options const &options) {
    auto device = tc::createDevice(tc::DeviceRequest{options.backend, options.device});
    auto const info = device->info();
    std::printf("backend %s: %s, %d timed rounds\n", info.backend.c_str(), info.deviceName.c_str(), options.rounds);
    if (not info.timers) {
        std::fprintf(stderr, "layout_probe: this device has no timers\n");
        return EXIT_FAILURE;
    }
    device->enableTimers(true);
    probeInstances(*device, options);
    probeVertices(*device, options);
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const options = parse(argc, argv);
    if (not options) {
        std::fprintf(stderr, "usage: layout_probe [--backend cpu|opengl|gles|vulkan] [--device <name part>] "
                             "[--rounds <n>] [--scale <s>]\n");
        return 2;
    }
    try {
        return run(*options);
    } catch (std::exception const &error) {
        std::fprintf(stderr, "layout_probe: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
