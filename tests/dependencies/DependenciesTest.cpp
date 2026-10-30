// The four libraries used together the way lrender uses them. A version bump that
// breaks one of these combinations fails here before anywhere else.

#include "SmokeKernels.h"

#include <lr/runtime/Runtime.h>

#include <co2/co2.hpp>
#include <lexec/coro/co2.hpp>
#include <lexec/execution.hpp>
#include <tc/devices.hpp>

#include <doctest/doctest.h>

#include <cstdlib>
#include <memory>
#include <tuple>

namespace {

lr::runtime::RuntimeConfig smallConfig() {
    auto config = lr::runtime::RuntimeConfig{};
    config.framePoolThreads = 2;
    return config;
}

auto incrementOnPool(lr::runtime::ComputeScheduler pool, int value) CO2_BEG(co2::Task<int>, (pool, value)) {
    CO2_AWAIT(lexec::schedule(pool));
    CO2_RETURN(value + 1);
}
CO2_END

void checkAddOne(tc::Device &device) {
    constexpr auto kCount = 256;
    auto values = tc::BufferResource<tc::uint>{kCount};
    for (auto index = 0; index < kCount; ++index) {
        values[index] = static_cast<tc::uint>(index);
    }
    auto kernel = lr::test::SmokeAddOne{};
    kernel.values.attach(values);

    device.uploadBuffer(values);
    device.useKernel(kernel);
    device.bindBuffer(kernel.values);
    device.execute(kernel, tc::uvec3{kCount / 64, 1, 1});
    device.downloadBuffer(values);

    auto mismatches = 0;
    for (auto index = 0; index < kCount; ++index) {
        if (values[index] != static_cast<tc::uint>(index + 1)) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE("a co2 task awaits a lexec sender and is itself a lexec sender") {
    auto runtime = lr::runtime::makeRuntime(smallConfig());

    auto const result = lexec::sync_wait(lexec::coro::as_sender(incrementOnPool(runtime->framePool(), 41)));

    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 42);
}

TEST_CASE("a kernel transpiled at build time is registered for every GPU dialect") {
    auto const program = tc::ProgramRegistry::instance().find(lr::test::SmokeAddOne::fileLocation);

    REQUIRE(program.has_value());
    CHECK(not program->glslDesktop.empty());
    CHECK(not program->glslEs.empty());
    CHECK(not program->spirv.empty());
}

TEST_CASE("the kernel runs on the CPU backend") {
    auto device = tc::createDevice(tc::DeviceRequest{"cpu", ""});

    checkAddOne(*device);
}

// Mesa's software devices (llvmpipe, lavapipe) behave the same on every machine. Without
// them the GPU backends are skipped, unless LRENDER_TEST_REQUIRE_GPU is set, as in CI.
TEST_CASE("the kernel runs on every GPU backend") {
    auto const requireGpu = std::getenv("LRENDER_TEST_REQUIRE_GPU") != nullptr;
    for (auto const &backend : tc::availableBackends()) {
        if (backend == "cpu") {
            continue;
        }
        CAPTURE(backend);
        auto device = std::unique_ptr<tc::Device>{};
        try {
            device = tc::createDevice(tc::DeviceRequest{backend, "llvmpipe"});
        } catch (tc::DeviceError const &error) {
            if (requireGpu) {
                FAIL(error.what());
            }
            MESSAGE("skipped: " << error.what());
            continue;
        }
        checkAddOne(*device);
    }
}

} // namespace
