#pragma once

#include <lr/render/Capabilities.h>

#include <tc/device.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lr::render {

// Whether a device runs a transpiled program.
struct KernelCheck {
    virtual ~KernelCheck() = default;
    // Why the device cannot run `program`, or nullopt when it can.
    virtual std::optional<std::string> whyNot(std::string_view program) const = 0;
};

// Device::checkKernel() of a device, which must outlive it.
std::unique_ptr<KernelCheck> makeKernelCheck(tc::Device const &device);

// lrender's programs: those some frame may need whatever the variants, and those of each
// optional variant.
struct LrenderPrograms {
    std::vector<std::string_view> required;
    std::vector<std::string_view> subgroups;
    std::vector<std::string_view> singlePassVisibility;
    std::vector<std::string_view> rayQueries;
};
LrenderPrograms const &lrenderPrograms();

struct VariantChoices {
    VariantUse subgroups = VariantUse::Auto;
    VariantUse singlePassVisibility = VariantUse::Auto;
    std::uint32_t subgroupSize = 0; // of the device
    VariantUse rayQueries = VariantUse::Auto;
};

// Each variant the choices allow and the device runs. Throws tc::DeviceError naming every
// required program the device cannot run, and why.
RendererCapabilities chooseCapabilities(KernelCheck const &check, VariantChoices const &choices);

// For a device made after the first: throws tc::DeviceError naming what it cannot run of
// the programs `chosen` needs and of `more`.
void verifyCapabilities(KernelCheck const &check, RendererCapabilities const &chosen,
                        std::vector<std::string_view> const &more);

} // namespace lr::render
