#include "KernelVariants.h"

#include "kernels/PathTrace.h"
#include "kernels/Post.h"
#include "kernels/Shading.h"
#include "kernels/Temporal.h"
#include "kernels/Textures.h"
#include "kernels/Visibility.h"

namespace lr::render {

namespace {

struct DeviceKernelCheck final : KernelCheck {
    explicit DeviceKernelCheck(tc::Device const &device_) noexcept : device(device_) {}

    std::optional<std::string> whyNot(std::string_view const program) const override {
        return device.checkKernel(program);
    }

private:
    tc::Device const &device;
};

// The programs of `programs` the device cannot run, as "name (reason)"; empty when it
// runs them all.
std::vector<std::string> failures(KernelCheck const &check, std::vector<std::string_view> const &programs) {
    auto result = std::vector<std::string>{};
    for (auto const program : programs) {
        if (auto const reason = check.whyNot(program)) {
            result.push_back(std::string{program} + " (" + *reason + ")");
        }
    }
    return result;
}

void throwIfAny(std::vector<std::string> const &failed) {
    if (failed.empty()) {
        return;
    }
    auto message = std::string{"lr::render: the device cannot run these kernels:"};
    for (auto const &failure : failed) {
        message += " " + failure + ";";
    }
    message.pop_back();
    throw tc::DeviceError{message};
}

bool runsAll(KernelCheck const &check, std::vector<std::string_view> const &programs) {
    return failures(check, programs).empty();
}

} // namespace

std::unique_ptr<KernelCheck> makeKernelCheck(tc::Device const &device) {
    return std::make_unique<DeviceKernelCheck>(device);
}

LrenderPrograms const &lrenderPrograms() {
    using namespace kernels;
    static auto const programs = LrenderPrograms{
        {ClearUint::fileLocation, CullInstances::fileLocation, SumVisibleBlocks::fileLocation,
         ScanVisibleBlocks::fileLocation, WriteVisible::fileLocation, CullClusters::fileLocation,
         SumClusterBlocks::fileLocation, WriteVisibleClusters::fileLocation,
         RasterVisibility::fileLocation, RasterVisibilityTiles::fileLocation, ScanLargeTriangles::fileLocation,
         CopyVisibleTotals::fileLocation, ClassifyPixels::fileLocation,
         AllocateModelLists::fileLocation, ScatterModelPixels::fileLocation,
         ShadeMaterialModel<MetallicRoughness>::fileLocation, BlendTransparent::fileLocation, CopyHdr::fileLocation,
         TemporalMotion::fileLocation, TemporalResolve::fileLocation, ShowResolved::fileLocation,
         PathTrace::fileLocation, Tonemap::fileLocation, DownsampleTexture::fileLocation,
         CopyTextureLayer::fileLocation},
        {SumVisibleBlocksSubgroups::fileLocation, WriteVisibleSubgroups::fileLocation,
         SumClusterBlocksSubgroups::fileLocation, WriteVisibleClustersSubgroups::fileLocation,
         ClassifyPixelsSubgroups::fileLocation},
        {ClearPackedVisibility::fileLocation, RasterPackedVisibility::fileLocation,
         RasterPackedVisibilityTiles::fileLocation, UnpackVisibility::fileLocation},
        {PathTraceRayQuery::fileLocation, CopyPositions::fileLocation, WriteTopLevelInstances::fileLocation}};
    return programs;
}

RendererCapabilities chooseCapabilities(KernelCheck const &check, VariantChoices const &choices) {
    auto const &programs = lrenderPrograms();
    throwIfAny(failures(check, programs.required));
    auto chosen = RendererCapabilities{};
    chosen.subgroupSize = choices.subgroupSize;
    chosen.subgroups = choices.subgroups == VariantUse::Auto and runsAll(check, programs.subgroups);
    chosen.singlePassVisibility =
        choices.singlePassVisibility == VariantUse::Auto and runsAll(check, programs.singlePassVisibility);
    chosen.rayQueries = choices.rayQueries == VariantUse::Auto and runsAll(check, programs.rayQueries);
    return chosen;
}

void verifyCapabilities(KernelCheck const &check, RendererCapabilities const &chosen,
                        std::vector<std::string_view> const &more) {
    auto const &programs = lrenderPrograms();
    auto needed = programs.required;
    if (chosen.subgroups) {
        needed.insert(needed.end(), programs.subgroups.begin(), programs.subgroups.end());
    }
    if (chosen.singlePassVisibility) {
        needed.insert(needed.end(), programs.singlePassVisibility.begin(), programs.singlePassVisibility.end());
    }
    if (chosen.rayQueries) {
        needed.insert(needed.end(), programs.rayQueries.begin(), programs.rayQueries.end());
    }
    needed.insert(needed.end(), more.begin(), more.end());
    throwIfAny(failures(check, needed));
}

} // namespace lr::render
