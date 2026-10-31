#pragma once

#if defined(__ANDROID__)

#include <lr/asset/AssetSource.h>

#include <memory>

struct AAssetManager;

namespace lr::asset {

// The assets packed into the APK, read on stackfull's blocking pool; `manager` outlives
// the source. Nothing to watch, so no hot reloading.
std::unique_ptr<AssetSource> makeAndroidAssetSource(AAssetManager *manager);

} // namespace lr::asset

#endif
