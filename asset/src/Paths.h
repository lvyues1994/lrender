#pragma once

#include <lr/asset/AssetSource.h>

#include <string>
#include <string_view>

namespace lr::asset {

// `relative` (a decoded glTF URI) resolved against the directory of `base`.
AssetPath resolvedPath(AssetPath const &base, std::string const &relative);

// Whether `path` ends in `extension` (".ktx2"), ignoring case.
bool hasExtension(AssetPath const &path, std::string_view extension);

AssetPath withExtension(AssetPath const &path, std::string_view extension);

} // namespace lr::asset
