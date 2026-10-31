#include "Paths.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace lr::asset {

AssetPath resolvedPath(AssetPath const &base, std::string const &relative) {
    auto const directory = std::filesystem::path{base}.parent_path();
    return (directory / relative).lexically_normal().generic_string();
}

bool hasExtension(AssetPath const &path, std::string_view const extension) {
    if (path.size() < extension.size()) {
        return false;
    }
    return std::equal(extension.begin(), extension.end(), path.end() - static_cast<std::ptrdiff_t>(extension.size()),
                      [](char const a, char const b) {
                          return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
                      });
}

AssetPath withExtension(AssetPath const &path, std::string_view const extension) {
    return std::filesystem::path{path}.replace_extension(std::string{extension}).generic_string();
}

} // namespace lr::asset
