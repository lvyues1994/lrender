#pragma once

#include <lr/asset/AssetResult.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lr::asset {

// '/'-separated and relative to the source's root; a glTF file's references resolve
// against its directory.
using AssetPath = std::string;
using Bytes = std::vector<std::uint8_t>;

// Where the bytes of assets come from. AssetLibrary calls it only inside its IO tasks,
// so read() may block in stackfull::blocking() or park the task.
struct AssetSource {
    virtual ~AssetSource() = default;

    virtual AssetResult<Bytes> read(AssetPath const &path) = 0;
    // The local file behind `path`, which hot reloading watches; nullopt where assets
    // are not files.
    virtual std::optional<std::string> localFile(AssetPath const &path) const = 0;
};

struct FileSourceConfig {
    std::string root; // empty: paths are relative to the working directory
};

// Reads whole files on stackfull's blocking pool. Absolute paths ignore the root.
std::unique_ptr<AssetSource> makeFileSource(FileSourceConfig const &config);

struct MemorySourceConfig {
    std::unordered_map<AssetPath, Bytes> files;
};

// A fixed set of files in memory, for tests.
std::unique_ptr<AssetSource> makeMemorySource(MemorySourceConfig config);

} // namespace lr::asset
