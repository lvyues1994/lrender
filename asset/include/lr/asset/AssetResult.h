#pragma once

#include <tl/expected.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace lr::asset {

enum class AssetErrorKind : std::uint8_t {
    NotFound,    // the source has nothing at the path
    ReadFailed,  // the source has it, but reading it failed
    Corrupt,     // not a valid file of its format
    Unsupported, // valid, but uses what lrender does not load, such as supercompressed KTX2
    Failed,      // the load itself could not run, e.g. no IO task could start
};

struct AssetError {
    AssetErrorKind kind = AssetErrorKind::Failed;
    std::string message;
};

// Missing and broken assets are values: a load completes with set_value either way.
template <class T>
using AssetResult = tl::expected<T, AssetError>;

inline tl::unexpected<AssetError> assetError(AssetErrorKind const kind, std::string message) {
    return tl::unexpected<AssetError>{AssetError{kind, std::move(message)}};
}

} // namespace lr::asset
