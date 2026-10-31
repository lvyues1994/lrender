#pragma once

#include <lr/asset/AssetSource.h>

#include <cstddef>
#include <cstdint>

namespace lr::asset {

// Bytes owned elsewhere: a whole file, or a glTF buffer view.
struct ByteView {
    static ByteView of(Bytes const &bytes) noexcept { return ByteView{bytes.data(), bytes.size()}; }

    std::uint8_t const *data = nullptr;
    std::size_t size = 0;
};

} // namespace lr::asset
