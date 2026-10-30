#pragma once

#include <lr/core/Math.h>

#include <cstdint>
#include <optional>
#include <string>

namespace lr::examples {

// "<width>x<height>", both positive.
std::optional<core::ivec2> parseSize(std::string const &text);

// A positive decimal number.
std::optional<std::uint32_t> parseCount(std::string const &text);

} // namespace lr::examples
