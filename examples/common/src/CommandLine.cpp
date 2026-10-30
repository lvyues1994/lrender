#include <lr/examples/CommandLine.h>

#include <cstddef>
#include <cstdio>

namespace lr::examples {

std::optional<core::ivec2> parseSize(std::string const &text) {
    auto width = 0;
    auto height = 0;
    if (std::sscanf(text.c_str(), "%dx%d", &width, &height) != 2 or width <= 0 or height <= 0) {
        return std::nullopt;
    }
    return core::ivec2{width, height};
}

std::optional<std::uint32_t> parseCount(std::string const &text) {
    auto count = 0u;
    auto end = 0;
    if (std::sscanf(text.c_str(), "%u%n", &count, &end) != 1 or static_cast<std::size_t>(end) != text.size()
        or count == 0) {
        return std::nullopt;
    }
    return count;
}

} // namespace lr::examples
