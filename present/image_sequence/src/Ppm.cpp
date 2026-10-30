#include <lr/present/Ppm.h>

#include <cstddef>
#include <fstream>
#include <vector>

namespace lr::present {

bool writePpm(render::ImageRgba8View const &image, std::filesystem::path const &path) {
    auto file = std::ofstream{path, std::ios::binary};
    if (not file) {
        return false;
    }
    file << "P6\n" << image.extent.x << " " << image.extent.y << "\n255\n";
    auto const count = static_cast<std::size_t>(image.extent.x) * static_cast<std::size_t>(image.extent.y);
    auto rgb = std::vector<char>(count * 3);
    for (auto index = std::size_t{0}; index < count; ++index) {
        auto const pixel = image.pixels[index];
        rgb[3 * index] = static_cast<char>(pixel & 0xFFu);
        rgb[3 * index + 1] = static_cast<char>((pixel >> 8u) & 0xFFu);
        rgb[3 * index + 2] = static_cast<char>((pixel >> 16u) & 0xFFu);
    }
    file.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(file);
}

} // namespace lr::present
