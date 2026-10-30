#pragma once

#include <lr/render/View.h>

#include <filesystem>

namespace lr::present {

// Binary PPM (P6), dropping alpha; false when the file cannot be written.
bool writePpm(render::ImageRgba8View const &image, std::filesystem::path const &path);

} // namespace lr::present
