#pragma once

#include <memory>
#include <vector>

namespace lr::render {

// What the GPU thread lets go of while it records a frame, kept until the device has
// finished that frame, and with it every earlier one.
using RetireList = std::vector<std::shared_ptr<void>>;

} // namespace lr::render
