#pragma once

#include "render/RenderTestSupport.h"

#include <lr/core/Primitives.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lr::test {

// A red quad in front of a blue sky.
inline core::RenderWorldDelta presentedScene() {
    auto delta = core::RenderWorldDelta{};
    delta.meshes = {{core::MeshId{0}, share(core::makeQuad(core::vec2{0.5f, 0.5f}))}};
    delta.materials = {{core::MaterialId{0}, emissive(core::vec3{0.8f, 0.1f, 0.05f})}};
    delta.instances = {{core::InstanceId{0}, core::MeshId{0}, core::MaterialId{0}, core::mat4{1.0f}}};
    delta.environment = core::EnvironmentDesc{core::vec3{0.05f, 0.1f, 0.4f}};
    return delta;
}

inline core::CameraDesc presentedCamera() { return cameraAt(core::vec3{0.0f, 0.0f, 3.0f}, core::vec3{0.0f}); }

// An empty directory of the build tree, for one test's files.
inline std::filesystem::path freshOutputDirectory(std::string const &name) {
    auto const directory = std::filesystem::path{LR_TEST_OUTPUT_DIR} / name;
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

} // namespace lr::test
