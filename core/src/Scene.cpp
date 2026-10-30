#include <lr/core/Scene.h>

namespace lr::core {

mat4 viewMatrix(CameraDesc const &camera) noexcept {
    return lookAt(camera.position, camera.target, camera.up);
}

mat4 projectionMatrix(CameraDesc const &camera, float const aspect) noexcept {
    return perspective(camera.verticalFov, aspect, camera.nearPlane, camera.farPlane);
}

} // namespace lr::core
