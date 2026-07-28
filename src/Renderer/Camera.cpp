#include <Renderer/Camera.hpp>

namespace ysq {

Vec3f Camera::forward() const {
    return normalized(target - position);
}

Vec3f Camera::right() const {
    return normalized(cross(forward(), up));
}

Matrix4<float> Camera::viewMatrix() const {
    return Matrix4<float>::lookAt(position, target, up);
}

Matrix4<float> Camera::projectionMatrix(float aspect) const {
    if (projection == ProjectionMode::Perspective) {
        return Matrix4<float>::perspective(perspectiveSettings.fovYRadians, aspect,
                                           perspectiveSettings.nearPlane,
                                           perspectiveSettings.farPlane);
    }
    const float height = orthographicSettings.height;
    const float width = height * aspect;
    return Matrix4<float>::orthographic(-width / 2.0f, width / 2.0f, -height / 2.0f,
                                        height / 2.0f, orthographicSettings.nearPlane,
                                        orthographicSettings.farPlane);
}

Matrix4<float> Camera::viewProjectionMatrix(float aspect) const {
    return projectionMatrix(aspect) * viewMatrix();
}

}  // namespace ysq
