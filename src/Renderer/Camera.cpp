#include <Renderer/Camera.hpp>

namespace ysq {

Vec3f Camera::forward() const {
    return normalized(target - position);
}

Vec3f Camera::right() const {
    return normalized(cross(forward(), up));
}

Vec3f Camera::trueUp() const {
    return cross(right(), forward());
}

Matrix4<float> Camera::viewMatrix() const {
    return Matrix4<float>::lookAt(position, target, up);
}

// Reversed-Z: nearPlane/farPlane are passed to Matrix4::perspective/orthographic
// swapped, not in their usual order. That's algebraically identical to a
// dedicated reversed-Z matrix (negating the standard formula's z-row), so
// Matrix4 itself stays a general-purpose, standard-convention primitive;
// only Renderer's own use of it is reversed. Result: near maps to NDC/window
// depth 1.0, far maps to 0.0. A standard depth buffer spends nearly all its
// precision right next to the near plane; at the astronomical dynamic
// ranges this engine's true-to-scale scenarios produce (standing on a small
// body's surface while the far plane reaches system-wide extent), that
// leaves almost none for anything farther out, which reads as z-fighting.
// Reversed-Z paired with a floating-point depth buffer (see
// RenderTarget::create() in Renderer.cpp) distributes precision evenly
// instead. See src/Renderer/README.md's Conventions section.
Matrix4<float> Camera::projectionMatrix(float aspect) const {
    if (projection == ProjectionMode::Perspective) {
        return Matrix4<float>::perspective(perspectiveSettings.fovYRadians, aspect,
                                           perspectiveSettings.farPlane,
                                           perspectiveSettings.nearPlane);
    }
    const float height = orthographicSettings.height;
    const float width = height * aspect;
    return Matrix4<float>::orthographic(-width / 2.0f, width / 2.0f, -height / 2.0f,
                                        height / 2.0f, orthographicSettings.farPlane,
                                        orthographicSettings.nearPlane);
}

Matrix4<float> Camera::viewProjectionMatrix(float aspect) const {
    return projectionMatrix(aspect) * viewMatrix();
}

}  // namespace ysq
