#pragma once

#include <Math/Matrix4.hpp>
#include <Math/Vector3.hpp>

namespace ysq {

enum class ProjectionMode { Perspective, Orthographic };

struct PerspectiveSettings {
    /// Full vertical field of view, in radians.
    float fovYRadians = 0.9599311f;  // 55 degrees
    float nearPlane = 0.1f;
    float farPlane = 10000.0f;
};

/// Orthographic extent in world units. `height` is the full vertical span;
/// width derives from it and the viewport aspect ratio each frame, the same
/// way perspective's horizontal FOV derives from fovYRadians and aspect.
struct OrthographicSettings {
    float height = 10.0f;
    float nearPlane = -10000.0f;
    float farPlane = 10000.0f;
};

/// A view into a scene: eye, target and up, plus a projection. Holds no
/// window or context state, and the aspect ratio is not stored here since it
/// belongs to whatever framebuffer is being drawn into, not to the camera
/// itself; Renderer takes it per frame.
///
/// An orthographic projection with every object at z = 0 is a 2D scene: a
/// planar orbit is this, not a separate rendering path. See
/// docs/rendering.md.
struct Camera {
    Vec3f position = {0.0f, 0.0f, 5.0f};
    Vec3f target = Vec3f::zero();
    Vec3f up = Vec3f::unitY();

    ProjectionMode projection = ProjectionMode::Perspective;
    PerspectiveSettings perspectiveSettings{};
    OrthographicSettings orthographicSettings{};

    [[nodiscard]] Vec3f forward() const;
    [[nodiscard]] Vec3f right() const;

    [[nodiscard]] Matrix4<float> viewMatrix() const;
    [[nodiscard]] Matrix4<float> projectionMatrix(float aspect) const;
    [[nodiscard]] Matrix4<float> viewProjectionMatrix(float aspect) const;
};

}  // namespace ysq
