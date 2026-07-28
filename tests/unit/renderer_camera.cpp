#include <Platform/Input.hpp>
#include <Renderer/Camera.hpp>
#include <Renderer/CameraController.hpp>

#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

// Pure math: no GL context, no window. Camera's projection/view factories
// and the two controllers are plain functions over Math and
// Platform::InputState, so they are testable exactly like any other engine
// module. See tests/integration/renderer_framebuffer.cpp for the part that
// needs a real context.

namespace {

using ysq::ButtonAction;
using ysq::Camera;
using ysq::FreeFlyCameraController;
using ysq::InputState;
using ysq::Key;
using ysq::Matrix4;
using ysq::Modifiers;
using ysq::MouseButton;
using ysq::OrbitCameraController;
using ysq::ProjectionMode;
using ysq::Vec3f;

}  // namespace

TEST(RendererCamera, PerspectiveProjectionMatchesMatrix4Perspective) {
    Camera camera;
    camera.projection = ProjectionMode::Perspective;
    camera.perspectiveSettings = {0.9f, 0.1f, 100.0f};
    const float aspect = 16.0f / 9.0f;

    const Matrix4<float> expected =
        Matrix4<float>::perspective(0.9f, aspect, 0.1f, 100.0f);
    EXPECT_MAT_APPROX(camera.projectionMatrix(aspect), expected);
}

TEST(RendererCamera, OrthographicProjectionMatchesMatrix4Orthographic) {
    Camera camera;
    camera.projection = ProjectionMode::Orthographic;
    camera.orthographicSettings = {10.0f, -10.0f, 10.0f};
    const float aspect = 2.0f;

    const Matrix4<float> expected =
        Matrix4<float>::orthographic(-10.0f, 10.0f, -5.0f, 5.0f, -10.0f, 10.0f);
    EXPECT_MAT_APPROX(camera.projectionMatrix(aspect), expected);
}

TEST(RendererCamera, ViewMatrixMatchesLookAt) {
    Camera camera;
    camera.position = {1.0f, 2.0f, 3.0f};
    camera.target = Vec3f::zero();
    camera.up = Vec3f::unitY();

    const Matrix4<float> expected =
        Matrix4<float>::lookAt(camera.position, camera.target, camera.up);
    EXPECT_MAT_APPROX(camera.viewMatrix(), expected);
}

TEST(RendererCamera, ForwardAndRightAreOrthonormalToEachOther) {
    Camera camera;
    camera.position = {5.0f, 1.0f, 0.0f};
    camera.target = Vec3f::zero();

    const Vec3f forward = camera.forward();
    const Vec3f right = camera.right();
    EXPECT_APPROX(length(forward), 1.0f);
    EXPECT_APPROX(length(right), 1.0f);
    EXPECT_NEAR(dot(forward, right), 0.0f, 1e-5f);
}

TEST(RendererCamera, OrbitControllerAtZeroAzimuthAndElevationLooksDownMinusZ) {
    InputState input;
    input.newFrame();

    OrbitCameraController controller;
    controller.target = Vec3f::zero();
    controller.distance = 10.0f;
    controller.azimuthRadians = 0.0f;
    controller.elevationRadians = 0.0f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_VEC_APPROX(camera.position, (Vec3f{0.0f, 0.0f, 10.0f}));
    EXPECT_VEC_APPROX(camera.target, Vec3f::zero());
}

TEST(RendererCamera, OrbitControllerRotatesOnlyWhileTheRightButtonIsHeld) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onCursorPosition(100.0, 0.0);  // moved, but no button held

    OrbitCameraController controller;
    controller.distance = 10.0f;
    controller.azimuthRadians = 0.0f;
    controller.elevationRadians = 0.0f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_VEC_APPROX(camera.position, (Vec3f{0.0f, 0.0f, 10.0f}))
        << "a drag with no button held must not rotate the camera";
}

TEST(RendererCamera, OrbitControllerRotatesWhileTheRightButtonIsHeld) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onMouseButton(MouseButton::Right, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);

    OrbitCameraController controller;
    controller.distance = 10.0f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_NE(camera.position, (Vec3f{0.0f, 0.0f, 10.0f}));
    EXPECT_APPROX(length(camera.position - controller.target), controller.distance)
        << "rotating must not change the orbit radius";
}

TEST(RendererCamera, OrbitControllerZoomsInOnPositiveScroll) {
    InputState input;
    input.newFrame();
    input.onScroll(0.0, 1.0);

    OrbitCameraController controller;
    controller.distance = 10.0f;
    controller.zoomSpeed = 0.1f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_LT(controller.distance, 10.0f);
}

TEST(RendererCamera, OrbitControllerNeverZoomsBelowTheMinimumDistance) {
    InputState input;
    input.newFrame();
    input.onScroll(0.0, 1000.0);  // an absurd scroll, to try to drive it negative

    OrbitCameraController controller;
    controller.distance = 10.0f;
    controller.minDistance = 0.5f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_GE(controller.distance, controller.minDistance);
}

TEST(RendererCamera, FreeFlyControllerMovesForwardOnW) {
    InputState input;
    input.newFrame();
    input.onKey(Key::W, ButtonAction::Press, Modifiers{});

    FreeFlyCameraController controller;
    controller.position = Vec3f::zero();
    controller.moveSpeed = 5.0f;
    // Default yaw already looks down -Z, matching Camera's own default.

    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_VEC_APPROX(controller.position, (Vec3f{0.0f, 0.0f, -5.0f}));
    EXPECT_VEC_APPROX(camera.position, controller.position);
}

TEST(RendererCamera, FreeFlyControllerShiftMultipliesSpeed) {
    InputState input;
    input.newFrame();
    input.onKey(Key::W, ButtonAction::Press, Modifiers{});
    input.onKey(Key::LeftShift, ButtonAction::Press, Modifiers{});

    FreeFlyCameraController controller;
    controller.position = Vec3f::zero();
    controller.moveSpeed = 5.0f;
    controller.fastMultiplier = 4.0f;

    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_APPROX(length(controller.position), 20.0f);
}

TEST(RendererCamera, FreeFlyControllerStaysStillWithNoKeysHeld) {
    InputState input;
    input.newFrame();

    FreeFlyCameraController controller;
    controller.position = {1.0f, 2.0f, 3.0f};

    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_VEC_APPROX(controller.position, (Vec3f{1.0f, 2.0f, 3.0f}));
}
