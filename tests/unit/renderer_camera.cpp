#include <Platform/Input.hpp>
#include <Renderer/Camera.hpp>
#include <Renderer/CameraController.hpp>
#include <Renderer/SceneCameraController.hpp>

#include <support/MathApprox.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

// Pure math: no GL context, no window. Camera's projection/view factories
// and the two controllers are plain functions over Math and
// Platform::InputState, so they are testable exactly like any other engine
// module. See tests/integration/renderer_framebuffer.cpp for the part that
// needs a real context.

namespace {

using ysq::ButtonAction;
using ysq::Camera;
using ysq::CameraMode;
using ysq::FreeFlyCameraController;
using ysq::InputState;
using ysq::Key;
using ysq::Matrix4;
using ysq::Modifiers;
using ysq::MouseButton;
using ysq::NamedSphere;
using ysq::OrbitCameraController;
using ysq::PerspectiveSettings;
using ysq::ProjectionMode;
using ysq::SceneCameraController;
using ysq::Vec3f;

}  // namespace

TEST(RendererCamera, PerspectiveProjectionMatchesMatrix4Perspective) {
    Camera camera;
    camera.projection = ProjectionMode::Perspective;
    camera.perspectiveSettings = {0.9f, 0.1f, 100.0f};
    const float aspect = 16.0f / 9.0f;

    // Reversed-Z: Camera::projectionMatrix() passes near/far to
    // Matrix4::perspective swapped, so the expected matrix here must be
    // built the same way -- see Camera.cpp and src/Renderer/README.md.
    const Matrix4<float> expected =
        Matrix4<float>::perspective(0.9f, aspect, 100.0f, 0.1f);
    EXPECT_MAT_APPROX(camera.projectionMatrix(aspect), expected);
}

TEST(RendererCamera, OrthographicProjectionMatchesMatrix4Orthographic) {
    Camera camera;
    camera.projection = ProjectionMode::Orthographic;
    camera.orthographicSettings = {10.0f, -10.0f, 10.0f};
    const float aspect = 2.0f;

    // Reversed-Z: near/far swapped, same reasoning as the perspective case
    // above.
    const Matrix4<float> expected =
        Matrix4<float>::orthographic(-10.0f, 10.0f, -5.0f, 5.0f, 10.0f, -10.0f);
    EXPECT_MAT_APPROX(camera.projectionMatrix(aspect), expected);
}

TEST(RendererCamera, PerspectiveProjectionIsReversedZ) {
    Camera camera;
    camera.projection = ProjectionMode::Perspective;
    camera.perspectiveSettings = {0.9f, 0.1f, 100.0f};
    const float aspect = 16.0f / 9.0f;
    const Matrix4<float> proj = camera.projectionMatrix(aspect);

    // A point at the near plane and one at the far plane, in eye space
    // (looking down -Z). Reversed-Z means near lands at NDC z = +1 and far
    // at NDC z = -1 -- the opposite of the textbook convention.
    const Matrix4<float>::Column nearClip =
        proj * Matrix4<float>::Column{0.0f, 0.0f, -0.1f, 1.0f};
    const Matrix4<float>::Column farClip =
        proj * Matrix4<float>::Column{0.0f, 0.0f, -100.0f, 1.0f};

    EXPECT_APPROX(nearClip.z / nearClip.w, 1.0f);
    EXPECT_APPROX(farClip.z / farClip.w, -1.0f);
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

TEST(RendererCamera, OrbitControllerPansWhileTheLeftButtonIsHeld) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onMouseButton(MouseButton::Left, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);

    OrbitCameraController controller;
    controller.target = Vec3f::zero();
    controller.distance = 10.0f;
    controller.azimuthRadians = 0.0f;
    controller.elevationRadians = 0.0f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_NE(controller.target, Vec3f::zero());
    EXPECT_APPROX(controller.azimuthRadians, 0.0f) << "panning must not rotate the view";
    EXPECT_APPROX(controller.elevationRadians, 0.0f);
    EXPECT_APPROX(length(camera.position - controller.target), controller.distance)
        << "panning must not change the orbit radius";
}

TEST(RendererCamera, OrbitControllerPansTheCorrectDirection) {
    // A regression test for a real bug: pan's right vector was built from
    // `direction` (target -> camera, outward) instead of the camera's own
    // forward (camera -> target), silently mirroring the horizontal pan
    // component -- vertical pan happened to come out correct regardless
    // (the sign error cancels there), which is why the less specific test
    // above didn't catch it.
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onMouseButton(MouseButton::Left, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);  // drag right

    OrbitCameraController controller;
    controller.target = Vec3f::zero();
    controller.distance = 10.0f;
    controller.azimuthRadians = 0.0f;
    controller.elevationRadians = 0.0f;

    Camera camera;
    controller.update(camera, input);

    // At this default orientation the camera looks down -Z with Camera::right()
    // == +X, so dragging right must move the target toward -X: content
    // (everything at/around the target) then appears to slide right, matching
    // the cursor, instead of mirrored.
    EXPECT_LT(controller.target.x, 0.0f);
    EXPECT_APPROX(controller.target.y, 0.0f);
    EXPECT_APPROX(controller.target.z, 0.0f);
}

TEST(RendererCamera, OrbitControllerDoesNotPanWithoutTheLeftButtonHeld) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onCursorPosition(100.0, 0.0);  // moved, but no button held

    OrbitCameraController controller;
    controller.target = Vec3f::zero();
    controller.distance = 10.0f;

    Camera camera;
    controller.update(camera, input);

    EXPECT_VEC_APPROX(controller.target, Vec3f::zero());
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

TEST(RendererCamera, FreeFlyControllerScrollChangesMoveSpeed) {
    InputState input;
    input.newFrame();
    input.onScroll(0.0, 1.0);

    FreeFlyCameraController controller;
    controller.moveSpeed = 5.0f;
    controller.scrollSpeedFactor = 0.1f;

    Camera camera;
    controller.update(camera, input, 0.0f);

    EXPECT_APPROX(controller.moveSpeed, 5.5f);
}

TEST(RendererCamera, FreeFlyControllerNoRollWithoutKeysHeld) {
    InputState input;
    input.newFrame();

    FreeFlyCameraController controller;
    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_VEC_APPROX(camera.up, Vec3f::unitY());
}

TEST(RendererCamera, FreeFlyControllerRollWhileCHeld) {
    InputState input;
    input.newFrame();
    input.onKey(Key::C, ButtonAction::Press, Modifiers{});

    FreeFlyCameraController controller;
    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_NE(camera.up, Vec3f::unitY());
    EXPECT_APPROX(length(camera.up), 1.0f);
}

TEST(RendererCamera, FreeFlyControllerAMovesAlongTheRolledRightAfterRolling) {
    // A regression test for a real reported bug: after rolling, A/D, Q/E,
    // and pan used to stay relative to fixed world-up, so rolling 90
    // degrees and pressing A moved along the *original* unrolled "left"
    // instead of the rolled view's actual left -- which showed up as "I
    // rolled -90 degrees, pressed A, and it went down instead of left".
    // Movement is now true 6-DOF: relative to the current rolled
    // orientation.
    InputState input;
    input.newFrame();
    input.onKey(Key::A, ButtonAction::Press, Modifiers{});

    FreeFlyCameraController controller;
    controller.position = Vec3f::zero();
    controller.moveSpeed = 5.0f;
    controller.accelerationPerSecond = 1000.0f;  // effectively instant for this one step
    controller.rollRadians = 1.5707963f;         // 90 degrees

    Camera camera;
    controller.update(camera, input, 1.0f);

    // Default yaw looks down -Z; recompute the expected rolled right
    // vector independently (not by copying the implementation's formula)
    // to check A actually follows it.
    const Vec3f forward{0.0f, 0.0f, -1.0f};
    const Vec3f rolledUp = rotateAbout(Vec3f::unitY(), forward, controller.rollRadians);
    const Vec3f rolledRight = normalized(cross(forward, rolledUp));

    EXPECT_VEC_APPROX(normalized(controller.position), -rolledRight)
        << "A must strafe opposite the rolled right vector, not the unrolled "
           "world-relative one";
}

TEST(RendererCamera, FreeFlyControllerEMovesAlongTheRolledUpAfterRolling) {
    InputState input;
    input.newFrame();
    input.onKey(Key::E, ButtonAction::Press, Modifiers{});

    FreeFlyCameraController controller;
    controller.position = Vec3f::zero();
    controller.moveSpeed = 5.0f;
    controller.accelerationPerSecond = 1000.0f;
    controller.rollRadians = 1.5707963f;  // 90 degrees

    Camera camera;
    controller.update(camera, input, 1.0f);

    const Vec3f forward{0.0f, 0.0f, -1.0f};
    const Vec3f rolledUp = rotateAbout(Vec3f::unitY(), forward, controller.rollRadians);

    EXPECT_VEC_APPROX(normalized(controller.position), rolledUp)
        << "E must move along the rolled up vector, not world-up";
}

TEST(RendererCamera, FreeFlyControllerLookLockToggleAppliesCursorDeltaWithoutRmb) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onKey(Key::T, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);

    FreeFlyCameraController controller;
    const float initialYaw = controller.yawRadians;

    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_TRUE(controller.lookLocked);
    EXPECT_NE(controller.yawRadians, initialYaw);
}

TEST(RendererCamera, FreeFlyControllerLookIsRolledAtANinetyDegreeRoll) {
    // A regression test for a real reported bug: the previous roll fix
    // (A/D, Q/E, pan) didn't cover right-click-drag look, which still
    // mapped screen-space mouse delta straight onto yaw/pitch regardless
    // of roll. At a 90 degree roll, a purely-horizontal drag must become a
    // pure *pitch* change, not yaw -- the inverse of the unrolled case.
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onMouseButton(MouseButton::Right, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);  // drag purely horizontal on screen

    FreeFlyCameraController controller;
    controller.rollRadians = 1.5707963f;  // 90 degrees
    const float initialYaw = controller.yawRadians;

    Camera camera;
    controller.update(camera, input, 1.0f);

    EXPECT_APPROX(controller.yawRadians, initialYaw)
        << "a horizontal drag at a 90 degree roll must not change yaw";
    EXPECT_NE(controller.pitchRadians, 0.0f)
        << "a horizontal drag at a 90 degree roll must change pitch instead";
}

TEST(RendererCamera, FreeFlyControllerInertiaApproachesTargetGradually) {
    InputState input;
    input.newFrame();
    input.onKey(Key::W, ButtonAction::Press, Modifiers{});

    FreeFlyCameraController controller;
    controller.moveSpeed = 5.0f;
    controller.accelerationPerSecond = 1.0f;

    Camera camera;
    controller.update(camera, input, 0.1f);

    // approach = clamp(1.0 * 0.1, 0, 1) = 0.1, so velocity is 10% of the way
    // to the full 5 units/second forward target, not there already.
    EXPECT_APPROX(length(controller.velocity), 0.5f);
}

TEST(RendererCamera, FreeFlyControllerPansWhileTheLeftButtonIsHeld) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onMouseButton(MouseButton::Left, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);

    FreeFlyCameraController controller;
    controller.position = Vec3f::zero();
    const float initialYaw = controller.yawRadians;
    const float initialPitch = controller.pitchRadians;

    Camera camera;
    // Zero deltaSeconds isolates the pan offset from WASD/inertia, neither
    // of which is in play here anyway (no movement keys held).
    controller.update(camera, input, 0.0f);

    EXPECT_NE(controller.position, Vec3f::zero());
    EXPECT_APPROX(controller.yawRadians, initialYaw)
        << "panning must not rotate the view";
    EXPECT_APPROX(controller.pitchRadians, initialPitch);
}

TEST(RendererCamera, FreeFlyControllerDoesNotPanWithoutTheLeftButtonHeld) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onCursorPosition(100.0, 0.0);  // moved, but no button held

    FreeFlyCameraController controller;
    controller.position = Vec3f::zero();

    Camera camera;
    controller.update(camera, input, 0.0f);

    EXPECT_VEC_APPROX(controller.position, Vec3f::zero());
}

TEST(RendererCamera, SceneCameraControllerBothSetAnchorsToClosestSurfacePoint) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f{0.0f, 0.0f, 0.0f}, 2.0f},
        NamedSphere{"Moon", Vec3f{10.0f, 0.0f, 0.0f}, 1.0f},
    };

    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_VEC_APPROX(camera.position, (Vec3f{2.002f, 0.0f, 0.0f}));
    EXPECT_VEC_APPROX(camera.target, (Vec3f{10.0f, 0.0f, 0.0f}));
    EXPECT_FALSE(controller.isHidden(0)) << "the POV body is shown by default";
    EXPECT_FALSE(controller.isHidden(1));
}

TEST(RendererCamera, SceneCameraControllerHidePovHasNoEffectWhenPovIsFree) {
    SceneCameraController controller;
    controller.hidePov = true;

    EXPECT_FALSE(controller.isHidden(0));
    EXPECT_FALSE(controller.isHidden(1));
}

TEST(RendererCamera, SceneCameraControllerHidePovHidesOnlyThePovBodyInLockedMode) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f{0.0f, 0.0f, 0.0f}, 2.0f},
        NamedSphere{"Moon", Vec3f{10.0f, 0.0f, 0.0f}, 1.0f},
    };

    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;
    controller.hidePov = true;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_TRUE(controller.isHidden(0));
    EXPECT_FALSE(controller.isHidden(1));
}

TEST(RendererCamera,
     SceneCameraControllerLockedModeUpVectorStaysContinuousThroughOverhead) {
    // A regression test for a real reported bug: watching a tracked body
    // sweep close to directly overhead used to make the camera visibly
    // flip, because the old up-hint logic switched between two unrelated
    // fixed axes right at a hardcoded threshold. Simulates the focus body
    // sweeping through being overhead of a fixed POV body across many
    // small steps (as a real orbit would, frame to frame) and asserts
    // consecutive frames' up vectors never differ by more than a small
    // angle -- this fails against the pre-fix binary-switch logic and
    // should pass now that up is derived continuously.
    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;

    InputState input;
    input.newFrame();

    Camera camera;
    Vec3f previousUp{};
    bool havePrevious = false;
    constexpr int kSteps = 400;
    for (int i = 0; i <= kSteps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSteps);  // 0..1
        // Sweeps from clearly off vertical (arccos(0.99) is ~0.1415 rad, so
        // 0.3 rad starts well outside the old code's near-pole threshold),
        // through exactly vertical, to clearly off vertical on the other
        // side -- crossing the threshold both ways, the exact scenario a
        // body passing near overhead produces.
        const float angle = (t - 0.5f) * 0.6f;  // -0.3..0.3 radians off vertical
        const Vec3f focusDirection{std::sin(angle), std::cos(angle), 0.0f};
        const std::vector<NamedSphere> objects{
            NamedSphere{"Body", Vec3f::zero(), 1.0f},
            NamedSphere{"Focus", focusDirection * 10.0f, 0.5f},
        };

        controller.update(camera, objects, input, 0.01f);

        if (havePrevious) {
            EXPECT_GT(dot(camera.up, previousUp), 0.9f)
                << "up vector jumped between consecutive frames at step " << i;
        }
        previousUp = camera.up;
        havePrevious = true;
    }
}

TEST(RendererCamera, SceneCameraControllerBothSetScrollZoomsFov) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
        NamedSphere{"Moon", Vec3f{10.0f, 0.0f, 0.0f}, 1.0f},
    };

    InputState input;
    input.newFrame();
    input.onScroll(0.0, 1.0);

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;
    controller.zoomSpeed = 0.1f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_APPROX(controller.zoomFovScale, 1.1f);
    EXPECT_APPROX(camera.perspectiveSettings.fovYRadians,
                  PerspectiveSettings{}.fovYRadians / 1.1f);
}

TEST(RendererCamera, SceneCameraControllerResetRestoresDefaultZoom) {
    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;
    controller.zoomFovScale = 5.0f;

    controller.reset();

    EXPECT_APPROX(controller.zoomFovScale, 1.0f);
}

TEST(RendererCamera, SceneCameraControllerRKeyResetsZoomDuringUpdate) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
        NamedSphere{"Moon", Vec3f{10.0f, 0.0f, 0.0f}, 1.0f},
    };

    InputState input;
    input.newFrame();
    input.onKey(Key::R, ButtonAction::Press, Modifiers{});

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;
    controller.zoomFovScale = 5.0f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_APPROX(controller.zoomFovScale, 1.0f);
}

TEST(RendererCamera, SceneCameraControllerRestoresFovAfterLeavingLockedMode) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
        NamedSphere{"Moon", Vec3f{10.0f, 0.0f, 0.0f}, 1.0f},
    };

    Camera camera;
    const float originalFov = camera.perspectiveSettings.fovYRadians;

    InputState zoomInput;
    zoomInput.newFrame();
    zoomInput.onScroll(0.0, 5.0);

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;
    controller.zoomSpeed = 0.5f;
    controller.update(camera, objects, zoomInput, 1.0f);

    EXPECT_LT(camera.perspectiveSettings.fovYRadians, originalFov)
        << "zooming in should have narrowed the FOV";

    controller.focusIndex = -1;  // leave the locked submode
    InputState idleInput;
    idleInput.newFrame();
    controller.update(camera, objects, idleInput, 1.0f);

    EXPECT_APPROX(camera.perspectiveSettings.fovYRadians, originalFov)
        << "leaving the locked submode must restore the FOV it captured on entry";
}

TEST(RendererCamera, SceneCameraControllerRKeyResetsAnchorAndHeightDuringUpdate) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
    };

    InputState input;
    input.newFrame();
    input.onKey(Key::R, ButtonAction::Press, Modifiers{});

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.azimuthRadians = 2.0f;
    controller.elevationRadians = 1.0f;
    controller.heightRadii = 3.0f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_APPROX(controller.azimuthRadians, 0.0f);
    EXPECT_APPROX(controller.elevationRadians, 0.3f);
    EXPECT_APPROX(controller.heightRadii, 0.0f);
}

TEST(RendererCamera, SceneCameraControllerResetIsNoOpWhenPovIsFree) {
    SceneCameraController controller;
    controller.zoomFovScale = 5.0f;

    controller.reset();

    EXPECT_APPROX(controller.zoomFovScale, 5.0f);
}

TEST(RendererCamera, SceneCameraControllerFocusFreeMouseDragPicksAnchor) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
    };

    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.newFrame();
    input.onMouseButton(MouseButton::Right, ButtonAction::Press, Modifiers{});
    input.onCursorPosition(100.0, 0.0);

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.azimuthRadians = 0.0f;
    controller.elevationRadians = 0.0f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_NE(controller.azimuthRadians, 0.0f);
    EXPECT_APPROX(length(camera.position),
                  2.0f);  // still exactly at the surface (height 0)
    EXPECT_FALSE(controller.isHidden(0)) << "the POV body is shown by default";
}

TEST(RendererCamera, SceneCameraControllerHidePovHidesInFocusFreeMode) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
    };

    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.hidePov = true;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_TRUE(controller.isHidden(0));
}

TEST(RendererCamera, SceneCameraControllerFocusFreeScrollRaisesHeight) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
    };

    InputState input;
    input.newFrame();
    input.onScroll(0.0, 50.0);

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.zoomSpeed = 0.5f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_GT(controller.heightRadii, 0.05f);
}

TEST(RendererCamera,
     SceneCameraControllerFocusFreeLooksAtCenterOnceWellAboveLookBlendCeiling) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f{1.0f, 2.0f, 3.0f}, 2.0f},
    };

    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.povIndex = 0;
    controller.azimuthRadians = 0.0f;
    controller.elevationRadians = 0.0f;
    controller.heightRadii = 5.0f;  // well past the look-blend ceiling

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_VEC_APPROX(camera.target, (Vec3f{1.0f, 2.0f, 3.0f}));
}

TEST(RendererCamera, SceneCameraControllerResetRestoresDefaultAnchorAndHeight) {
    SceneCameraController controller;
    controller.povIndex = 0;
    controller.azimuthRadians = 2.0f;
    controller.elevationRadians = 1.0f;
    controller.heightRadii = 3.0f;

    controller.reset();

    EXPECT_APPROX(controller.azimuthRadians, 0.0f);
    EXPECT_APPROX(controller.elevationRadians, 0.3f);
    EXPECT_APPROX(controller.heightRadii, 0.0f);
}

TEST(RendererCamera, SceneCameraControllerAutoTracksFocusInOrbitMode) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Sun", Vec3f::zero(), 5.0f},
        NamedSphere{"Earth", Vec3f{20.0f, 0.0f, 0.0f}, 1.0f},
    };

    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.mode = CameraMode::Orbit;
    controller.focusIndex = 1;
    controller.orbit.distance = 10.0f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_VEC_APPROX(controller.orbit.target, (Vec3f{20.0f, 0.0f, 0.0f}));
    EXPECT_VEC_APPROX(camera.target, (Vec3f{20.0f, 0.0f, 0.0f}));
}

TEST(RendererCamera, SceneCameraControllerNoAutoTrackInFreeFlyMode) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Sun", Vec3f::zero(), 5.0f},
        NamedSphere{"Earth", Vec3f{20.0f, 0.0f, 0.0f}, 1.0f},
    };

    InputState input;
    input.newFrame();
    input.onKey(Key::W, ButtonAction::Press, Modifiers{});

    SceneCameraController controller;
    controller.mode = CameraMode::FreeFly;
    controller.focusIndex = 1;
    controller.freeFly.position = Vec3f::zero();
    controller.freeFly.moveSpeed = 5.0f;

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    EXPECT_VEC_APPROX(controller.orbit.target, Vec3f::zero())
        << "focus must not force the orbit target while flying free";
    EXPECT_VEC_APPROX(controller.freeFly.position, (Vec3f{0.0f, 0.0f, -5.0f}));
}

TEST(RendererCamera, SceneCameraControllerSwitchingFromOrbitToFreeFlyPreservesTheView) {
    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.mode = CameraMode::Orbit;
    controller.orbit.target = Vec3f::zero();
    controller.orbit.distance = 10.0f;
    controller.orbit.azimuthRadians = 0.7f;
    controller.orbit.elevationRadians = 0.4f;

    Camera camera;
    const std::vector<NamedSphere> objects{};
    controller.update(camera, objects, input, 1.0f);

    const Vec3f positionBefore = camera.position;
    const Vec3f forwardBefore = normalized(camera.target - camera.position);

    controller.mode = CameraMode::FreeFly;
    InputState noInput;
    noInput.newFrame();
    controller.update(camera, objects, noInput, 0.0f);

    EXPECT_VEC_APPROX(camera.position, positionBefore)
        << "switching camera mode must not move the camera";
    EXPECT_VEC_APPROX(normalized(camera.target - camera.position), forwardBefore)
        << "switching camera mode must not change which way the camera is looking";
}

TEST(RendererCamera, SceneCameraControllerSwitchingFromFreeFlyToOrbitPreservesTheView) {
    InputState input;
    input.newFrame();
    input.onKey(Key::W, ButtonAction::Press, Modifiers{});

    SceneCameraController controller;
    controller.mode = CameraMode::FreeFly;
    controller.freeFly.position = Vec3f::zero();
    controller.freeFly.yawRadians = 0.9f;
    controller.freeFly.pitchRadians = 0.2f;
    controller.freeFly.moveSpeed = 5.0f;
    controller.freeFly.accelerationPerSecond =
        1000.0f;  // effectively instant for this one step

    Camera camera;
    const std::vector<NamedSphere> objects{};
    controller.update(camera, objects, input, 1.0f);

    const Vec3f positionBefore = camera.position;
    const Vec3f forwardBefore = normalized(camera.target - camera.position);

    controller.mode = CameraMode::Orbit;
    InputState noInput;
    noInput.newFrame();
    controller.update(camera, objects, noInput, 0.0f);

    EXPECT_VEC_APPROX(camera.position, positionBefore);
    EXPECT_VEC_APPROX(normalized(camera.target - camera.position), forwardBefore);
    EXPECT_APPROX(length(camera.position - controller.orbit.target),
                  controller.orbit.distance)
        << "the reconstructed pivot must be exactly orbit.distance from the camera";
}

TEST(RendererCamera,
     SceneCameraControllerAAfterSwitchingToFreeFlyMovesRelativeToCurrentView) {
    // The originally reported bug: rotate in Orbit, switch to FreeFly, then
    // strafe -- without the handoff sync, freeFly.yawRadians would still be
    // its own stale default, and A would move in a direction unrelated to
    // wherever Orbit had actually rotated to.
    InputState rotateInput;
    rotateInput.onCursorPosition(0.0, 0.0);
    rotateInput.newFrame();
    rotateInput.onMouseButton(MouseButton::Right, ButtonAction::Press, Modifiers{});
    rotateInput.onCursorPosition(300.0,
                                 0.0);  // a large drag, well away from the default facing

    SceneCameraController controller;
    controller.mode = CameraMode::Orbit;
    controller.orbit.target = Vec3f::zero();
    controller.orbit.distance = 10.0f;

    Camera camera;
    const std::vector<NamedSphere> objects{};
    controller.update(camera, objects, rotateInput, 1.0f);

    controller.mode = CameraMode::FreeFly;
    controller.freeFly.moveSpeed = 5.0f;

    InputState switchInput;
    switchInput.newFrame();
    controller.update(camera, objects, switchInput, 0.0f);  // just the mode switch/sync

    const Vec3f positionAfterSwitch = camera.position;
    const Vec3f rightAfterSwitch =
        normalized(cross(normalized(camera.target - camera.position), Vec3f::unitY()));

    InputState strafeInput;
    strafeInput.newFrame();
    strafeInput.onKey(Key::A, ButtonAction::Press, Modifiers{});
    controller.freeFly.accelerationPerSecond =
        1000.0f;  // effectively instant for this one step
    controller.update(camera, objects, strafeInput, 1.0f);

    const Vec3f displacement = normalized(camera.position - positionAfterSwitch);
    EXPECT_GT(dot(displacement, -rightAfterSwitch), 0.9f)
        << "A must strafe opposite the current (post-rotation) right vector, not a stale "
           "default";
}

TEST(RendererCamera, SceneCameraControllerFirstUpdateDoesNotSyncFromADefaultCamera) {
    // A caller may set mode/povIndex to something other than their own
    // defaults before ever calling update() once -- an Application
    // configuring its camera at setup, same as it already does for e.g.
    // orbit.distance. That first call must respect whatever the caller
    // explicitly configured, not "sync" freeFly from a never-rendered
    // default Camera and clobber it.
    SceneCameraController controller;
    controller.mode = CameraMode::FreeFly;
    controller.freeFly.position = Vec3f{1.0f, 2.0f, 3.0f};
    controller.freeFly.yawRadians = 1.1f;

    InputState input;
    input.newFrame();

    Camera camera;  // freshly default-constructed, deliberately not pre-driven
    const std::vector<NamedSphere> objects{};
    controller.update(camera, objects, input, 0.0f);

    EXPECT_VEC_APPROX(controller.freeFly.position, (Vec3f{1.0f, 2.0f, 3.0f}))
        << "the explicitly configured starting position must survive the first update() "
           "call";
    EXPECT_APPROX(controller.freeFly.yawRadians, 1.1f);
}

TEST(RendererCamera, SceneCameraControllerHandoffWithADegenerateCameraDoesNotProduceNaN) {
    // Position and target coinciding (no well-defined facing direction) is
    // an edge case a caller could leave the camera in before a mode switch
    // -- the handoff sync must degrade gracefully (leave rotation state
    // untouched) rather than propagate a NaN from normalizing a zero vector.
    SceneCameraController controller;
    controller.mode = CameraMode::Orbit;
    controller.orbit.target = Vec3f::zero();
    controller.orbit.distance = 10.0f;

    InputState input;
    input.newFrame();
    Camera camera;
    const std::vector<NamedSphere> objects{};
    controller.update(camera, objects, input, 1.0f);  // establish a real transform first

    camera.position = camera.target;  // now degenerate: zero-length forward

    controller.mode = CameraMode::FreeFly;
    InputState noInput;
    noInput.newFrame();
    controller.update(camera, objects, noInput, 0.0f);

    EXPECT_TRUE(std::isfinite(controller.freeFly.yawRadians));
    EXPECT_TRUE(std::isfinite(controller.freeFly.pitchRadians));
}

TEST(RendererCamera,
     SceneCameraControllerHandoffResetsStaleRollFromAnEarlierFreeFlySession) {
    // rollRadians is FreeFly-only state; every other driver's camera.up is
    // unrolled (Orbit always sets up = worldUp). A stale nonzero roll left
    // over from an earlier FreeFly session surviving the sync would roll
    // the very next FreeFly frame's up out from under the unrolled view the
    // camera was just actually showing -- exactly the snap the handoff sync
    // exists to prevent.
    SceneCameraController controller;
    controller.mode = CameraMode::Orbit;
    controller.orbit.target = Vec3f::zero();
    controller.orbit.distance = 10.0f;
    controller.freeFly.rollRadians = 1.234f;  // left over from an earlier FreeFly session

    InputState input;
    input.newFrame();
    Camera camera;
    const std::vector<NamedSphere> objects{};
    controller.update(camera, objects, input,
                      1.0f);  // establishes m_hasUpdatedBefore, driver=Orbit

    controller.mode = CameraMode::FreeFly;
    InputState noInput;
    noInput.newFrame();
    controller.update(camera, objects, noInput, 0.0f);  // Orbit -> FreeFly handoff sync

    EXPECT_APPROX(controller.freeFly.rollRadians, 0.0f)
        << "the sync must not leave a stale roll from a previous FreeFly session in "
           "place";
}

TEST(RendererCamera, SceneCameraControllerReturningFromPovToFreePreservesTheView) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f{5.0f, 0.0f, 0.0f}, 2.0f},
    };

    InputState input;
    input.newFrame();

    SceneCameraController controller;
    controller.mode = CameraMode::Orbit;
    controller.povIndex =
        0;  // focus-free POV submode places the camera on Earth's surface

    Camera camera;
    controller.update(camera, objects, input, 1.0f);

    const Vec3f positionBefore = camera.position;
    const Vec3f forwardBefore = normalized(camera.target - camera.position);

    controller.povIndex = -1;  // back to Free; mode stays Orbit
    InputState noInput;
    noInput.newFrame();
    controller.update(camera, objects, noInput, 0.0f);

    EXPECT_VEC_APPROX(camera.position, positionBefore);
    EXPECT_VEC_APPROX(normalized(camera.target - camera.position), forwardBefore);
}

TEST(RendererCamera, SceneCameraControllerStatusTextDescribesOrbitMode) {
    const std::vector<NamedSphere> objects{NamedSphere{"Sun", Vec3f::zero(), 5.0f}};
    SceneCameraController controller;
    controller.orbit.distance = 12.5f;

    Camera camera;
    camera.position = {1.0f, 2.0f, 3.0f};
    const std::string text = controller.statusText(camera, objects);

    EXPECT_NE(text.find("Mode: Orbit"), std::string::npos);
    EXPECT_NE(text.find("Position"), std::string::npos);
}

TEST(RendererCamera, SceneCameraControllerStatusTextDescribesFreeFlyMode) {
    const std::vector<NamedSphere> objects{NamedSphere{"Sun", Vec3f::zero(), 5.0f}};
    SceneCameraController controller;
    controller.mode = CameraMode::FreeFly;

    Camera camera;
    const std::string text = controller.statusText(camera, objects);

    EXPECT_NE(text.find("Free fly"), std::string::npos);
}

TEST(RendererCamera, SceneCameraControllerStatusTextDescribesLockedPov) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Earth", Vec3f::zero(), 2.0f},
        NamedSphere{"Moon", Vec3f{10.0f, 0.0f, 0.0f}, 1.0f},
    };
    SceneCameraController controller;
    controller.povIndex = 0;
    controller.focusIndex = 1;

    Camera camera;
    const std::string text = controller.statusText(camera, objects);

    EXPECT_NE(text.find("Earth"), std::string::npos);
    EXPECT_NE(text.find("Moon"), std::string::npos);
    EXPECT_NE(text.find("locked"), std::string::npos);
}

TEST(RendererCamera, SceneCameraControllerStatusTextDescribesFocusFreePov) {
    const std::vector<NamedSphere> objects{NamedSphere{"Earth", Vec3f::zero(), 2.0f}};
    SceneCameraController controller;
    controller.povIndex = 0;

    Camera camera;
    const std::string text = controller.statusText(camera, objects);

    EXPECT_NE(text.find("Earth"), std::string::npos);
    EXPECT_NE(text.find("Height"), std::string::npos);
}

TEST(RendererCamera, SceneCameraControllerOptionListsExcludePovFromFocus) {
    const std::vector<NamedSphere> objects{
        NamedSphere{"Sun", Vec3f::zero(), 5.0f},
        NamedSphere{"Earth", Vec3f{20.0f, 0.0f, 0.0f}, 1.0f},
        NamedSphere{"Moon", Vec3f{22.0f, 0.0f, 0.0f}, 0.3f},
    };

    SceneCameraController controller;
    controller.povIndex = 1;  // Earth

    const std::vector<std::string> pov = controller.povOptions(objects);
    const std::vector<std::string> focus = controller.focusOptions(objects);

    EXPECT_EQ(pov, (std::vector<std::string>{"Free", "Sun", "Earth", "Moon"}));
    EXPECT_EQ(focus, (std::vector<std::string>{"Free", "Sun", "Moon"}));

    EXPECT_EQ(SceneCameraController::indexFromPovSelection(2),
              1);  // "Earth" is pov index 1
    EXPECT_EQ(controller.indexFromFocusSelection(2, objects),
              2);  // "Moon" is object index 2
    EXPECT_EQ(controller.indexFromFocusSelection(0, objects), -1);  // "Free"
}
