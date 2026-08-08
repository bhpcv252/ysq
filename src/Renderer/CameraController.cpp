#include <Renderer/CameraController.hpp>

#include <algorithm>
#include <cmath>

namespace ysq {

Vec3f directionFromAzimuthElevation(float azimuthRadians,
                                    float elevationRadians) noexcept {
    const float cosElevation = std::cos(elevationRadians);
    return Vec3f{cosElevation * std::sin(azimuthRadians), std::sin(elevationRadians),
                 cosElevation * std::cos(azimuthRadians)};
}

void OrbitCameraController::update(Camera& camera, const InputState& input) noexcept {
    if (input.mouseButtonDown(MouseButton::Right)) {
        const CursorPosition delta = input.cursorDelta();
        azimuthRadians -= static_cast<float>(delta.x) * rotateSpeed;
        elevationRadians -= static_cast<float>(delta.y) * rotateSpeed;
        elevationRadians = std::clamp(elevationRadians, -kMaxCameraElevationRadians,
                                      kMaxCameraElevationRadians);
    }

    const Vec3f direction =
        directionFromAzimuthElevation(azimuthRadians, elevationRadians);

    if (input.mouseButtonDown(MouseButton::Left)) {
        const CursorPosition delta = input.cursorDelta();
        // `direction` points target -> camera (outward); right/trueUp need
        // the camera's actual forward (camera -> target, the opposite),
        // matching Camera::right()/trueUp()'s own convention -- using
        // `direction` directly here would mirror the horizontal pan
        // component.
        const Vec3f forward = -direction;
        const Vec3f right = normalized(cross(forward, Vec3f::unitY()));
        const Vec3f trueUp = cross(right, forward);
        // Content follows the cursor (drag right -> the view slides right),
        // which means target/camera move the opposite way. Scaled by
        // distance so a drag pans roughly the same fraction of the view
        // regardless of current zoom.
        target += (-right * static_cast<float>(delta.x) +
                   trueUp * static_cast<float>(delta.y)) *
                  panSpeed * distance;
    }

    const ScrollOffset scroll = input.scrollDelta();
    distance *= (1.0f - zoomSpeed * static_cast<float>(scroll.y));
    distance = std::max(distance, minDistance);

    camera.target = target;
    camera.position = target + direction * distance;
    camera.up = Vec3f::unitY();
}

void FreeFlyCameraController::update(Camera& camera, const InputState& input,
                                     float deltaSeconds) noexcept {
    if (input.keyPressed(Key::T)) {
        lookLocked = !lookLocked;
    }
    if (lookLocked || input.mouseButtonDown(MouseButton::Right)) {
        const CursorPosition delta = input.cursorDelta();
        // The mouse delta is measured in screen space, which is rolled
        // relative to the unrolled yaw/pitch frame yawRadians/pitchRadians
        // assume -- un-rotate it by the current roll first, or a drag that
        // looks purely horizontal on the rolled screen gets read as pure
        // yaw regardless of how the view is actually rolled. Identity at
        // rollRadians == 0, so this doesn't change anything for anyone not
        // using roll.
        const float cosRoll = std::cos(rollRadians);
        const float sinRoll = std::sin(rollRadians);
        const float rawDx = static_cast<float>(delta.x);
        const float rawDy = static_cast<float>(delta.y);
        const float unrolledDx = rawDx * cosRoll - rawDy * sinRoll;
        const float unrolledDy = rawDx * sinRoll + rawDy * cosRoll;
        yawRadians += unrolledDx * lookSpeed;
        const float pitchDelta = unrolledDy * lookSpeed;
        pitchRadians += invertY ? pitchDelta : -pitchDelta;
        pitchRadians = std::clamp(pitchRadians, -kMaxCameraElevationRadians,
                                  kMaxCameraElevationRadians);
    }

    if (input.keyDown(Key::Z)) {
        rollRadians -= rollSpeed * deltaSeconds;
    }
    if (input.keyDown(Key::C)) {
        rollRadians += rollSpeed * deltaSeconds;
    }

    const Vec3f forward{std::cos(pitchRadians) * std::cos(yawRadians),
                        std::sin(pitchRadians),
                        std::cos(pitchRadians) * std::sin(yawRadians)};
    // worldUp is only the fixed reference roll rotates *from*, computed
    // once per frame from the accumulated rollRadians -- not itself the
    // basis anything else moves relative to. right/up (and so A/D, Q/E,
    // and pan) are relative to the *current rolled* orientation: true
    // 6-DOF, so rolling 90 degrees and then pressing A moves along
    // whatever "left" now means from the rolled view, not the original
    // unrolled one. At rollRadians == 0 (the default) `up` is exactly
    // worldUp, so nothing here changes for anyone not using roll.
    const Vec3f worldUp = Vec3f::unitY();
    const Vec3f up =
        (rollRadians != 0.0f) ? rotateAbout(worldUp, forward, rollRadians) : worldUp;
    const Vec3f right = normalized(cross(forward, up));

    if (input.mouseButtonDown(MouseButton::Left)) {
        const CursorPosition delta = input.cursorDelta();
        // Translation only -- look direction is untouched, so this reads as
        // sliding a photograph rather than turning your head. Content
        // follows the cursor, same convention as OrbitCameraController's
        // pan, scaled by moveSpeed so it tracks whatever flight speed is
        // already dialed in via scroll instead of being a fixed constant.
        position +=
            (-right * static_cast<float>(delta.x) + up * static_cast<float>(delta.y)) *
            panSpeed * moveSpeed;
    }

    const ScrollOffset scroll = input.scrollDelta();
    moveSpeed *= (1.0f + scrollSpeedFactor * static_cast<float>(scroll.y));
    moveSpeed = std::max(moveSpeed, minMoveSpeed);

    float speed = moveSpeed;
    if (input.keyDown(Key::LeftShift) || input.keyDown(Key::RightShift)) {
        speed *= fastMultiplier;
    }

    Vec3f targetVelocity = Vec3f::zero();
    if (input.keyDown(Key::W)) {
        targetVelocity += forward * speed;
    }
    if (input.keyDown(Key::S)) {
        targetVelocity -= forward * speed;
    }
    if (input.keyDown(Key::D)) {
        targetVelocity += right * speed;
    }
    if (input.keyDown(Key::A)) {
        targetVelocity -= right * speed;
    }
    if (input.keyDown(Key::E)) {
        targetVelocity += up * speed;
    }
    if (input.keyDown(Key::Q)) {
        targetVelocity -= up * speed;
    }

    const float approach = std::clamp(accelerationPerSecond * deltaSeconds, 0.0f, 1.0f);
    velocity += (targetVelocity - velocity) * approach;
    position += velocity * deltaSeconds;

    camera.position = position;
    camera.target = position + forward;
    camera.up = up;
}

}  // namespace ysq
