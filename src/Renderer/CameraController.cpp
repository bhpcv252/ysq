#include <Renderer/CameraController.hpp>

#include <algorithm>
#include <cmath>

namespace ysq {

namespace {

// Just short of the poles, where azimuth becomes degenerate against `up`.
constexpr float kMaxElevation = 1.5533431f;  // 89 degrees

}  // namespace

void OrbitCameraController::update(Camera& camera, const InputState& input) noexcept {
    if (input.mouseButtonDown(MouseButton::Right)) {
        const CursorPosition delta = input.cursorDelta();
        azimuthRadians -= static_cast<float>(delta.x) * rotateSpeed;
        elevationRadians -= static_cast<float>(delta.y) * rotateSpeed;
        elevationRadians = std::clamp(elevationRadians, -kMaxElevation, kMaxElevation);
    }

    const ScrollOffset scroll = input.scrollDelta();
    distance *= (1.0f - zoomSpeed * static_cast<float>(scroll.y));
    distance = std::max(distance, minDistance);

    const float cosElevation = std::cos(elevationRadians);
    const Vec3f direction{cosElevation * std::sin(azimuthRadians),
                          std::sin(elevationRadians),
                          cosElevation * std::cos(azimuthRadians)};

    camera.target = target;
    camera.position = target + direction * distance;
    camera.up = Vec3f::unitY();
}

void FreeFlyCameraController::update(Camera& camera, const InputState& input,
                                     float deltaSeconds) noexcept {
    if (input.mouseButtonDown(MouseButton::Right)) {
        const CursorPosition delta = input.cursorDelta();
        yawRadians += static_cast<float>(delta.x) * lookSpeed;
        pitchRadians -= static_cast<float>(delta.y) * lookSpeed;
        pitchRadians = std::clamp(pitchRadians, -kMaxElevation, kMaxElevation);
    }

    const Vec3f forward{std::cos(pitchRadians) * std::cos(yawRadians),
                        std::sin(pitchRadians),
                        std::cos(pitchRadians) * std::sin(yawRadians)};
    const Vec3f worldUp = Vec3f::unitY();
    const Vec3f right = normalized(cross(forward, worldUp));

    float speed = moveSpeed;
    if (input.keyDown(Key::LeftShift) || input.keyDown(Key::RightShift)) {
        speed *= fastMultiplier;
    }
    const float step = speed * deltaSeconds;

    if (input.keyDown(Key::W)) {
        position += forward * step;
    }
    if (input.keyDown(Key::S)) {
        position -= forward * step;
    }
    if (input.keyDown(Key::D)) {
        position += right * step;
    }
    if (input.keyDown(Key::A)) {
        position -= right * step;
    }
    if (input.keyDown(Key::E)) {
        position += worldUp * step;
    }
    if (input.keyDown(Key::Q)) {
        position -= worldUp * step;
    }

    camera.position = position;
    camera.target = position + forward;
    camera.up = worldUp;
}

}  // namespace ysq
