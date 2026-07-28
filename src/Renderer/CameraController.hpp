#pragma once

#include <Platform/Input.hpp>
#include <Renderer/Camera.hpp>

namespace ysq {

/// Rotates and zooms a Camera around a fixed target: hold the right mouse
/// button to look, scroll to zoom. The natural fit for a scene with an
/// obvious center, which is most of what this engine renders — a star, a
/// black hole, a single body.
struct OrbitCameraController {
    Vec3f target = Vec3f::zero();
    float distance = 10.0f;
    /// Around the up axis; zero looks down -Z.
    float azimuthRadians = 0.0f;
    /// Clamped away from the poles in update(), where the up vector would
    /// otherwise become degenerate.
    float elevationRadians = 0.3f;
    float rotateSpeed = 0.005f;  // radians per pixel of cursor delta
    float zoomSpeed = 0.1f;      // fraction of current distance per scroll notch
    float minDistance = 0.01f;

    /// Reads mouse input from `input` and writes the resulting eye/target/up
    /// into `camera`. Call once per frame, after input.newFrame() and
    /// Platform::pollEvents() have run.
    void update(Camera& camera, const InputState& input) noexcept;
};

/// A first-person free-fly camera: WASD moves relative to the look
/// direction, Q/E move straight down/up, the right mouse button looks
/// around, and Shift moves faster. The fit for exploring a scene with no
/// single center, such as a galaxy collision.
struct FreeFlyCameraController {
    Vec3f position = Vec3f::zero();
    /// Zero yaw looks down -Z, matching Camera's default forward().
    float yawRadians = -1.5707963f;
    float pitchRadians = 0.0f;
    float lookSpeed = 0.0025f;    // radians per pixel of cursor delta
    float moveSpeed = 5.0f;       // units per second
    float fastMultiplier = 4.0f;  // applied while Shift is held

    /// Reads keyboard and mouse input from `input` and writes the resulting
    /// eye/target/up into `camera`. `deltaSeconds` is the frame's wall-clock
    /// duration, since movement (unlike look, which is a direct cursor
    /// mapping) is rate-based.
    void update(Camera& camera, const InputState& input, float deltaSeconds) noexcept;
};

}  // namespace ysq
