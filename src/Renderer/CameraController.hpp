#pragma once

#include <Platform/Input.hpp>
#include <Renderer/Camera.hpp>

namespace ysq {

/// The elevation clamp every controller here uses, in radians: just short of
/// the poles, where the up vector this module builds from azimuth/elevation
/// becomes degenerate.
inline constexpr float kMaxCameraElevationRadians = 1.5533431f;  // 89 degrees

/// Direction from azimuth/elevation, matching OrbitCameraController's own
/// convention: zero azimuth and elevation looks down -Z. Shared with
/// SceneCameraController's surface-anchor math (see SceneCameraController.hpp),
/// which picks a point on a sphere the same way OrbitCameraController picks a
/// viewing direction.
[[nodiscard]] Vec3f directionFromAzimuthElevation(float azimuthRadians,
                                                  float elevationRadians) noexcept;

/// Rotates and zooms a Camera around a fixed target: hold the right mouse
/// button to look, hold the left to pan (drag the target sideways, content
/// follows the cursor, à la Figma), scroll to zoom. The natural fit for a
/// scene with an obvious center, which is most of what this engine renders
/// — a star, a black hole, a single body.
struct OrbitCameraController {
    Vec3f target = Vec3f::zero();
    float distance = 10.0f;
    /// Around the up axis; zero looks down -Z.
    float azimuthRadians = 0.0f;
    /// Clamped away from the poles in update(), where the up vector would
    /// otherwise become degenerate.
    float elevationRadians = 0.3f;
    float rotateSpeed = 0.005f;  // radians per pixel of cursor delta
    /// World units of target movement per render-distance-unit, per pixel
    /// of left-drag -- scaled by distance so a drag pans roughly the same
    /// fraction of the view regardless of current zoom.
    float panSpeed = 0.0015f;
    float zoomSpeed = 0.1f;  // fraction of current distance per scroll notch
    float minDistance = 0.01f;

    /// Reads mouse input from `input` and writes the resulting eye/target/up
    /// into `camera`. Call once per frame, after input.newFrame() and
    /// Platform::pollEvents() have run.
    void update(Camera& camera, const InputState& input) noexcept;
};

/// A first-person free-fly camera: WASD moves relative to the look
/// direction, Q/E move straight down/up, the right mouse button (or the T
/// toggle, see lookLocked) looks around, the left mouse button pans (slides
/// the camera sideways without rotating, à la Figma), and Shift moves
/// faster. The fit for exploring a scene with no single center, such as a
/// galaxy collision, or for crossing the enormous scale range between an
/// AU-wide orbit and a planet's own surface, which scroll's speed control
/// exists for.
struct FreeFlyCameraController {
    Vec3f position = Vec3f::zero();
    /// Zero yaw looks down -Z, matching Camera's default forward().
    float yawRadians = -1.5707963f;
    float pitchRadians = 0.0f;
    /// Accumulated rotation of `up` about the forward axis, from holding
    /// Z/C. True 6-DOF: A/D, Q/E, and pan's vertical component are all
    /// relative to the *current rolled* orientation, not fixed world-up,
    /// so rolling 90 degrees and then pressing A moves along whatever
    /// "left" now means from the rolled view.
    float rollRadians = 0.0f;
    float lookSpeed = 0.0025f;       // radians per pixel of cursor delta
    float rollSpeed = 1.5f;          // radians per second while Z or C is held
    float moveSpeed = 5.0f;          // units per second
    float minMoveSpeed = 1.0e-4f;    // scroll never drives moveSpeed to zero or negative
    float scrollSpeedFactor = 0.1f;  // fraction of current moveSpeed per scroll notch
    /// World units per pixel of left-drag, per unit of moveSpeed -- pan
    /// distance tracks whatever flight speed is already dialed in via
    /// scroll, instead of being a disconnected constant.
    float panSpeed = 0.002f;
    float fastMultiplier = 4.0f;  // applied while Shift is held
    /// Units/second^2 the current velocity approaches the input-driven
    /// target velocity at: smooths starts and stops instead of an instant
    /// speed change. High enough that one full-second step (as in a fixed
    /// large deltaSeconds) still reaches the target speed within that step.
    float accelerationPerSecond = 8.0f;
    /// T toggles this instead of requiring the right mouse button held, for
    /// long stretches of navigation.
    bool lookLocked = false;
    bool invertY = false;

    /// Smoothed velocity accelerationPerSecond is chasing the input-driven
    /// target with; not meant to be set directly, but not private either,
    /// matching this module's other controllers, which are plain state a
    /// caller can inspect or reset.
    Vec3f velocity = Vec3f::zero();

    /// Reads keyboard and mouse input from `input` and writes the resulting
    /// eye/target/up into `camera`. `deltaSeconds` is the frame's wall-clock
    /// duration, since movement (unlike look, which is a direct cursor
    /// mapping) is rate-based.
    void update(Camera& camera, const InputState& input, float deltaSeconds) noexcept;
};

}  // namespace ysq
