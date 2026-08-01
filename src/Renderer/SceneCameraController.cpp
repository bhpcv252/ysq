#include <Renderer/SceneCameraController.hpp>

#include <Math/Scalar.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>

namespace ysq {

namespace {

// A small outward nudge off the exact surface, so the anchor point never
// sits exactly on the mesh (which would z-fight against it).
constexpr float kSurfaceEpsilonRadii = 0.001f;

// By this height, the look target has fully blended from "straight
// outward" to "the body's center", so continuing to pull back reads as an
// ordinary view of the body rather than a snap.
constexpr float kLookBlendMaxHeightRadii = 1.0f;

constexpr float kMaxHeightRadii = 20.0f;      // "space view" scroll ceiling
constexpr float kMinHeightStepRadii = 0.01f;  // scroll step floor near the surface

constexpr float kMaxZoomFovScale = 20.0f;  // binocular-style zoom ceiling

float smoothstep01(float t) noexcept {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Derives a continuous up hint from `forward` and the previous frame's own
// hint, instead of switching between two unrelated fixed axes near a pole
// -- that discontinuous switch is what causes a visible flip as a tracked
// direction sweeps through vertical. Falls back to the previous hint
// unchanged only in the practically-unreachable instant `forward` is
// exactly parallel to it (needed for NaN-safety, not for the general
// near-pole case, which this handles smoothly on its own).
Vec3f continuousUpHint(const Vec3f& forward, Vec3f& lastUp) noexcept {
    if (const auto side = tryNormalized(cross(forward, lastUp))) {
        lastUp = cross(*side, forward);
    }
    return lastUp;
}

// Carries the camera's actual last position/facing (wherever it came
// from -- orbit, freeFly, or POV, it doesn't matter) into freeFly, so
// switching into FreeFly doesn't snap the view to freeFly's own separate,
// possibly stale state. Position always syncs; if target and position
// happen to coincide (no well-defined facing direction), yaw/pitch are
// left as freeFly's own previous values rather than propagating NaN --
// the same defensive tryNormalized() pattern continuousUpHint() already
// uses above, since a degenerate direction is a real, if rare,
// possibility (e.g. a caller-constructed Camera before anything has
// actually rendered it).
void syncFreeFlyFromCamera(const Camera& camera,
                           FreeFlyCameraController& freeFly) noexcept {
    freeFly.position = camera.position;
    freeFly.velocity = Vec3f::zero();  // no stale drift from a much earlier session
    // rollRadians is FreeFly-only state: no other driver's camera.up carries
    // a "banked" interpretation, so there is nothing to sync it from. Reset
    // it rather than leaving a stale value from an earlier FreeFly session
    // in place -- that would roll the very next frame's up out from under
    // whatever unrolled view the camera was just actually showing.
    freeFly.rollRadians = 0.0f;
    if (const auto forward = tryNormalized(camera.target - camera.position)) {
        freeFly.pitchRadians = std::asin(std::clamp(forward->y, -1.0f, 1.0f));
        freeFly.yawRadians = std::atan2(forward->z, forward->x);
    }
}

// Carries the camera's actual last position/facing into orbit, keeping
// orbit's own distance (the only length scale available), so the pivot
// lands exactly `distance` in front of the current facing direction and
// reconstructing camera.position from it lands back exactly where the
// camera already was. Leaves orbit's target/azimuth/elevation untouched
// in the degenerate zero-direction case, for the same reason
// syncFreeFlyFromCamera() does.
void syncOrbitFromCamera(const Camera& camera, OrbitCameraController& orbit) noexcept {
    const auto forward = tryNormalized(camera.target - camera.position);
    if (!forward) {
        return;
    }
    orbit.target = camera.position + *forward * orbit.distance;
    const Vec3f direction = -*forward;  // orbit's direction points target -> camera
    orbit.elevationRadians =
        std::clamp(std::asin(std::clamp(direction.y, -1.0f, 1.0f)),
                   -kMaxCameraElevationRadians, kMaxCameraElevationRadians);
    // atan2(x, z), not the more usual atan2(y, x): matches
    // directionFromAzimuthElevation()'s own convention, where
    // direction.x = cosElevation*sin(azimuth) and direction.z =
    // cosElevation*cos(azimuth).
    orbit.azimuthRadians = std::atan2(direction.x, direction.z);
}

}  // namespace

void SceneCameraController::update(Camera& camera, std::span<const NamedSphere> objects,
                                   const InputState& input, float deltaSeconds) noexcept {
    if (input.keyPressed(Key::R)) {
        reset();
    }

    const bool povValid =
        povIndex >= 0 && static_cast<std::size_t>(povIndex) < objects.size();
    const bool hasFocus = povValid && focusIndex >= 0 &&
                          static_cast<std::size_t>(focusIndex) < objects.size() &&
                          focusIndex != povIndex;

    // The locked submode is the only path that writes fovYRadians; restore
    // it as soon as that submode is no longer active, or the camera would
    // stay zoomed in even after POV/focus moves away from it.
    if (!hasFocus && m_wasLocked) {
        camera.perspectiveSettings.fovYRadians = m_preLockFovYRadians;
        m_wasLocked = false;
    }

    const DrivingMechanism thisFrameDriver =
        povValid ? DrivingMechanism::Pov
                 : (mode == CameraMode::Orbit ? DrivingMechanism::Orbit
                                              : DrivingMechanism::FreeFly);
    if (thisFrameDriver != m_lastDriver && m_hasUpdatedBefore) {
        // Whichever of orbit/freeFly is about to take over inherits the
        // camera's actual last transform, so switching mode (or returning
        // from POV to Free) never snaps the view. POV needs no sync
        // target-side -- its own position is fully determined by body
        // positions, not stored rotation state.
        if (thisFrameDriver == DrivingMechanism::FreeFly) {
            syncFreeFlyFromCamera(camera, freeFly);
        } else if (thisFrameDriver == DrivingMechanism::Orbit) {
            syncOrbitFromCamera(camera, orbit);
        }
    }
    m_lastDriver = thisFrameDriver;
    m_hasUpdatedBefore = true;

    if (!povValid) {
        const bool focusValid =
            focusIndex >= 0 && static_cast<std::size_t>(focusIndex) < objects.size();
        if (focusValid && mode == CameraMode::Orbit) {
            orbit.target = objects[static_cast<std::size_t>(focusIndex)].position;
        }
        if (mode == CameraMode::Orbit) {
            orbit.update(camera, input);
        } else {
            freeFly.update(camera, input, deltaSeconds);
        }
        return;
    }

    if (povIndex != m_lastPovIndexForUp) {
        m_lastUp = Vec3f::unitY();
    }
    m_lastPovIndexForUp = povIndex;

    const NamedSphere& pov = objects[static_cast<std::size_t>(povIndex)];

    if (hasFocus) {
        if (!m_wasLocked) {
            m_preLockFovYRadians = camera.perspectiveSettings.fovYRadians;
            m_wasLocked = true;
        }

        const NamedSphere& focus = objects[static_cast<std::size_t>(focusIndex)];
        const Vec3f toFocus = focus.position - pov.position;
        const float focusDistance = length(toFocus);
        const Vec3f outward =
            (focusDistance > 0.0f) ? toFocus / focusDistance : Vec3f::unitY();
        const Vec3f anchor =
            pov.position + outward * (pov.radius * (1.0f + kSurfaceEpsilonRadii));

        const ScrollOffset scroll = input.scrollDelta();
        zoomFovScale *= (1.0f + zoomSpeed * static_cast<float>(scroll.y));
        zoomFovScale = std::clamp(zoomFovScale, 1.0f, kMaxZoomFovScale);

        camera.position = anchor;
        camera.target = focus.position;
        camera.up = continuousUpHint(outward, m_lastUp);
        camera.perspectiveSettings.fovYRadians = m_preLockFovYRadians / zoomFovScale;
        return;
    }

    if (input.mouseButtonDown(MouseButton::Right)) {
        const CursorPosition delta = input.cursorDelta();
        azimuthRadians -= static_cast<float>(delta.x) * rotateSpeed;
        elevationRadians -= static_cast<float>(delta.y) * rotateSpeed;
        elevationRadians = std::clamp(elevationRadians, -kMaxCameraElevationRadians,
                                      kMaxCameraElevationRadians);
    }

    const ScrollOffset scroll = input.scrollDelta();
    heightRadii +=
        zoomSpeed * static_cast<float>(scroll.y) * (heightRadii + kMinHeightStepRadii);
    heightRadii = std::clamp(heightRadii, 0.0f, kMaxHeightRadii);

    const Vec3f direction =
        directionFromAzimuthElevation(azimuthRadians, elevationRadians);
    const Vec3f position = pov.position + direction * (pov.radius * (1.0f + heightRadii));
    const Vec3f outwardTarget = position + direction;
    const float blend = smoothstep01(heightRadii / kLookBlendMaxHeightRadii);
    const Vec3f target = lerp(outwardTarget, pov.position, blend);

    camera.position = position;
    camera.target = target;
    camera.up = continuousUpHint(normalized(target - position), m_lastUp);
}

bool SceneCameraController::isHidden(std::size_t objectIndex) const noexcept {
    return hidePov && povIndex >= 0 && static_cast<std::size_t>(povIndex) == objectIndex;
}

void SceneCameraController::reset() noexcept {
    if (povIndex < 0) {
        return;
    }
    if (focusIndex >= 0 && focusIndex != povIndex) {
        zoomFovScale = 1.0f;
    } else {
        azimuthRadians = 0.0f;
        elevationRadians = 0.3f;
        heightRadii = 0.0f;
    }
}

std::vector<std::string>
SceneCameraController::povOptions(std::span<const NamedSphere> objects) const {
    std::vector<std::string> options;
    options.reserve(objects.size() + 1);
    options.emplace_back("Free");
    for (const NamedSphere& object : objects) {
        options.push_back(object.name);
    }
    return options;
}

std::vector<std::string>
SceneCameraController::focusOptions(std::span<const NamedSphere> objects) const {
    std::vector<std::string> options;
    options.reserve(objects.size() + 1);
    options.emplace_back("Free");
    for (std::size_t i = 0; i < objects.size(); ++i) {
        if (static_cast<int>(i) == povIndex) {
            continue;
        }
        options.push_back(objects[i].name);
    }
    return options;
}

std::string
SceneCameraController::statusText(const Camera& camera,
                                  std::span<const NamedSphere> objects) const {
    std::string text =
        std::format("Position: ({:.3f}, {:.3f}, {:.3f})\n", camera.position.x,
                    camera.position.y, camera.position.z);

    const bool povValid =
        povIndex >= 0 && static_cast<std::size_t>(povIndex) < objects.size();
    if (!povValid) {
        if (mode == CameraMode::Orbit) {
            text += std::format("Mode: Orbit\nDistance: {:.3f}\nAzimuth: {:.1f} deg  "
                                "Elevation: {:.1f} deg",
                                orbit.distance, degrees(orbit.azimuthRadians),
                                degrees(orbit.elevationRadians));
        } else {
            text += std::format(
                "Mode: Free fly\nMove speed: {:.3f} u/s\nCurrent speed: {:.3f} u/s\n"
                "Look-locked: {}  Roll: {:.1f} deg",
                freeFly.moveSpeed, length(freeFly.velocity),
                freeFly.lookLocked ? "yes" : "no", degrees(freeFly.rollRadians));
        }
        return text;
    }

    const std::string& povName = objects[static_cast<std::size_t>(povIndex)].name;
    const bool hasFocus = focusIndex >= 0 &&
                          static_cast<std::size_t>(focusIndex) < objects.size() &&
                          focusIndex != povIndex;
    if (hasFocus) {
        const std::string& focusName = objects[static_cast<std::size_t>(focusIndex)].name;
        text += std::format(
            "Mode: POV\nStanding on: {}\nLooking at: {} (locked)\nZoom: {:.2f}x", povName,
            focusName, zoomFovScale);
    } else {
        text += std::format(
            "Mode: POV\nStanding on: {}\nHeight: {:.3f} radii\nAzimuth: {:.1f} deg  "
            "Elevation: {:.1f} deg",
            povName, heightRadii, degrees(azimuthRadians), degrees(elevationRadians));
    }
    return text;
}

int SceneCameraController::indexFromFocusSelection(
    int selection, std::span<const NamedSphere> objects) const noexcept {
    if (selection <= 0) {
        return -1;
    }
    int displayIndex = 0;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        if (static_cast<int>(i) == povIndex) {
            continue;
        }
        ++displayIndex;
        if (displayIndex == selection) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

}  // namespace ysq
