#pragma once

#include <Platform/Input.hpp>
#include <Renderer/Camera.hpp>
#include <Renderer/CameraController.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ysq {

/// Just enough presentation data to stand on or look at a body: no
/// Physics::Body, no application-specific naming. A consumer rebuilds this
/// list fresh every frame from its own bodies, the same way it already
/// rebuilds render positions for draw() calls; see Renderer/README.md's "No
/// scene graph" section.
struct NamedSphere {
    std::string name;
    Vec3f position;
    float radius = 1.0f;
};

enum class CameraMode { Orbit, FreeFly };

/// A complete, drop-in scene camera: Orbit or FreeFly navigation, plus an
/// opt-in POV/Focus mode that stands the camera on one body's surface and
/// looks at another. Free/Free (povIndex == -1) behaves exactly like using
/// `orbit`/`freeFly` directly, so a consumer that never sets `povIndex`
/// gets today's plain navigation with none of this switched on.
///
/// POV, Focus -> behavior:
///   Free, Free   plain orbit/free-fly, unchanged (today's behavior)
///   Free, X      auto-track: in Orbit mode, `orbit.target` snaps to X's
///                position every frame while mouse/scroll still work
///                normally; a no-op in FreeFly mode, where forcing the
///                look direction would fight WASD/mouse steering
///   X, Y         locked: camera anchors to the point on X's surface
///                closest to Y and looks at Y; only scroll (FOV zoom) and
///                R (reset zoom) do anything
///   X, Free      camera stands on X's surface; right-mouse-drag picks
///                which point (reusing OrbitCameraController's own
///                azimuth/elevation math), scroll walks a height axis from
///                the surface up to a "space view" altitude, and the look
///                target blends from looking straight outward near the
///                surface to X's center further out, so it reads as an
///                ordinary view of the body once you've pulled back. R
///                resets angle and height to their defaults.
///
/// X (the POV body) is shown by default in both cases -- `hidePov` is a
/// manual toggle a consumer can bind to e.g. a checkbox, not something
/// this decides automatically.
///
/// See src/Renderer/README.md.
class SceneCameraController {
public:
    OrbitCameraController orbit;
    FreeFlyCameraController freeFly;
    CameraMode mode = CameraMode::Orbit;

    int povIndex = -1;    // -1 = Free
    int focusIndex = -1;  // -1 = Free

    /// Whether the POV body itself is drawn. False (shown) by default --
    /// see the class comment -- a consumer flips this explicitly, e.g. via
    /// a checkbox bound directly to this field.
    bool hidePov = false;

    // POV set, focus free: mouse-picked anchor and altitude.
    float azimuthRadians = 0.0f;
    float elevationRadians = 0.3f;
    float heightRadii =
        0.0f;  // 0 = at the surface; grows in units of the body's own radius

    // POV and focus both set: scroll-driven FOV zoom around whatever
    // vertical FOV the camera had the moment POV+focus locked (captured in
    // m_preLockFovYRadians below), restored once you leave that submode.
    float zoomFovScale = 1.0f;

    float rotateSpeed =
        0.005f;  // radians per pixel of cursor delta, focus-free anchor picking
    float zoomSpeed =
        0.1f;  // scroll sensitivity, shared by the FOV-zoom and height-walk paths

    /// Reads input, drives `camera`. Call once per frame with the scene's
    /// current object list; positions change every frame, but the list
    /// itself is cheap to rebuild the same way an Application already
    /// rebuilds render positions for its own draw() calls. R resets the
    /// active POV submode's live-adjusted state; see the class comment.
    void update(Camera& camera, std::span<const NamedSphere> objects,
                const InputState& input, float deltaSeconds) noexcept;

    /// True only for the current POV object, and only when `hidePov` is
    /// set. Never true for anything else, regardless of `hidePov`.
    [[nodiscard]] bool isHidden(std::size_t objectIndex) const noexcept;

    /// Resets the active POV submode's live-adjusted state to its default
    /// framing (zoom back to 1x, or angle/height back to their defaults).
    /// No effect while POV is Free. Also bound to the R key inside update().
    void reset() noexcept;

    /// Pure-data option list for a POV dropdown: "Free" first, then every
    /// object's name in order. Selecting index i (i > 0) from this list
    /// means `povIndex = indexFromPovSelection(i)`.
    [[nodiscard]] std::vector<std::string>
    povOptions(std::span<const NamedSphere> objects) const;

    /// Like povOptions(), but excludes whichever name is currently
    /// `povIndex`, so a caller cannot select the same body for both.
    /// Because an entry may be skipped, a selection against this list is
    /// not directly an object index -- translate it with
    /// indexFromFocusSelection().
    [[nodiscard]] std::vector<std::string>
    focusOptions(std::span<const NamedSphere> objects) const;

    /// Translates a selection made against povOptions() (0 = Free, else
    /// that list's index) into the value povIndex expects.
    [[nodiscard]] static constexpr int indexFromPovSelection(int selection) noexcept {
        return selection - 1;
    }

    /// Translates a selection made against focusOptions() (0 = Free, else
    /// that list's index, which skips whatever object is currently POV)
    /// into the value focusIndex expects.
    [[nodiscard]] int
    indexFromFocusSelection(int selection,
                            std::span<const NamedSphere> objects) const noexcept;

    /// A short, human-readable multi-line summary of the camera's current
    /// state -- position, plus whichever of mode/speed/POV detail is
    /// relevant right now -- for a HUD like UI::CameraOverlay. Plain text
    /// rather than a shared struct type: Renderer and UI are peers (neither
    /// may depend on the other), and a std::string needs no shared type
    /// definition on either side.
    [[nodiscard]] std::string statusText(const Camera& camera,
                                         std::span<const NamedSphere> objects) const;

private:
    // The locked (POV+focus both set) submode is the only path that writes
    // camera.perspectiveSettings.fovYRadians. These let update() restore it
    // the frame that submode stops being active, instead of leaving the
    // camera zoomed in forever once POV/focus changes away from it.
    bool m_wasLocked = false;
    float m_preLockFovYRadians = 0.0f;

    // Both POV submodes derive camera.up continuously from the previous
    // frame's own up hint (see continuousUpHint() in the .cpp) rather than
    // switching between fixed world axes near a pole, which would flip the
    // view as a tracked direction sweeps through vertical. Resets to a
    // predictable Vec3f::unitY() whenever POV newly engages or switches to
    // a different body, tracked via m_lastPovIndexForUp.
    Vec3f m_lastUp = Vec3f::unitY();
    int m_lastPovIndexForUp = -1;

    // Which of orbit/freeFly/POV actually placed the camera as of the last
    // update() call. update() compares this against what's about to drive
    // the camera this frame and, on a change, carries the camera's current
    // position/facing into whichever of orbit/freeFly is taking over (see
    // syncOrbitFromCamera()/syncFreeFlyFromCamera() in the .cpp) -- without
    // this, switching mode (or returning from POV to Free) would snap the
    // view to whichever controller's own, possibly stale, state.
    enum class DrivingMechanism { Orbit, FreeFly, Pov };
    DrivingMechanism m_lastDriver =
        DrivingMechanism::Orbit;  // matches mode's own default

    // False only before update() has ever run. A caller may set mode/
    // povIndex to something other than their own defaults before the very
    // first update() call (an Application configuring its camera at
    // setup, exactly like existing code already does for e.g.
    // orbit.distance) -- that first call must not "sync" from a
    // never-actually-rendered default Camera and clobber whatever the
    // caller explicitly configured. From the second call on, a driver
    // change is a real mid-session switch and does get synced.
    bool m_hasUpdatedBefore = false;
};

}  // namespace ysq
