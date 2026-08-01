#pragma once

#include <string>
#include <utility>

namespace ysq {

/// A small always-on-top overlay for a camera's current status: position
/// plus whatever mode-specific detail (speed, POV/Focus, zoom, ...) the
/// caller hands it. Content is plain text a Renderer-side source builds
/// (e.g. SceneCameraController::statusText()) and hands in whole, the same
/// way StatsOverlay takes plain numbers rather than a Renderer type --
/// Renderer and UI are peers, so UI cannot depend on Renderer to know a
/// richer type. Cheap enough to leave on in every application, same as
/// StatsOverlay.
class CameraOverlay {
public:
    /// Call once per frame with the latest status text, then draw() inside
    /// an ImGuiLayer frame.
    void update(std::string statusText) { m_statusText = std::move(statusText); }

    void draw();

private:
    std::string m_statusText;
};

}  // namespace ysq
