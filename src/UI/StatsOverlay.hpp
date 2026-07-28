#pragma once

#include <cstdint>

namespace ysq {

/// A small always-on-top overlay: smoothed frame time and FPS, plus how many
/// draw calls the frame issued (Renderer::drawCallCount()). Cheap enough to
/// leave on in every application from day one.
class StatsOverlay {
public:
    /// Call once per frame with that frame's wall-clock duration and the
    /// Renderer's draw call count, then draw() inside an ImGuiLayer frame.
    void update(float deltaSeconds, std::uint32_t drawCallCount) noexcept;

    void draw();

private:
    /// Zero means "no sample yet"; update() seeds it on the first call
    /// instead of averaging against a fake zero frame time.
    float m_smoothedFrameSeconds = 0.0f;
    std::uint32_t m_drawCallCount = 0;
};

}  // namespace ysq
