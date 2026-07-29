#include <UI/StatsOverlay.hpp>

#include <imgui.h>

namespace ysq {

void StatsOverlay::update(float deltaSeconds, std::uint32_t drawCallCount) noexcept {
    // Exponential moving average: one dropped frame would otherwise make a
    // raw per-frame reading unreadable, and this needs no history buffer.
    constexpr float kSmoothing = 0.1f;
    m_smoothedFrameSeconds =
        (m_smoothedFrameSeconds == 0.0f)
            ? deltaSeconds
            : m_smoothedFrameSeconds +
                  kSmoothing * (deltaSeconds - m_smoothedFrameSeconds);
    m_drawCallCount = drawCallCount;
}

void StatsOverlay::draw() {
    // FirstUseEver: a corner to start in, not a place the user is locked to
    // -- there is nothing to lock them to anyway, since NoMove already
    // covers that.
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Stats", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove);
    const float ms = m_smoothedFrameSeconds * 1000.0f;
    const float fps =
        (m_smoothedFrameSeconds > 0.0f) ? 1.0f / m_smoothedFrameSeconds : 0.0f;
    ImGui::Text("%.2f ms (%.0f FPS)", static_cast<double>(ms), static_cast<double>(fps));
    ImGui::Text("%u draw calls", m_drawCallCount);
    ImGui::End();
}

}  // namespace ysq
