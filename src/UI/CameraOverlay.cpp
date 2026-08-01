#include <UI/CameraOverlay.hpp>

#include <imgui.h>

namespace ysq {

void CameraOverlay::draw() {
    // Below StatsOverlay's own top-left corner, so the two don't overlap;
    // FirstUseEver rather than Always, so a user who drags it stays put.
    ImGui::SetNextWindowPos(ImVec2(10.0f, 90.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Camera", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove);
    ImGui::TextUnformatted(m_statusText.c_str());
    ImGui::End();
}

}  // namespace ysq
