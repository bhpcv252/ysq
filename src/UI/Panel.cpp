#include <UI/Panel.hpp>

#include <imgui.h>

#include <cfloat>
#include <cstddef>
#include <utility>

namespace ysq {

namespace {

// A window narrow enough to fit next to a 3D view can't also fit a slider
// bar plus its full inline label; ImGui doesn't wrap a widget's own label,
// it just clips whatever doesn't fit. So the label is drawn on its own
// line first, with real word-wrapping, and the widget itself follows at
// full width with an empty (but still ID-unique) label -- "##<label>"
// keeps the text after "##" out of what's displayed while still making
// the widget's ID depend on it.
void wrappedLabel(const std::string& label) {
    ImGui::TextWrapped("%s", label.c_str());
    ImGui::SetNextItemWidth(-FLT_MIN);
}

}  // namespace

void Panel::slider(std::string label, float& value, float min, float max) {
    m_widgets.push_back({[label = std::move(label), &value, min, max] {
        wrappedLabel(label);
        ImGui::SliderFloat(("##" + label).c_str(), &value, min, max);
    }});
}

void Panel::slider(std::string label, int& value, int min, int max) {
    m_widgets.push_back({[label = std::move(label), &value, min, max] {
        wrappedLabel(label);
        ImGui::SliderInt(("##" + label).c_str(), &value, min, max);
    }});
}

void Panel::inputFloat(std::string label, float& value, float step, float stepFast,
                       const char* format) {
    m_widgets.push_back({[label = std::move(label), &value, step, stepFast, format] {
        wrappedLabel(label);
        ImGui::InputFloat(("##" + label).c_str(), &value, step, stepFast, format);
    }});
}

void Panel::checkbox(std::string label, bool& value) {
    m_widgets.push_back(
        {[label = std::move(label), &value] { ImGui::Checkbox(label.c_str(), &value); }});
}

void Panel::colorEdit(std::string label, Vec3f& value) {
    m_widgets.push_back({[label = std::move(label), &value] {
        wrappedLabel(label);
        ImGui::ColorEdit3(("##" + label).c_str(), &value.x);
    }});
}

void Panel::text(std::string label, std::string& value) {
    m_widgets.push_back({[label = std::move(label), &value] {
        ImGui::TextWrapped("%s: %s", label.c_str(), value.c_str());
    }});
}

void Panel::button(std::string label, std::function<void()> onClick) {
    m_widgets.push_back({[label = std::move(label), onClick = std::move(onClick)] {
        if (ImGui::Button(label.c_str()) && onClick) {
            onClick();
        }
    }});
}

void Panel::combo(std::string label, std::vector<std::string> options, int& selected) {
    m_widgets.push_back(
        {[label = std::move(label), options = std::move(options), &selected] {
            const bool hasCurrent =
                selected >= 0 && static_cast<std::size_t>(selected) < options.size();
            const char* preview =
                hasCurrent ? options[static_cast<std::size_t>(selected)].c_str() : "";
            wrappedLabel(label);
            if (ImGui::BeginCombo(("##" + label).c_str(), preview)) {
                for (std::size_t i = 0; i < options.size(); ++i) {
                    const bool isSelected =
                        hasCurrent && static_cast<std::size_t>(selected) == i;
                    if (ImGui::Selectable(options[i].c_str(), isSelected)) {
                        selected = static_cast<int>(i);
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }});
}

void Panel::comboLive(std::string label, std::vector<std::string>& options,
                      int& selected) {
    m_widgets.push_back({[label = std::move(label), &options, &selected] {
        const bool hasCurrent =
            selected >= 0 && static_cast<std::size_t>(selected) < options.size();
        const char* preview =
            hasCurrent ? options[static_cast<std::size_t>(selected)].c_str() : "";
        wrappedLabel(label);
        if (ImGui::BeginCombo(("##" + label).c_str(), preview)) {
            for (std::size_t i = 0; i < options.size(); ++i) {
                const bool isSelected =
                    hasCurrent && static_cast<std::size_t>(selected) == i;
                if (ImGui::Selectable(options[i].c_str(), isSelected)) {
                    selected = static_cast<int>(i);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }});
}

void Panel::draw() {
    // Anchored to the top-right corner by its right edge (a pivot of
    // (1, 0)) rather than a fixed x: Panel auto-sizes to whatever controls
    // were bound, so anchoring by the left edge instead would leave a gap
    // or run off-screen depending on how wide that turns out to be.
    //
    // ImGuiCond_FirstUseEver would only place it once, ever, so resizing
    // the window later would leave it pinned to a corner that no longer
    // exists. Re-anchoring only when the display size actually changed
    // (ImGuiCond_Always, gated by hand) gets the corner to follow a resize
    // without fighting the user's own drag in between.
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    if (displaySize.x != m_lastDisplayWidth || displaySize.y != m_lastDisplayHeight) {
        ImGui::SetNextWindowPos(ImVec2(displaySize.x - 10.0f, 10.0f), ImGuiCond_Always,
                                ImVec2(1.0f, 0.0f));
        m_lastDisplayWidth = displaySize.x;
        m_lastDisplayHeight = displaySize.y;
    }
    // Wide enough for a slider bar and a short label on one line; capped
    // well short of the display so a long label wraps to more lines
    // instead of the window just growing to fit it. AlwaysAutoResize then
    // grows or shrinks the *height* to fit however many lines that turns
    // into, rather than leaving dead space or clipping.
    ImGui::SetNextWindowSizeConstraints(ImVec2(220.0f, 0.0f), ImVec2(360.0f, FLT_MAX));
    ImGui::Begin(m_title.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    for (const Widget& widget : m_widgets) {
        widget.draw();
    }
    ImGui::End();
}

}  // namespace ysq
