#include <UI/Panel.hpp>

#include <imgui.h>

#include <cstddef>
#include <utility>

namespace ysq {

void Panel::slider(std::string label, float& value, float min, float max) {
    m_widgets.push_back({[label = std::move(label), &value, min, max] {
        ImGui::SliderFloat(label.c_str(), &value, min, max);
    }});
}

void Panel::slider(std::string label, int& value, int min, int max) {
    m_widgets.push_back({[label = std::move(label), &value, min, max] {
        ImGui::SliderInt(label.c_str(), &value, min, max);
    }});
}

void Panel::checkbox(std::string label, bool& value) {
    m_widgets.push_back(
        {[label = std::move(label), &value] { ImGui::Checkbox(label.c_str(), &value); }});
}

void Panel::colorEdit(std::string label, Vec3f& value) {
    m_widgets.push_back({[label = std::move(label), &value] {
        ImGui::ColorEdit3(label.c_str(), &value.x);
    }});
}

void Panel::text(std::string label, std::string& value) {
    m_widgets.push_back({[label = std::move(label), &value] {
        ImGui::Text("%s: %s", label.c_str(), value.c_str());
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
            if (ImGui::BeginCombo(label.c_str(), preview)) {
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
    ImGui::Begin(m_title.c_str());
    for (const Widget& widget : m_widgets) {
        widget.draw();
    }
    ImGui::End();
}

}  // namespace ysq
