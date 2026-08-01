#include <UI/ImGuiLayer.hpp>
#include <UI/PlotPanel.hpp>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include <utility>

namespace ysq {

std::optional<ImGuiLayer> ImGuiLayer::create(Window& window,
                                             const ImGuiLayerSettings& settings,
                                             std::string* error) {
    IMGUI_CHECKVERSION();
    ImGuiContext* context = ImGui::CreateContext();
    if (context == nullptr) {
        if (error) {
            *error = "ImGui::CreateContext failed";
        }
        return std::nullopt;
    }
    if (!settings.persistLayout) {
        // No ini file: a fresh run gets the same default layout every time
        // rather than one that quietly depends on whatever a previous run
        // (or a previous version of that run's own default positions) left
        // behind.
        ImGui::GetIO().IniFilename = nullptr;
    }
    // A new session starts plots back at the first cascade slot, regardless
    // of how many a previous session (or, in a test binary, a previous
    // TEST()) already constructed.
    detail::resetPlotLayoutCascade();
    ImPlot::CreateContext();

    if (!ImGui_ImplGlfw_InitForOpenGL(window.nativeHandle(), true)) {
        ImPlot::DestroyContext();
        ImGui::DestroyContext(context);
        if (error) {
            *error = "ImGui_ImplGlfw_InitForOpenGL failed";
        }
        return std::nullopt;
    }
    // 410, not 130 or the default: this must match the version Window
    // actually requests, and Platform's ContextSettings default is 4.1
    // (macOS's ceiling). See src/Platform/Window.hpp.
    if (!ImGui_ImplOpenGL3_Init("#version 410")) {
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext(context);
        if (error) {
            *error = "ImGui_ImplOpenGL3_Init failed";
        }
        return std::nullopt;
    }

    ImGuiLayer layer;
    layer.m_valid = true;
    return std::optional<ImGuiLayer>{std::move(layer)};
}

ImGuiLayer::ImGuiLayer(ImGuiLayer&& other) noexcept
    : m_valid(std::exchange(other.m_valid, false)) {}

ImGuiLayer& ImGuiLayer::operator=(ImGuiLayer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();
    m_valid = std::exchange(other.m_valid, false);
    return *this;
}

ImGuiLayer::~ImGuiLayer() {
    destroy();
}

void ImGuiLayer::destroy() noexcept {
    if (m_valid) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        m_valid = false;
    }
}

void ImGuiLayer::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool ImGuiLayer::wantsMouseCapture() const noexcept {
    return ImGui::GetIO().WantCaptureMouse;
}

}  // namespace ysq
