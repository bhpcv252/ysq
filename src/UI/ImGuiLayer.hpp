#pragma once

#include <Platform/Window.hpp>

#include <optional>
#include <string>

namespace ysq {

struct ImGuiLayerSettings {
    /// Remember every ImGui window's position, size and collapsed state in
    /// an imgui.ini next to the working directory, across runs. Off by
    /// default: nothing built on this engine currently needs it, and a
    /// stale .ini silently masks whether a panel's *default* layout is
    /// actually right, which matters for anything (like PlotPanel's default
    /// cascade) whose default position is itself part of what's being
    /// verified.
    bool persistLayout = false;
};

/// Owns the ImGui and ImPlot contexts and their GLFW+OpenGL3 backends, tied
/// to one Window.
///
/// Panels draw as an overlay on whatever Renderer already put in the
/// window's framebuffer that frame: beginFrame() starts a new ImGui frame
/// without clearing anything, endFrame() renders every ImGui:: call made
/// since on top of it. Renderer and UI drawing into the same window in the
/// same frame is the whole point; see src/UI/README.md.
///
/// One at a time: ImGui's context is global process state, the same
/// constraint GLAD's loader has, so a second live ImGuiLayer would fight the
/// first over it. The context this was created under must already be
/// current, the same rule as Renderer.
class ImGuiLayer {
public:
    [[nodiscard]] static std::optional<ImGuiLayer>
    create(Window& window, const ImGuiLayerSettings& settings = {},
           std::string* error = nullptr);

    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;
    ImGuiLayer(ImGuiLayer&& other) noexcept;
    ImGuiLayer& operator=(ImGuiLayer&& other) noexcept;
    ~ImGuiLayer();

    /// Starts a new ImGui frame. Call once per frame, before any ImGui::,
    /// Panel, or plot calls.
    void beginFrame();

    /// Renders every call made since beginFrame() into the currently bound
    /// framebuffer.
    void endFrame();

private:
    ImGuiLayer() noexcept = default;
    void destroy() noexcept;

    bool m_valid = false;
};

}  // namespace ysq
