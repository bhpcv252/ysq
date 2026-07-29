#pragma once

#include <Math/Vector3.hpp>

#include <functional>
#include <string>
#include <vector>

namespace ysq {

/// A generic, declarative list of controls bound to plain references, so the
/// same panel machinery drives whatever an application's simulation
/// parameters turn out to be without UI having to know what a parameter is.
///
/// Bindings are built once, typically at setup, against long-lived
/// references (an Application's own state); draw() reads through them fresh
/// every call, the same way ImGui itself works.
///
///     ysq::Panel panel("Simulation");
///     panel.slider("Time scale", timeScale, 0.0f, 10.0f);
///     panel.checkbox("Paused", paused);
///     // once per frame, inside an ImGuiLayer frame:
///     panel.draw();
class Panel {
public:
    explicit Panel(std::string title) : m_title(std::move(title)) {}

    void slider(std::string label, float& value, float min, float max);
    void slider(std::string label, int& value, int min, int max);
    void checkbox(std::string label, bool& value);
    void colorEdit(std::string label, Vec3f& value);
    /// A live readout: `value` is read fresh every draw(), so this reflects
    /// whatever the caller last wrote to it, the same as the bindings above.
    /// Takes a non-const reference, like the other bindings, even though
    /// nothing here writes through it: a `const&` would silently accept a
    /// temporary (`text("FPS", std::to_string(fps))`), and the stored lambda
    /// capturing that temporary's address would dangle from the very next
    /// draw() call. A named `std::string` the caller keeps alive is required
    /// instead, the same as slider/checkbox/colorEdit already require.
    void text(std::string label, std::string& value);
    void button(std::string label, std::function<void()> onClick);
    /// `options` and `selected` share an index. `selected` out of range is
    /// drawn as no current selection; this does not clamp it.
    void combo(std::string label, std::vector<std::string> options, int& selected);

    /// Draws every bound control inside one ImGui window titled title().
    /// Call inside an ImGuiLayer frame (between beginFrame() and endFrame()).
    void draw();

    [[nodiscard]] const std::string& title() const noexcept { return m_title; }

private:
    struct Widget {
        std::function<void()> draw;
    };

    std::string m_title;
    std::vector<Widget> m_widgets;
    /// The display size draw() last anchored its default position against.
    /// Negative means never: the very first draw() always counts as a
    /// change. See Panel.cpp.
    float m_lastDisplayWidth = -1.0f;
    float m_lastDisplayHeight = -1.0f;
};

}  // namespace ysq
