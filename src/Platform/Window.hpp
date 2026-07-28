#pragma once

#include <Platform/Input.hpp>
#include <Platform/Platform.hpp>

#include <functional>
#include <optional>
#include <string>
#include <string_view>

/// Declared rather than included: GLFW stays out of this header, so nothing that
/// consumes Platform compiles it. nativeHandle() is the escape hatch for the one
/// place that genuinely needs it, Dear ImGui's GLFW backend.
struct GLFWwindow;

namespace ysq {

namespace detail {

/// The C callbacks the windowing system calls back through. Defined in
/// Window.cpp, befriended below so they can reach a window's state without
/// widening its public interface or naming GLFW types here.
struct WindowCallbacks;

}  // namespace detail

struct GLVersion {
    int major = 0;
    int minor = 0;

    [[nodiscard]] friend constexpr bool operator==(GLVersion,
                                                   GLVersion) noexcept = default;
    [[nodiscard]] friend constexpr auto operator<=>(GLVersion,
                                                    GLVersion) noexcept = default;
};

struct Extent {
    int width = 0;
    int height = 0;

    [[nodiscard]] friend constexpr bool operator==(Extent, Extent) noexcept = default;
};

/// Framebuffer pixels per window coordinate. Not 1 on a HiDPI display.
struct ContentScale {
    float x = 1.0f;
    float y = 1.0f;
};

enum class CursorMode {
    Normal,
    /// Visible cursor hidden over the window, still bounded by the screen.
    Hidden,
    /// Hidden and unbounded, reporting virtual motion. This is what a
    /// mouse-look camera wants.
    Disabled,
};

enum class WindowErrorCode {
    PlatformNotInitialized,
    /// Rejected before the windowing system was asked. See WindowSettings.
    InvalidSettings,
    /// The windowing system refused, most often because the driver cannot
    /// provide the requested context.
    CreationFailed,
    /// The context exists but no OpenGL entry points could be resolved.
    LoaderFailed,
    /// The context reports a version below the one asked for. Drivers are
    /// allowed to hand back more than was requested, never less.
    VersionBelowRequest,
};

struct WindowError {
    WindowErrorCode code = WindowErrorCode::CreationFailed;
    std::string message;
    /// Set for VersionBelowRequest, so a caller stepping down its request knows
    /// what it can actually have.
    GLVersion loaded{};
};

/// The highest OpenGL version this build can resolve entry points for, which is
/// what third_party/glad was generated against. Requesting more cannot work, so
/// Window rejects it rather than letting a driver report success and leaving
/// half the loader null.
///
/// Window.cpp checks this against what GLAD actually defines, so regenerating
/// the loader at a different version stops the build instead of quietly making
/// this constant a lie.
inline constexpr GLVersion kMaxSupportedGL{4, 6};

struct ContextSettings {
    /// 4.1 is the default because it is the highest macOS provides, and it is
    /// enough for the visualiser. Compute's OpenGL backend asks for 4.3, which
    /// is where compute shaders start, and gets a clean failure where that is
    /// not available.
    int versionMajor = 4;
    int versionMinor = 1;
    /// Core profile requires 3.2 or higher, forward compatibility 3.0. macOS
    /// gives nothing above 2.1 unless both are set.
    bool coreProfile = true;
    bool forwardCompatible = true;
    bool debugContext = false;
    int depthBits = 24;
    int stencilBits = 8;
    /// Multisample samples per pixel. Zero disables it.
    int samples = 0;
};

/// The settings that will actually be requested of `backend`.
///
/// OSMesa, which is where the Null backend's contexts come from, refuses
/// forward compatibility outright, and forward compatibility is on by default
/// because macOS gives nothing above 2.1 without it. Dropping it there is the
/// difference between a headless context and none at all, and it is safe in the
/// direction that matters: forward compatibility only removes deprecated
/// functionality, so a context without it can do strictly more, never less.
///
/// Public and pure so a caller can see what a backend will do to its request
/// before creating anything, and so the adjustment is testable with no context.
[[nodiscard]] ContextSettings effectiveContext(const ContextSettings& requested,
                                               PlatformBackend backend) noexcept;

struct WindowSettings {
    std::string title = "YSQ";
    int width = 1280;
    int height = 720;
    bool visible = true;
    bool resizable = true;
    bool decorated = true;
    /// Cap presentation at the display refresh rate.
    bool vsync = true;
    ContextSettings context{};
};

/// A window and its OpenGL context, or with visible off, a context on its own.
///
///     const auto platform = Platform::initialize();
///     auto window = Window::create({.title = "YSQ"});
///     while (!window->shouldClose()) {
///         window->input().newFrame();
///         Platform::pollEvents();
///         // draw
///         window->swapBuffers();
///     }
///
/// Event delivery is global and input state is per window, so a program with
/// more than one window calls newFrame() on every one of them before the single
/// pollEvents(). Skipping one leaves its edges reading a frame old.
///
/// An offscreen context is a window with visible off, so there is one code path
/// rather than two, and createOffscreen() is a name for the settings that make
/// it. On the Null backend every window is offscreen regardless.
///
/// **One loader, one context at a time.** GLAD's entry points are process-wide
/// globals, so they describe whichever context was made current last. Two
/// contexts from different drivers cannot both be live. makeContextCurrent()
/// reloads them when the context changes, which costs a few hundred symbol
/// lookups, so switch between contexts at setup time, not per frame.
///
/// Holds a Platform handle, so the windowing system cannot be shut down while a
/// window is alive.
class Window {
public:
    [[nodiscard]] static std::optional<Window> create(const WindowSettings& settings,
                                                      WindowError* error = nullptr);

    /// A context with no visible window, for offscreen rendering and for
    /// compute on machines with no display.
    [[nodiscard]] static std::optional<Window>
    createOffscreen(int width, int height, const ContextSettings& context = {},
                    WindowError* error = nullptr);

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;
    ~Window();

    /// Makes this window's context current and, if it changed, reloads the GL
    /// entry points to match.
    void makeContextCurrent();

    /// Detaches whatever context is current, on any window.
    static void clearCurrentContext();

    [[nodiscard]] bool isContextCurrent() const;

    void swapBuffers();

    /// Makes the context current, since the swap interval belongs to it.
    void setVSync(bool enabled);
    [[nodiscard]] bool vsync() const noexcept { return m_vsync; }

    [[nodiscard]] bool shouldClose() const;
    void setShouldClose(bool close);

    void setTitle(std::string_view title);
    void setCursorMode(CursorMode mode);
    [[nodiscard]] CursorMode cursorMode() const noexcept { return m_cursorMode; }

    void show();
    void hide();
    [[nodiscard]] bool visible() const;

    /// In window coordinates, which are not pixels on a HiDPI display.
    [[nodiscard]] Extent size() const;
    /// In pixels. This is what glViewport wants.
    [[nodiscard]] Extent framebufferSize() const;
    [[nodiscard]] ContentScale contentScale() const;

    /// What the context reports, which may exceed what was requested.
    [[nodiscard]] GLVersion glVersion() const noexcept { return m_glVersion; }

    /// What was asked of the driver, after the backend's own constraints were
    /// applied. Not necessarily what the caller passed in: see
    /// effectiveContext(). Whether the profile is forward compatible decides
    /// whether deprecated OpenGL is callable, so it is worth being able to ask.
    [[nodiscard]] const ContextSettings& contextSettings() const noexcept {
        return m_context;
    }

    [[nodiscard]] InputState& input() noexcept { return m_input; }
    [[nodiscard]] const InputState& input() const noexcept { return m_input; }

    /// Fired during event polling, in pixels. Set before the first poll.
    void setFramebufferSizeCallback(std::function<void(Extent)> callback);
    /// Fired when the window manager asks the window to close. The close flag
    /// is already set by then; clear it in the handler to refuse.
    void setCloseCallback(std::function<void()> callback);

    [[nodiscard]] GLFWwindow* nativeHandle() const noexcept { return m_handle; }

private:
    friend struct detail::WindowCallbacks;

    Window(Platform platform, GLFWwindow* handle, GLVersion version,
           ContextSettings context) noexcept;

    /// Points the window's user pointer back at this object. Called on every
    /// move, because the callbacks reach the InputState through it and a moved
    /// window would otherwise feed the shell it left behind.
    void bindCallbacks() noexcept;

    void destroy() noexcept;

    Platform m_platform;
    GLFWwindow* m_handle = nullptr;
    GLVersion m_glVersion{};
    ContextSettings m_context{};
    InputState m_input;
    CursorMode m_cursorMode = CursorMode::Normal;
    bool m_vsync = false;
    std::function<void(Extent)> m_onFramebufferSize;
    std::function<void()> m_onClose;
};

}  // namespace ysq
