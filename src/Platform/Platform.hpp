#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ysq {

/// The windowing system underneath. Null is the headless one: it needs no
/// display server and gets its OpenGL contexts from OSMesa, a software
/// rasteriser loaded at run time. That is how the offscreen path is tested on
/// machines and CI runners with no display at all.
enum class PlatformBackend { Win32, Cocoa, Wayland, X11, Null };

struct PlatformError {
    /// The backend's own error code, kept so a caller can distinguish "no
    /// display" from "no such backend compiled in" without parsing text.
    int code = 0;
    std::string message;
};

struct PlatformSettings {
    /// Unset auto-selects: the native backend, and on Unix the session's own.
    /// Null is never auto-selected and must be asked for by name.
    std::optional<PlatformBackend> backend{};
};

/// Reference-counted handle on the windowing system.
///
/// The first live handle initialises it, the last one to die shuts it down.
/// Copying retains, which is what lets a Window hold one and so makes it
/// impossible to shut the windowing system down underneath a live window.
///
///     const auto platform = Platform::initialize();
///     if (!platform) { /* headless machine, or no display */ }
///
/// Initialising twice with different backends fails rather than silently
/// keeping the first: the backend cannot be changed without a full shutdown,
/// and a test that asked for Null must not quietly get X11.
///
/// Not thread-safe, and on macOS the windowing system additionally requires
/// that all of this happens on the main thread.
class Platform {
public:
    [[nodiscard]] static std::optional<Platform> initialize(
        const PlatformSettings& settings = {}, PlatformError* error = nullptr);

    /// A handle on the already-initialised windowing system, or nullopt. Retains
    /// like a copy; it does not initialise.
    [[nodiscard]] static std::optional<Platform> current() noexcept;

    Platform(const Platform& other) noexcept;
    Platform& operator=(const Platform& other) noexcept;
    Platform(Platform&& other) noexcept;
    Platform& operator=(Platform&& other) noexcept;
    ~Platform();

    [[nodiscard]] static bool initialized() noexcept;

    /// Nullopt when not initialised.
    [[nodiscard]] static std::optional<PlatformBackend> backend() noexcept;

    /// Whether support for a backend was compiled in. Safe before initialise,
    /// and false is a build-time fact, not a "no display right now".
    [[nodiscard]] static bool backendAvailable(PlatformBackend backend) noexcept;

    /// Delivers every event waiting, then returns. No-op when not initialised.
    static void pollEvents() noexcept;

    /// Sleeps until an event arrives. This is for an editor or viewer that
    /// should idle at zero cost; a simulation loop wants pollEvents().
    static void waitEvents() noexcept;

    /// Sleeps until an event arrives or the timeout expires, whichever is first.
    /// A non-positive or non-finite timeout returns immediately rather than
    /// waiting forever, since a caller that named a timeout did not ask to
    /// block indefinitely.
    static void waitEventsFor(double timeoutSeconds) noexcept;

private:
    Platform() noexcept = default;

    /// False in a moved-from handle, which then releases nothing.
    bool m_owns = false;
};

[[nodiscard]] std::string_view toString(PlatformBackend backend) noexcept;

}  // namespace ysq
