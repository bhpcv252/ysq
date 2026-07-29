#include <Platform/Platform.hpp>

#include <Core/Logger.hpp>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>

#include <cmath>
#include <utility>

namespace ysq {

namespace {

/// Live Platform handles. Single-threaded by contract; on macOS the windowing
/// system requires the main thread anyway.
int g_handles = 0;

int nativeFromBackend(PlatformBackend backend) noexcept {
    switch (backend) {
        case PlatformBackend::Win32:
            return GLFW_PLATFORM_WIN32;
        case PlatformBackend::Cocoa:
            return GLFW_PLATFORM_COCOA;
        case PlatformBackend::Wayland:
            return GLFW_PLATFORM_WAYLAND;
        case PlatformBackend::X11:
            return GLFW_PLATFORM_X11;
        case PlatformBackend::Null:
            break;
    }
    return GLFW_PLATFORM_NULL;
}

std::optional<PlatformBackend> backendFromNative(int nativeId) noexcept {
    switch (nativeId) {
        case GLFW_PLATFORM_WIN32:
            return PlatformBackend::Win32;
        case GLFW_PLATFORM_COCOA:
            return PlatformBackend::Cocoa;
        case GLFW_PLATFORM_WAYLAND:
            return PlatformBackend::Wayland;
        case GLFW_PLATFORM_X11:
            return PlatformBackend::X11;
        case GLFW_PLATFORM_NULL:
            return PlatformBackend::Null;
        default:
            return std::nullopt;
    }
}

/// Debug, not error.
///
/// This module is built around probing: try the native backend then Null, try
/// one OpenGL version then a lower one. Most failures that reach here are
/// expected steps in a sequence that goes on to succeed, and logging them as
/// errors means a healthy run fills the log with errors, which teaches people to
/// ignore the level. Every one of them is also returned to the caller as a
/// PlatformError or WindowError, so the code that knows whether a failure was
/// fatal is the code that decides how loudly to say so.
void errorCallback(int code, const char* description) {
    logging::debug("GLFW error {}: {}", code,
                   description ? description : "no description");
}

/// Takes GLFW's pending error, which also clears it, so the next failure is not
/// reported with a stale description.
void takeError(PlatformError* out, int fallbackCode, std::string_view fallbackMessage) {
    const char* description = nullptr;
    const int code = glfwGetError(&description);
    if (!out) {
        return;
    }
    if (code == GLFW_NO_ERROR) {
        out->code = fallbackCode;
        out->message = fallbackMessage;
        return;
    }
    out->code = code;
    out->message = description ? description : std::string{fallbackMessage};
}

}  // namespace

std::optional<Platform> Platform::initialize(const PlatformSettings& settings,
                                             PlatformError* error) {
    if (g_handles > 0) {
        const std::optional<PlatformBackend> active = backend();
        if (settings.backend && active && *settings.backend != *active) {
            if (error) {
                error->code = GLFW_INVALID_ENUM;
                error->message = std::string{"already initialised with the "} +
                                 std::string{toString(*active)} + " backend; " +
                                 std::string{toString(*settings.backend)} +
                                 " needs every handle released first";
            }
            return std::nullopt;
        }
        return current();
    }

    // Legal before initialisation, and the only way to see why initialisation
    // itself failed.
    glfwSetErrorCallback(&errorCallback);

    if (settings.backend) {
        if (!backendAvailable(*settings.backend)) {
            if (error) {
                error->code = GLFW_PLATFORM_UNAVAILABLE;
                error->message = std::string{"the "} +
                                 std::string{toString(*settings.backend)} +
                                 " backend was not compiled in";
            }
            return std::nullopt;
        }
        glfwInitHint(GLFW_PLATFORM, nativeFromBackend(*settings.backend));
    } else {
        glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
    }

    if (glfwInit() != GLFW_TRUE) {
        takeError(error, GLFW_PLATFORM_UNAVAILABLE, "no windowing system available");
        return std::nullopt;
    }

    Platform handle;
    handle.m_owns = true;
    g_handles = 1;

    if (const std::optional<PlatformBackend> active = backend()) {
        logging::debug("Platform initialised on the {} backend", toString(*active));
    }

    return std::optional<Platform>{std::move(handle)};
}

std::optional<Platform> Platform::current() noexcept {
    if (g_handles == 0) {
        return std::nullopt;
    }
    Platform handle;
    handle.m_owns = true;
    ++g_handles;
    return std::optional<Platform>{std::move(handle)};
}

Platform::Platform(const Platform& other) noexcept : m_owns(other.m_owns) {
    if (m_owns) {
        ++g_handles;
    }
}

Platform& Platform::operator=(const Platform& other) noexcept {
    if (this == &other) {
        return *this;
    }
    // Retain before releasing, so assigning the last handle to itself through
    // two objects cannot shut the windowing system down mid-assignment.
    if (other.m_owns) {
        ++g_handles;
    }
    if (m_owns && --g_handles == 0) {
        glfwTerminate();
    }
    m_owns = other.m_owns;
    return *this;
}

Platform::Platform(Platform&& other) noexcept
    : m_owns(std::exchange(other.m_owns, false)) {}

Platform& Platform::operator=(Platform&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (m_owns && --g_handles == 0) {
        glfwTerminate();
    }
    m_owns = std::exchange(other.m_owns, false);
    return *this;
}

Platform::~Platform() {
    if (m_owns && --g_handles == 0) {
        glfwTerminate();
    }
}

bool Platform::initialized() noexcept {
    return g_handles > 0;
}

std::optional<PlatformBackend> Platform::backend() noexcept {
    if (!initialized()) {
        return std::nullopt;
    }
    return backendFromNative(glfwGetPlatform());
}

bool Platform::backendAvailable(PlatformBackend backend) noexcept {
    return glfwPlatformSupported(nativeFromBackend(backend)) == GLFW_TRUE;
}

void Platform::pollEvents() noexcept {
    if (!initialized()) {
        return;
    }
    glfwPollEvents();
}

void Platform::waitEvents() noexcept {
    if (!initialized()) {
        return;
    }
    glfwWaitEvents();
}

void Platform::waitEventsFor(double timeoutSeconds) noexcept {
    if (!initialized()) {
        return;
    }
    if (!(timeoutSeconds > 0.0) || !std::isfinite(timeoutSeconds)) {
        // Not glfwWaitEvents(): a nonsense timeout turning into an indefinite
        // block is the kind of hang that gets blamed on everything but this.
        glfwPollEvents();
        return;
    }
    glfwWaitEventsTimeout(timeoutSeconds);
}

std::string_view toString(PlatformBackend backend) noexcept {
    switch (backend) {
        case PlatformBackend::Win32:
            return "Win32";
        case PlatformBackend::Cocoa:
            return "Cocoa";
        case PlatformBackend::Wayland:
            return "Wayland";
        case PlatformBackend::X11:
            return "X11";
        case PlatformBackend::Null:
            break;
    }
    return "Null";
}

}  // namespace ysq
