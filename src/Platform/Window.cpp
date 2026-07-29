#include <Platform/Window.hpp>

#include <Core/Logger.hpp>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <format>
#include <utility>

// kMaxSupportedGL claims a version on the loader's behalf, and nothing else ties
// the two together: regenerate GLAD lower and the claim silently over-promises,
// with null entry points at run time as the payoff. GLAD defines a
// GL_VERSION_x_y for the version it was generated for, so pin the two here.
#if !defined(GL_VERSION_4_6)
#error "GLAD is not generated for OpenGL 4.6; update kMaxSupportedGL to match it"
#endif

namespace ysq {

static_assert(kMaxSupportedGL == GLVersion{4, 6},
              "kMaxSupportedGL and the GL_VERSION check above must name the same "
              "version, or the check is guarding nothing");

namespace {

/// The context GLAD's process-wide entry points currently describe. Null when
/// nothing has been loaded, or when the context they came from is gone.
GLFWwindow* g_loaderContext = nullptr;

/// Makes `context` current and points the loader at it, reloading only when it
/// is not already the one loaded. Null detaches without touching the loader,
/// whose entry points stay valid for the context they came from.
void bindContext(GLFWwindow* context) {
    glfwMakeContextCurrent(context);
    if (!context || g_loaderContext == context) {
        return;
    }
    g_loaderContext = gladLoadGL(glfwGetProcAddress) != 0 ? context : nullptr;
}

void setError(WindowError* error, WindowErrorCode code, std::string message,
              GLVersion loaded = {}) {
    if (error) {
        error->code = code;
        error->message = std::move(message);
        error->loaded = loaded;
    }
}

/// Takes the windowing system's pending error, which also clears it.
std::string takeErrorMessage(std::string_view fallback) {
    const char* description = nullptr;
    if (glfwGetError(&description) != GLFW_NO_ERROR && description) {
        return description;
    }
    return std::string{fallback};
}

/// The reason the settings are unusable, or nothing. Checked before the
/// windowing system is involved, so these failures are the same everywhere and
/// need no display to provoke.
std::optional<std::string> validate(const WindowSettings& settings) {
    if (settings.width <= 0 || settings.height <= 0) {
        return std::format("window size must be positive, got {}x{}", settings.width,
                           settings.height);
    }

    const ContextSettings& context = settings.context;
    const GLVersion requested{context.versionMajor, context.versionMinor};

    if (requested.major < 1 || requested.minor < 0) {
        return std::format("OpenGL {}.{} is not a version", requested.major,
                           requested.minor);
    }
    if (requested > kMaxSupportedGL) {
        return std::format(
            "OpenGL {}.{} was requested but the loader is generated for {}.{}, so its "
            "entry points could not be resolved",
            requested.major, requested.minor, kMaxSupportedGL.major,
            kMaxSupportedGL.minor);
    }
    if (context.coreProfile && requested < GLVersion{3, 2}) {
        return std::format("the core profile requires OpenGL 3.2 or higher, got {}.{}",
                           requested.major, requested.minor);
    }
    if (context.forwardCompatible && requested < GLVersion{3, 0}) {
        return std::format("forward compatibility requires OpenGL 3.0 or higher, got "
                           "{}.{}",
                           requested.major, requested.minor);
    }
    if (context.depthBits < 0 || context.stencilBits < 0 || context.samples < 0) {
        return std::string{"depth, stencil and sample counts cannot be negative"};
    }
    return std::nullopt;
}

void applyHints(const WindowSettings& settings, const ContextSettings& context) {
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, context.versionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, context.versionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, context.coreProfile ? GLFW_OPENGL_CORE_PROFILE
                                                            : GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                   context.forwardCompatible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,
                   context.debugContext ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_DEPTH_BITS, context.depthBits);
    glfwWindowHint(GLFW_STENCIL_BITS, context.stencilBits);
    glfwWindowHint(GLFW_SAMPLES, context.samples);

    glfwWindowHint(GLFW_VISIBLE, settings.visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, settings.resizable ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, settings.decorated ? GLFW_TRUE : GLFW_FALSE);
}

int nativeFromCursorMode(CursorMode mode) noexcept {
    switch (mode) {
        case CursorMode::Hidden:
            return GLFW_CURSOR_HIDDEN;
        case CursorMode::Disabled:
            return GLFW_CURSOR_DISABLED;
        case CursorMode::Normal:
            break;
    }
    return GLFW_CURSOR_NORMAL;
}

}  // namespace

ContextSettings effectiveContext(const ContextSettings& requested,
                                 PlatformBackend backend) noexcept {
    ContextSettings context = requested;
    if (context.forwardCompatible && backend == PlatformBackend::Null) {
        context.forwardCompatible = false;
    }
    return context;
}

namespace detail {

struct WindowCallbacks {
    [[nodiscard]] static Window* owner(GLFWwindow* handle) noexcept {
        return static_cast<Window*>(glfwGetWindowUserPointer(handle));
    }

    static void key(GLFWwindow* handle, int key, int /*scancode*/, int action, int mods) {
        if (Window* window = owner(handle)) {
            window->m_input.onKey(keyFromNative(key), buttonActionFromNative(action),
                                  modifiersFromNative(mods));
        }
    }

    static void mouseButton(GLFWwindow* handle, int button, int action, int mods) {
        if (Window* window = owner(handle)) {
            window->m_input.onMouseButton(mouseButtonFromNative(button),
                                          buttonActionFromNative(action),
                                          modifiersFromNative(mods));
        }
    }

    static void cursorPosition(GLFWwindow* handle, double x, double y) {
        if (Window* window = owner(handle)) {
            window->m_input.onCursorPosition(x, y);
        }
    }

    static void scroll(GLFWwindow* handle, double x, double y) {
        if (Window* window = owner(handle)) {
            window->m_input.onScroll(x, y);
        }
    }

    static void focus(GLFWwindow* handle, int focused) {
        if (Window* window = owner(handle); window && focused == GLFW_FALSE) {
            window->m_input.onFocusLost();
        }
    }

    static void framebufferSize(GLFWwindow* handle, int width, int height) {
        Window* window = owner(handle);
        if (window && window->m_onFramebufferSize) {
            window->m_onFramebufferSize(Extent{width, height});
        }
    }

    static void close(GLFWwindow* handle) {
        Window* window = owner(handle);
        if (window && window->m_onClose) {
            window->m_onClose();
        }
    }
};

}  // namespace detail

std::optional<Window> Window::create(const WindowSettings& settings, WindowError* error) {
    std::optional<Platform> platform = Platform::current();
    if (!platform) {
        setError(error, WindowErrorCode::PlatformNotInitialized,
                 "Platform::initialize() must succeed before a window can be created");
        return std::nullopt;
    }

    if (const std::optional<std::string> problem = validate(settings)) {
        setError(error, WindowErrorCode::InvalidSettings, *problem);
        return std::nullopt;
    }

    const PlatformBackend backend = Platform::backend().value_or(PlatformBackend::Null);
    const ContextSettings context = effectiveContext(settings.context, backend);
    if (context.forwardCompatible != settings.context.forwardCompatible) {
        logging::debug("Forward compatibility dropped: the {} backend refuses it",
                       toString(backend));
    }
    applyHints(settings, context);

    // A failed creation must not cost the caller the context it already had.
    GLFWwindow* const previous = glfwGetCurrentContext();

    GLFWwindow* handle = glfwCreateWindow(settings.width, settings.height,
                                          settings.title.c_str(), nullptr, nullptr);
    if (!handle) {
        setError(error, WindowErrorCode::CreationFailed,
                 takeErrorMessage("the windowing system refused to create the window"));
        return std::nullopt;
    }

    glfwMakeContextCurrent(handle);

    // Loaded unconditionally: another context's entry points are not this one's,
    // and after a shutdown they point at nothing at all.
    const int loaded = gladLoadGL(glfwGetProcAddress);
    g_loaderContext = loaded != 0 ? handle : nullptr;

    const auto abandon = [&](WindowErrorCode code, std::string message,
                             GLVersion reported) {
        glfwDestroyWindow(handle);
        if (g_loaderContext == handle) {
            g_loaderContext = nullptr;
        }
        bindContext(previous);
        setError(error, code, std::move(message), reported);
    };

    if (loaded == 0) {
        abandon(WindowErrorCode::LoaderFailed,
                takeErrorMessage("no OpenGL entry points could be resolved"), {});
        return std::nullopt;
    }

    const GLVersion version{GLAD_VERSION_MAJOR(loaded), GLAD_VERSION_MINOR(loaded)};
    const GLVersion requested{settings.context.versionMajor,
                              settings.context.versionMinor};
    if (version < requested) {
        abandon(WindowErrorCode::VersionBelowRequest,
                std::format("OpenGL {}.{} was requested but the context reports {}.{}",
                            requested.major, requested.minor, version.major,
                            version.minor),
                version);
        return std::nullopt;
    }

    Window window{std::move(*platform), handle, version, context};
    window.bindCallbacks();
    window.setVSync(settings.vsync);

    logging::debug("Window \"{}\" {}x{} on OpenGL {}.{}", settings.title, settings.width,
                   settings.height, version.major, version.minor);

    return std::optional<Window>{std::move(window)};
}

std::optional<Window> Window::createOffscreen(int width, int height,
                                              const ContextSettings& context,
                                              WindowError* error) {
    WindowSettings settings;
    settings.title = "YSQ (offscreen)";
    settings.width = width;
    settings.height = height;
    settings.visible = false;
    settings.resizable = false;
    settings.decorated = false;
    // Nothing is presented, so pacing to a display would only add latency.
    settings.vsync = false;
    settings.context = context;
    return create(settings, error);
}

Window::Window(Platform platform, GLFWwindow* handle, GLVersion version,
               ContextSettings context) noexcept
    : m_platform(std::move(platform)),
      m_handle(handle),
      m_glVersion(version),
      m_context(context) {}

Window::Window(Window&& other) noexcept
    : m_platform(std::move(other.m_platform)),
      m_handle(std::exchange(other.m_handle, nullptr)),
      m_glVersion(other.m_glVersion),
      m_context(other.m_context),
      m_input(other.m_input),
      m_cursorMode(other.m_cursorMode),
      m_vsync(other.m_vsync),
      m_onFramebufferSize(std::move(other.m_onFramebufferSize)),
      m_onClose(std::move(other.m_onClose)) {
    bindCallbacks();
}

Window& Window::operator=(Window&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    destroy();

    m_platform = std::move(other.m_platform);
    m_handle = std::exchange(other.m_handle, nullptr);
    m_glVersion = other.m_glVersion;
    m_context = other.m_context;
    m_input = other.m_input;
    m_cursorMode = other.m_cursorMode;
    m_vsync = other.m_vsync;
    m_onFramebufferSize = std::move(other.m_onFramebufferSize);
    m_onClose = std::move(other.m_onClose);
    bindCallbacks();
    return *this;
}

Window::~Window() {
    destroy();
}

void Window::bindCallbacks() noexcept {
    if (!m_handle) {
        return;
    }
    // Re-pointed on every move: the callbacks find their InputState through
    // this, so a moved window would otherwise keep filling the husk it left.
    glfwSetWindowUserPointer(m_handle, this);
    glfwSetKeyCallback(m_handle, &detail::WindowCallbacks::key);
    glfwSetMouseButtonCallback(m_handle, &detail::WindowCallbacks::mouseButton);
    glfwSetCursorPosCallback(m_handle, &detail::WindowCallbacks::cursorPosition);
    glfwSetScrollCallback(m_handle, &detail::WindowCallbacks::scroll);
    glfwSetWindowFocusCallback(m_handle, &detail::WindowCallbacks::focus);
    glfwSetFramebufferSizeCallback(m_handle, &detail::WindowCallbacks::framebufferSize);
    glfwSetWindowCloseCallback(m_handle, &detail::WindowCallbacks::close);
}

void Window::destroy() noexcept {
    if (!m_handle) {
        return;
    }
    if (g_loaderContext == m_handle) {
        // The entry points belong to a context that is about to stop existing.
        g_loaderContext = nullptr;
    }
    if (glfwGetCurrentContext() == m_handle) {
        glfwMakeContextCurrent(nullptr);
    }
    glfwDestroyWindow(m_handle);
    m_handle = nullptr;
}

void Window::makeContextCurrent() {
    if (!m_handle) {
        return;
    }
    bindContext(m_handle);
    if (g_loaderContext != m_handle) {
        logging::error("Could not resolve OpenGL entry points after a context switch");
    }
}

void Window::clearCurrentContext() {
    glfwMakeContextCurrent(nullptr);
}

bool Window::isContextCurrent() const {
    return m_handle && glfwGetCurrentContext() == m_handle;
}

void Window::swapBuffers() {
    if (m_handle) {
        glfwSwapBuffers(m_handle);
    }
}

void Window::setVSync(bool enabled) {
    if (!m_handle) {
        return;
    }
    // The swap interval is a property of the current context, not of a window.
    makeContextCurrent();
    glfwSwapInterval(enabled ? 1 : 0);
    m_vsync = enabled;
}

bool Window::shouldClose() const {
    return m_handle && glfwWindowShouldClose(m_handle) == GLFW_TRUE;
}

void Window::setShouldClose(bool close) {
    if (m_handle) {
        glfwSetWindowShouldClose(m_handle, close ? GLFW_TRUE : GLFW_FALSE);
    }
}

void Window::setTitle(std::string_view title) {
    if (m_handle) {
        glfwSetWindowTitle(m_handle, std::string{title}.c_str());
    }
}

void Window::setCursorMode(CursorMode mode) {
    if (!m_handle) {
        return;
    }
    glfwSetInputMode(m_handle, GLFW_CURSOR, nativeFromCursorMode(mode));
    m_cursorMode = mode;
}

void Window::show() {
    if (m_handle) {
        glfwShowWindow(m_handle);
    }
}

void Window::hide() {
    if (m_handle) {
        glfwHideWindow(m_handle);
    }
}

bool Window::visible() const {
    return m_handle && glfwGetWindowAttrib(m_handle, GLFW_VISIBLE) == GLFW_TRUE;
}

Extent Window::size() const {
    Extent extent;
    if (m_handle) {
        glfwGetWindowSize(m_handle, &extent.width, &extent.height);
    }
    return extent;
}

Extent Window::framebufferSize() const {
    Extent extent;
    if (m_handle) {
        glfwGetFramebufferSize(m_handle, &extent.width, &extent.height);
    }
    return extent;
}

ContentScale Window::contentScale() const {
    ContentScale scale;
    if (m_handle) {
        glfwGetWindowContentScale(m_handle, &scale.x, &scale.y);
    }
    return scale;
}

void Window::setFramebufferSizeCallback(std::function<void(Extent)> callback) {
    m_onFramebufferSize = std::move(callback);
}

void Window::setCloseCallback(std::function<void()> callback) {
    m_onClose = std::move(callback);
}

}  // namespace ysq
