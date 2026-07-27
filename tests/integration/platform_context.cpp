#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>

#include <gtest/gtest.h>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <array>
#include <optional>
#include <string>
#include <utility>

// A real OpenGL context, opened and used.
//
// Whether one can exist at all is a property of the machine, not of the code. A
// CI runner has no display, so the native backend cannot start; there the Null
// backend and OSMesa are the only route to a context, and OSMesa is dlopened at
// run time and usually absent. So every test here tries the native backend,
// falls back to Null, and skips if neither yields a context.
//
// Configuring with -DYSQ_REQUIRE_HEADLESS_GL=ON turns those skips into failures.
// CI sets it on the one job where OSMesa is installed, which is what stops the
// whole file from skipping everywhere and quietly testing nothing.

namespace {

using ysq::ContextSettings;
using ysq::GLVersion;
using ysq::Platform;
using ysq::PlatformBackend;
using ysq::PlatformError;
using ysq::PlatformSettings;
using ysq::Window;
using ysq::WindowError;
using ysq::WindowErrorCode;

// A compile definition rather than an environment variable: MSVC deprecates
// std::getenv, the build treats warnings as errors, and the answer cannot change
// during a run anyway. See tests/integration/CMakeLists.txt.
#if defined(YSQ_REQUIRE_HEADLESS_GL)
constexpr bool kContextIsRequired = true;
#else
constexpr bool kContextIsRequired = false;
#endif

// GTEST_SKIP and FAIL both have to appear in the test body itself, so this is a
// macro rather than the helper function it would otherwise be.
#define SKIP_UNLESS_REQUIRED(reason)                                              \
    do {                                                                          \
        if (kContextIsRequired) {                                                 \
            FAIL() << "built with YSQ_REQUIRE_HEADLESS_GL, so this must not skip: " \
                   << (reason);                                                   \
        }                                                                         \
        GTEST_SKIP() << (reason);                                                 \
    } while (false)

/// 4.1 is the default request and the most macOS offers. OSMesa builds vary and
/// some stop lower, so an older context beats no context: this is the ladder a
/// test walks down before giving up.
constexpr std::array<GLVersion, 2> kVersionLadder{GLVersion{4, 1}, GLVersion{3, 3}};

ContextSettings contextFor(GLVersion version) {
    ContextSettings context;
    context.versionMajor = version.major;
    context.versionMinor = version.minor;
    return context;
}

/// A live context, however this machine can provide one.
struct Session {
    std::optional<Platform> platform;
    std::optional<Window> window;
    /// The version the window was asked for, which it is guaranteed to meet.
    GLVersion requested{};
    /// Why there is no window, when there is none.
    std::string failure;

    [[nodiscard]] bool opened() const { return window.has_value(); }
};

/// The native backend first, then Null. A machine with a display exercises the
/// path an application will actually take; a runner without one still exercises
/// everything but the display server itself.
Session openContext(int width = 64, int height = 64) {
    const std::array<std::optional<PlatformBackend>, 2> backends{
        std::nullopt, PlatformBackend::Null};

    Session session;
    for (const std::optional<PlatformBackend>& backend : backends) {
        PlatformSettings settings;
        settings.backend = backend;

        PlatformError platformError;
        // Released at the end of the iteration if nothing comes of it, because
        // the backend cannot be changed while any handle is alive.
        session.platform = Platform::initialize(settings, &platformError);
        if (!session.platform) {
            session.failure = platformError.message;
            continue;
        }

        for (const GLVersion version : kVersionLadder) {
            WindowError windowError;
            session.window = Window::createOffscreen(width, height,
                                                     contextFor(version), &windowError);
            if (session.window) {
                session.requested = version;
                return session;
            }
            session.failure = windowError.message;
        }
        session.platform.reset();
    }
    return session;
}

}  // namespace

TEST(PlatformContext, AContextOpensAndAcceptsCommands) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }
    Window& window = *session.window;

    EXPECT_TRUE(window.isContextCurrent());
    EXPECT_GE(window.glVersion(), session.requested)
        << "a driver may hand back more than was asked for, never less";

    // The entry points resolved, and calling one reaches a live context.
    const GLubyte* reported = glGetString(GL_VERSION);
    ASSERT_NE(reported, nullptr);
    EXPECT_FALSE(std::string{reinterpret_cast<const char*>(reported)}.empty());

    EXPECT_EQ(window.size(), (ysq::Extent{64, 64}));
    // Not compared to the window size: on a HiDPI display the framebuffer is
    // larger, which is exactly why the two are separate calls.
    EXPECT_GE(window.framebufferSize().width, 64);
    EXPECT_GE(window.framebufferSize().height, 64);
    EXPECT_FALSE(window.visible()) << "an offscreen window must not appear";

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_EQ(glGetError(), static_cast<GLenum>(GL_NO_ERROR));

    window.swapBuffers();
    Platform::pollEvents();
    EXPECT_FALSE(window.shouldClose());
}

TEST(PlatformContext, TheNullBackendGivesAContextWithNoDisplayAtAll) {
    ASSERT_TRUE(Platform::backendAvailable(PlatformBackend::Null))
        << "the headless backend must be compiled in everywhere";

    PlatformSettings settings;
    settings.backend = PlatformBackend::Null;

    PlatformError platformError;
    const std::optional<Platform> platform = Platform::initialize(settings, &platformError);
    ASSERT_TRUE(platform) << platformError.message;
    ASSERT_EQ(Platform::backend(), PlatformBackend::Null)
        << "Null must never be substituted for something else";

    WindowError windowError;
    const std::optional<Window> window =
        Window::createOffscreen(64, 64, contextFor(kVersionLadder.front()), &windowError);
    if (!window) {
        // The Null backend gets its context from OSMesa, which is dlopened at
        // run time and simply absent on most machines.
        SKIP_UNLESS_REQUIRED("no OSMesa: " + windowError.message);
    }

    EXPECT_TRUE(window->isContextCurrent());
    EXPECT_GE(window->glVersion(), kVersionLadder.front());
    ASSERT_NE(glGetString(GL_VERSION), nullptr);

    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_EQ(glGetError(), static_cast<GLenum>(GL_NO_ERROR));
}

TEST(PlatformContext, AWindowHoldsThePlatformOpenByItself) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    session.platform.reset();
    EXPECT_TRUE(Platform::initialized())
        << "the window's own handle must outlive the caller's";

    session.window.reset();
    EXPECT_FALSE(Platform::initialized());
}

TEST(PlatformContext, MovingAWindowTakesItsCallbacksWithIt) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    GLFWwindow* const handle = session.window->nativeHandle();
    ASSERT_EQ(glfwGetWindowUserPointer(handle), &*session.window);

    // The callbacks reach the input state through the user pointer, so a move
    // that forgets to re-point it feeds the husk instead. That shows up as input
    // silently dying, which is miserable to track down.
    Window moved = std::move(*session.window);
    EXPECT_EQ(moved.nativeHandle(), handle);
    EXPECT_EQ(glfwGetWindowUserPointer(handle), &moved);

    session.window.reset();
    EXPECT_EQ(glfwGetWindowUserPointer(handle), &moved)
        << "destroying the moved-from window must not touch the live one";
}

TEST(PlatformContext, EveryInputCallbackIsRegistered) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }
    GLFWwindow* const handle = session.window->nativeHandle();

    // Each setter returns the callback it replaced, so installing null and
    // looking at what comes back proves one was there. Dropping a single
    // glfwSet*Callback line in bindCallbacks() is otherwise invisible: that
    // input just silently stops arriving, and no other test would notice.
    //
    // This proves registration, not routing. Delivering a synthetic key event
    // would need the window system to send one, and no backend here can, so the
    // hop from the callback into InputState stays uncovered on purpose.
    EXPECT_NE(glfwSetKeyCallback(handle, nullptr), nullptr) << "key";
    EXPECT_NE(glfwSetMouseButtonCallback(handle, nullptr), nullptr) << "mouse button";
    EXPECT_NE(glfwSetCursorPosCallback(handle, nullptr), nullptr) << "cursor position";
    EXPECT_NE(glfwSetScrollCallback(handle, nullptr), nullptr) << "scroll";
    EXPECT_NE(glfwSetWindowFocusCallback(handle, nullptr), nullptr) << "focus";
    EXPECT_NE(glfwSetFramebufferSizeCallback(handle, nullptr), nullptr) << "framebuffer";
    EXPECT_NE(glfwSetWindowCloseCallback(handle, nullptr), nullptr) << "close";
}

TEST(PlatformContext, TheGrantedContextIsVisibleToTheCaller) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    const ysq::ContextSettings& granted = session.window->contextSettings();
    EXPECT_EQ(granted.versionMajor, session.requested.major);
    EXPECT_EQ(granted.versionMinor, session.requested.minor);
    EXPECT_TRUE(granted.coreProfile);

    // Whether deprecated OpenGL is callable is not something a caller should
    // have to infer from which backend it happens to be running on.
    const bool onNull = Platform::backend() == PlatformBackend::Null;
    EXPECT_EQ(granted.forwardCompatible, !onNull);
}

TEST(PlatformContext, VSyncAndTitleAreSettableOnALiveWindow) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }
    Window& window = *session.window;

    EXPECT_FALSE(window.vsync()) << "an offscreen window presents nothing to pace to";
    window.setVSync(true);
    EXPECT_TRUE(window.vsync());
    EXPECT_TRUE(window.isContextCurrent())
        << "the swap interval belongs to the context, so setting it makes it current";
    window.setVSync(false);
    EXPECT_FALSE(window.vsync());

    // No way to read a title back, so this asserts only that it survives the
    // call. Same for setCursorMode and contentScale below: they are exercised
    // rather than checked, because the Null backend has neither a cursor nor a
    // monitor to answer for them.
    window.setTitle("renamed");
    window.setCursorMode(ysq::CursorMode::Disabled);
    EXPECT_EQ(window.cursorMode(), ysq::CursorMode::Disabled);
    window.setCursorMode(ysq::CursorMode::Normal);

    const ysq::ContentScale scale = window.contentScale();
    EXPECT_GT(scale.x, 0.0f);
    EXPECT_GT(scale.y, 0.0f);
}

TEST(PlatformContext, AnOffscreenWindowCanBeShown) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }
    if (Platform::backend() != PlatformBackend::Null) {
        // Only on the backend that has nothing to display. Elsewhere this puts a
        // real window on someone's screen mid-test and takes their focus with
        // it, which is not a thing a test suite should do.
        GTEST_SKIP() << "would show a real window";
    }

    ASSERT_FALSE(session.window->visible());
    session.window->show();
    EXPECT_TRUE(session.window->visible());
    session.window->hide();
    EXPECT_FALSE(session.window->visible());
}

TEST(PlatformContext, SwitchingContextsReloadsTheEntryPoints) {
    Session session = openContext(32, 32);
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    WindowError error;
    std::optional<Window> second =
        Window::createOffscreen(64, 64, contextFor(session.requested), &error);
    ASSERT_TRUE(second) << error.message;

    // Creating the second made it current. GLAD's entry points are process-wide,
    // so they describe whichever context was made current last.
    EXPECT_TRUE(second->isContextCurrent());
    EXPECT_FALSE(session.window->isContextCurrent());
    ASSERT_NE(glGetString(GL_VERSION), nullptr);

    session.window->makeContextCurrent();
    EXPECT_TRUE(session.window->isContextCurrent());
    EXPECT_NE(glGetString(GL_VERSION), nullptr);
    EXPECT_EQ(glGetError(), static_cast<GLenum>(GL_NO_ERROR));

    Window::clearCurrentContext();
    EXPECT_FALSE(session.window->isContextCurrent());
    EXPECT_FALSE(second->isContextCurrent());
}

TEST(PlatformContext, DestroyingAWindowDetachesItsContext) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    ASSERT_TRUE(session.window->isContextCurrent());
    session.window.reset();
    EXPECT_EQ(glfwGetCurrentContext(), nullptr)
        << "a destroyed context must not stay current";
}

TEST(PlatformContext, TheCloseFlagIsSettableWithoutAWindowManager) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    EXPECT_FALSE(session.window->shouldClose());
    session.window->setShouldClose(true);
    EXPECT_TRUE(session.window->shouldClose()) << "this is what ends the frame loop";
    session.window->setShouldClose(false);
    EXPECT_FALSE(session.window->shouldClose());
}

TEST(PlatformContext, AVersionTheDriverCannotGiveFailsCleanly) {
    Session session = openContext();
    if (!session.opened()) {
        SKIP_UNLESS_REQUIRED("no OpenGL context: " + session.failure);
    }

    // 4.6 is within what the loader supports, so this request reaches the
    // driver. It succeeds on some machines and not others, and both are correct;
    // what must hold either way is that a refusal is clean and typed.
    WindowError error;
    const std::optional<Window> window =
        Window::createOffscreen(32, 32, contextFor(GLVersion{4, 6}), &error);
    if (window) {
        EXPECT_GE(window->glVersion(), (GLVersion{4, 6}));
        return;
    }

    EXPECT_TRUE(error.code == WindowErrorCode::CreationFailed ||
                error.code == WindowErrorCode::VersionBelowRequest)
        << "unexpected failure: " << error.message;
    EXPECT_FALSE(error.message.empty());
    EXPECT_TRUE(session.window->isContextCurrent())
        << "a failed creation must leave the context that was already current alone";
}
