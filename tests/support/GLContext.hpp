#pragma once

#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>

/// A live offscreen OpenGL context, obtained however this machine can
/// provide one. Every Renderer/UI integration test that needs a real
/// context shares this rather than repeating the fallback ladder
/// tests/integration/platform_context.cpp established for exactly this
/// purpose: the native backend first, then GLFW's headless Null backend over
/// OSMesa, skipping only if neither yields a context.
///
/// Configuring with -DYSQ_REQUIRE_HEADLESS_GL=ON turns that skip into a
/// failure; see tests/README.md.

namespace ysq::test {

#if defined(YSQ_REQUIRE_HEADLESS_GL)
inline constexpr bool kHeadlessGLIsRequired = true;
#else
inline constexpr bool kHeadlessGLIsRequired = false;
#endif

struct GLSession {
    std::optional<Platform> platform;
    std::optional<Window> window;
    /// Why there is no window, when there is none.
    std::string failure;

    [[nodiscard]] bool opened() const { return window.has_value(); }
};

/// The native backend first, then Null. A machine with a display exercises
/// the path an application will actually take; a runner without one still
/// exercises everything but the display server itself.
[[nodiscard]] inline GLSession openGLSession(int width = 64, int height = 64) {
    const std::array<std::optional<PlatformBackend>, 2> backends{std::nullopt,
                                                                 PlatformBackend::Null};
    // 4.1 is the default request and the most macOS offers. OSMesa builds
    // vary and some stop lower, so an older context beats no context.
    const std::array<GLVersion, 2> versionLadder{GLVersion{4, 1}, GLVersion{3, 3}};

    GLSession session;
    for (const std::optional<PlatformBackend>& backend : backends) {
        PlatformSettings settings;
        settings.backend = backend;

        PlatformError platformError;
        session.platform = Platform::initialize(settings, &platformError);
        if (!session.platform) {
            session.failure = platformError.message;
            continue;
        }

        for (const GLVersion version : versionLadder) {
            ContextSettings context;
            context.versionMajor = version.major;
            context.versionMinor = version.minor;

            WindowError windowError;
            session.window =
                Window::createOffscreen(width, height, context, &windowError);
            if (session.window) {
                return session;
            }
            session.failure = windowError.message;
        }
        session.platform.reset();
    }
    return session;
}

}  // namespace ysq::test

// GTEST_SKIP and FAIL both have to appear in the test body itself, so this is
// a macro rather than a helper function, the same constraint
// tests/integration/platform_context.cpp documents for its own copy.
#define YSQ_SKIP_UNLESS_HEADLESS_GL_REQUIRED(reason)                                     \
    do {                                                                                 \
        if (::ysq::test::kHeadlessGLIsRequired) {                                        \
            FAIL() << "built with YSQ_REQUIRE_HEADLESS_GL, so this must not skip: "      \
                   << (reason);                                                          \
        }                                                                                \
        GTEST_SKIP() << (reason);                                                        \
    } while (false)
