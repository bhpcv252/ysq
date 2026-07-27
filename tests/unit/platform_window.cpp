#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <string>
#include <utility>

// Everything here runs on the Null backend, which needs no display server and
// is compiled in everywhere. It stops short of creating a context, because that
// is where a software rasteriser has to exist; that part is
// tests/integration/platform_context.cpp.
//
// The settings a window refuses are checked before the windowing system is
// asked anything, so these failures are identical on every machine.

namespace {

using ysq::Platform;
using ysq::PlatformBackend;
using ysq::PlatformError;
using ysq::PlatformSettings;
using ysq::Window;
using ysq::WindowError;
using ysq::WindowErrorCode;
using ysq::WindowSettings;

PlatformSettings nullBackend() {
    PlatformSettings settings;
    settings.backend = PlatformBackend::Null;
    return settings;
}

/// A window that would be valid, so a rejection in a test is the one thing that
/// test changed.
WindowSettings validSettings() {
    WindowSettings settings;
    settings.title = "test";
    settings.width = 64;
    settings.height = 32;
    settings.visible = false;
    return settings;
}

/// The reason create() refused, or nothing if it did not refuse.
std::optional<WindowError> rejection(const WindowSettings& settings) {
    WindowError error;
    if (Window::create(settings, &error)) {
        return std::nullopt;
    }
    return error;
}

}  // namespace

TEST(PlatformWindow, CreateFailsBeforeThePlatformIsInitialized) {
    ASSERT_FALSE(Platform::initialized());

    WindowError error;
    EXPECT_FALSE(Window::create(validSettings(), &error));
    EXPECT_EQ(error.code, WindowErrorCode::PlatformNotInitialized);
    EXPECT_FALSE(error.message.empty());
}

TEST(PlatformWindow, QueriesAreSafeBeforeInitialization) {
    ASSERT_FALSE(Platform::initialized());

    EXPECT_EQ(Platform::backend(), std::nullopt);
}

TEST(PlatformWindow, TheNullBackendIsAlwaysCompiledIn) {
    // It is the one backend that needs no display, so the headless path can be
    // asked for by name on any machine.
    EXPECT_TRUE(Platform::backendAvailable(PlatformBackend::Null));
}

TEST(PlatformWindow, InitializeSelectsTheRequestedBackend) {
    PlatformError error;
    const std::optional<Platform> platform = Platform::initialize(nullBackend(), &error);
    ASSERT_TRUE(platform) << error.message;

    EXPECT_TRUE(Platform::initialized());
    EXPECT_EQ(Platform::backend(), PlatformBackend::Null);
}

TEST(PlatformWindow, HandlesAreReferenceCounted) {
    {
        const std::optional<Platform> first = Platform::initialize(nullBackend());
        ASSERT_TRUE(first);
        {
            const std::optional<Platform> second = Platform::initialize(nullBackend());
            ASSERT_TRUE(second);
            const Platform copy = *second;
            EXPECT_TRUE(Platform::initialized());
        }
        EXPECT_TRUE(Platform::initialized())
            << "the outer handle still holds the platform open";
    }
    EXPECT_FALSE(Platform::initialized()) << "the last handle shuts it down";
}

TEST(PlatformWindow, MovedFromHandlesReleaseNothing) {
    {
        std::optional<Platform> platform = Platform::initialize(nullBackend());
        ASSERT_TRUE(platform);

        const Platform moved = std::move(*platform);
        platform.reset();  // destroys the husk, which owns nothing
        EXPECT_TRUE(Platform::initialized());
    }
    EXPECT_FALSE(Platform::initialized());
}

TEST(PlatformWindow, CurrentRetainsButDoesNotInitialize) {
    EXPECT_FALSE(Platform::current().has_value());

    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);
    EXPECT_TRUE(Platform::current().has_value());
}

TEST(PlatformWindow, AskingForADifferentBackendFails) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    PlatformSettings other;
    other.backend = PlatformBackend::X11;

    PlatformError error;
    EXPECT_FALSE(Platform::initialize(other, &error))
        << "a test that asked for Null must never silently get something else";
    EXPECT_FALSE(error.message.empty());
    EXPECT_EQ(Platform::backend(), PlatformBackend::Null);
}

TEST(PlatformWindow, AutoSelectionJoinsWhateverIsAlreadyRunning) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    EXPECT_TRUE(Platform::initialize()) << "no backend was requested, so none conflicts";
    EXPECT_EQ(Platform::backend(), PlatformBackend::Null);
}

TEST(PlatformWindow, NonPositiveSizesAreRejected) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    WindowSettings settings = validSettings();
    settings.width = 0;
    const std::optional<WindowError> zero = rejection(settings);
    ASSERT_TRUE(zero);
    EXPECT_EQ(zero->code, WindowErrorCode::InvalidSettings);

    settings.width = 640;
    settings.height = -1;
    const std::optional<WindowError> negative = rejection(settings);
    ASSERT_TRUE(negative);
    EXPECT_EQ(negative->code, WindowErrorCode::InvalidSettings);
}

TEST(PlatformWindow, VersionsAboveTheLoaderAreRejected) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    // The loader is generated for a fixed version. A driver claiming to satisfy
    // a higher request would leave half its entry points null, so the request
    // never reaches the driver at all.
    WindowSettings settings = validSettings();
    settings.context.versionMajor = 9;
    settings.context.versionMinor = 9;

    const std::optional<WindowError> error = rejection(settings);
    ASSERT_TRUE(error);
    EXPECT_EQ(error->code, WindowErrorCode::InvalidSettings);
    EXPECT_NE(error->message.find("9.9"), std::string::npos);

    settings.context.versionMajor = ysq::kMaxSupportedGL.major;
    settings.context.versionMinor = ysq::kMaxSupportedGL.minor;
    const std::optional<WindowError> atCeiling = rejection(settings);
    if (atCeiling) {
        EXPECT_NE(atCeiling->code, WindowErrorCode::InvalidSettings)
            << "the loader's own version must not be rejected as a setting";
    }
}

TEST(PlatformWindow, ProfileRequirementsAreRejectedBeforeTheDriverSeesThem) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    WindowSettings core = validSettings();
    core.context.versionMajor = 2;
    core.context.versionMinor = 1;
    core.context.coreProfile = true;
    core.context.forwardCompatible = false;
    const std::optional<WindowError> coreError = rejection(core);
    ASSERT_TRUE(coreError);
    EXPECT_EQ(coreError->code, WindowErrorCode::InvalidSettings);
    EXPECT_NE(coreError->message.find("3.2"), std::string::npos);

    WindowSettings forward = validSettings();
    forward.context.versionMajor = 2;
    forward.context.versionMinor = 1;
    forward.context.coreProfile = false;
    forward.context.forwardCompatible = true;
    const std::optional<WindowError> forwardError = rejection(forward);
    ASSERT_TRUE(forwardError);
    EXPECT_EQ(forwardError->code, WindowErrorCode::InvalidSettings);
    EXPECT_NE(forwardError->message.find("3.0"), std::string::npos);
}

TEST(PlatformWindow, NegativeBufferSizesAreRejected) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    WindowSettings settings = validSettings();
    settings.context.samples = -4;

    const std::optional<WindowError> error = rejection(settings);
    ASSERT_TRUE(error);
    EXPECT_EQ(error->code, WindowErrorCode::InvalidSettings);
}

TEST(PlatformWindow, TheNullBackendDropsForwardCompatibility) {
    // A pure function, so what each backend does to a request is testable with
    // no platform, no window and no context. OSMesa refuses forward-compatible
    // contexts outright, and forward compatibility is on by default because
    // macOS needs it, so without this adjustment the default settings could
    // never open a headless context at all.
    ysq::ContextSettings requested;
    ASSERT_TRUE(requested.forwardCompatible);

    const ysq::ContextSettings onNull =
        ysq::effectiveContext(requested, PlatformBackend::Null);
    EXPECT_FALSE(onNull.forwardCompatible);
    EXPECT_TRUE(onNull.coreProfile) << "only forward compatibility is at issue";
    EXPECT_EQ(onNull.versionMajor, requested.versionMajor);
    EXPECT_EQ(onNull.versionMinor, requested.versionMinor);

    for (const PlatformBackend backend :
         {PlatformBackend::Win32, PlatformBackend::Cocoa, PlatformBackend::Wayland,
          PlatformBackend::X11}) {
        EXPECT_TRUE(ysq::effectiveContext(requested, backend).forwardCompatible)
            << "only the Null backend refuses it";
    }

    // Nothing is added that was not asked for.
    ysq::ContextSettings plain;
    plain.forwardCompatible = false;
    EXPECT_FALSE(ysq::effectiveContext(plain, PlatformBackend::Cocoa).forwardCompatible);
}

TEST(PlatformWindow, WaitingForEventsReturnsRatherThanBlocking) {
    const std::optional<Platform> platform = Platform::initialize(nullBackend());
    ASSERT_TRUE(platform);

    // A timeout that is not a timeout must not become an indefinite block. That
    // failure mode is a hang, which gets blamed on everything except the call
    // that caused it.
    Platform::waitEventsFor(0.0);
    Platform::waitEventsFor(-1.0);
    Platform::waitEventsFor(std::numeric_limits<double>::quiet_NaN());
    Platform::waitEventsFor(0.001);

    // The Null backend has no event source, so this returns at once. On a real
    // backend it would block, which is why it is not called here.
    Platform::waitEvents();
}

TEST(PlatformWindow, EventPumpingIsSafeBeforeInitialization) {
    ASSERT_FALSE(Platform::initialized());

    // No-ops rather than errors, so a headless run need not guard every call.
    // waitEvents() would block forever if it reached the windowing system, so
    // this also pins that the guard comes first.
    Platform::pollEvents();
    Platform::waitEvents();
    Platform::waitEventsFor(1.0);
}

TEST(PlatformWindow, DefaultSettingsAskForTheVersionMacOSCaps) {
    // 4.1 core, forward compatible. Anything else and the visualiser stops
    // working on macOS, which is a first-class rendering target.
    const WindowSettings settings;
    EXPECT_EQ(settings.context.versionMajor, 4);
    EXPECT_EQ(settings.context.versionMinor, 1);
    EXPECT_TRUE(settings.context.coreProfile);
    EXPECT_TRUE(settings.context.forwardCompatible);
}
