#include <Platform/Input.hpp>

#include <gtest/gtest.h>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>

#include <cstddef>
#include <set>

// Nothing here initialises the windowing system or opens a window. InputState
// takes events and answers questions; a test can play the part of the window,
// which is the whole reason its logic knows nothing about GLFW.
//
// The test does know about GLFW, deliberately: the mapping is only worth
// asserting against the codes it actually has to match.

namespace {

using ysq::ButtonAction;
using ysq::InputState;
using ysq::Key;
using ysq::Modifier;
using ysq::Modifiers;
using ysq::MouseButton;

constexpr std::size_t kKeyCount = static_cast<std::size_t>(Key::Count);
constexpr std::size_t kButtonCount = static_cast<std::size_t>(MouseButton::Count);

Key keyAt(std::size_t index) {
    return static_cast<Key>(index);
}

void press(InputState& input, Key key, Modifiers modifiers = {}) {
    input.onKey(key, ButtonAction::Press, modifiers);
}

void release(InputState& input, Key key, Modifiers modifiers = {}) {
    input.onKey(key, ButtonAction::Release, modifiers);
}

}  // namespace

TEST(PlatformInput, EveryKeyRoundTripsThroughItsNativeCode) {
    for (std::size_t i = 0; i < kKeyCount; ++i) {
        const Key key = keyAt(i);
        EXPECT_EQ(ysq::keyFromNative(ysq::nativeFromKey(key)), key)
            << "key index " << i << " does not survive the round trip";
    }
}

TEST(PlatformInput, NativeCodesAreUnique) {
    // A duplicated row in the mapping table round-trips for one of the two keys
    // and silently aliases the other, so the round-trip test alone would not
    // catch it.
    std::set<int> seen;
    for (std::size_t i = 1; i < kKeyCount; ++i) {  // index 0 is Unknown
        const int native = ysq::nativeFromKey(keyAt(i));
        EXPECT_TRUE(seen.insert(native).second)
            << "native code " << native << " is claimed by two keys";
    }
}

TEST(PlatformInput, KeysMapToTheExpectedNativeCodes) {
    EXPECT_EQ(ysq::nativeFromKey(Key::A), GLFW_KEY_A);
    EXPECT_EQ(ysq::nativeFromKey(Key::Z), GLFW_KEY_Z);
    EXPECT_EQ(ysq::nativeFromKey(Key::D0), GLFW_KEY_0);
    EXPECT_EQ(ysq::nativeFromKey(Key::D9), GLFW_KEY_9);
    EXPECT_EQ(ysq::nativeFromKey(Key::Space), GLFW_KEY_SPACE);
    EXPECT_EQ(ysq::nativeFromKey(Key::Escape), GLFW_KEY_ESCAPE);
    EXPECT_EQ(ysq::nativeFromKey(Key::F25), GLFW_KEY_F25);
    EXPECT_EQ(ysq::nativeFromKey(Key::Kp0), GLFW_KEY_KP_0);
    EXPECT_EQ(ysq::nativeFromKey(Key::KpEqual), GLFW_KEY_KP_EQUAL);
    EXPECT_EQ(ysq::nativeFromKey(Key::Menu), GLFW_KEY_MENU);
    EXPECT_EQ(ysq::nativeFromKey(Key::World2), GLFW_KEY_WORLD_2);

    EXPECT_EQ(ysq::keyFromNative(GLFW_KEY_LEFT_SUPER), Key::LeftSuper);
    EXPECT_EQ(ysq::keyFromNative(GLFW_KEY_GRAVE_ACCENT), Key::GraveAccent);
}

TEST(PlatformInput, UnmappedNativeCodesAreUnknown) {
    EXPECT_EQ(ysq::keyFromNative(GLFW_KEY_UNKNOWN), Key::Unknown);
    EXPECT_EQ(ysq::keyFromNative(-42), Key::Unknown);
    EXPECT_EQ(ysq::keyFromNative(GLFW_KEY_LAST + 1), Key::Unknown);
    EXPECT_EQ(ysq::keyFromNative(58), Key::Unknown);  // between '9' and ';'
    EXPECT_EQ(ysq::nativeFromKey(Key::Unknown), GLFW_KEY_UNKNOWN);
    EXPECT_EQ(ysq::nativeFromKey(Key::Count), GLFW_KEY_UNKNOWN);
}

TEST(PlatformInput, MouseButtonsRoundTrip) {
    for (std::size_t i = 0; i < kButtonCount; ++i) {
        const auto button = static_cast<MouseButton>(i);
        EXPECT_EQ(ysq::mouseButtonFromNative(ysq::nativeFromMouseButton(button)), button);
    }
    EXPECT_EQ(ysq::nativeFromMouseButton(MouseButton::Left), GLFW_MOUSE_BUTTON_LEFT);
    EXPECT_EQ(ysq::nativeFromMouseButton(MouseButton::Middle), GLFW_MOUSE_BUTTON_MIDDLE);

    // No unknown button exists to name, so an unrecognised code must not come
    // back as Left.
    EXPECT_EQ(ysq::mouseButtonFromNative(99), MouseButton::Count);
    EXPECT_EQ(ysq::nativeFromMouseButton(MouseButton::Count), -1);
}

TEST(PlatformInput, ActionsRoundTripAndUnknownActionsRelease) {
    EXPECT_EQ(ysq::buttonActionFromNative(GLFW_PRESS), ButtonAction::Press);
    EXPECT_EQ(ysq::buttonActionFromNative(GLFW_RELEASE), ButtonAction::Release);
    EXPECT_EQ(ysq::buttonActionFromNative(GLFW_REPEAT), ButtonAction::Repeat);
    // A state that cannot be read must not leave a button stuck down.
    EXPECT_EQ(ysq::buttonActionFromNative(77), ButtonAction::Release);

    EXPECT_EQ(ysq::nativeFromButtonAction(ButtonAction::Press), GLFW_PRESS);
    EXPECT_EQ(ysq::nativeFromButtonAction(ButtonAction::Release), GLFW_RELEASE);
    EXPECT_EQ(ysq::nativeFromButtonAction(ButtonAction::Repeat), GLFW_REPEAT);
}

TEST(PlatformInput, ModifiersRoundTripAndCombine) {
    const Modifiers both = Modifier::Shift | Modifier::Control;
    EXPECT_TRUE(both.has(Modifier::Shift));
    EXPECT_TRUE(both.has(Modifier::Control));
    EXPECT_FALSE(both.has(Modifier::Alt));
    EXPECT_FALSE(both.none());
    EXPECT_TRUE(Modifiers{}.none());

    // An empty requirement matches anything, so a binding without modifiers is
    // not blocked by a stray caps lock.
    EXPECT_TRUE(both.contains(Modifiers{}));
    EXPECT_TRUE(both.contains(Modifier::Shift));
    EXPECT_FALSE(Modifiers{Modifier::Shift}.contains(both));

    const int native = ysq::nativeFromModifiers(both);
    EXPECT_EQ(native, GLFW_MOD_SHIFT | GLFW_MOD_CONTROL);
    EXPECT_EQ(ysq::modifiersFromNative(native), both);

    const Modifiers all = Modifier::Shift | Modifier::Control | Modifier::Alt |
                          Modifier::Super | Modifier::CapsLock | Modifier::NumLock;
    EXPECT_EQ(ysq::modifiersFromNative(ysq::nativeFromModifiers(all)), all);
    EXPECT_TRUE(ysq::modifiersFromNative(0).none());
}

TEST(PlatformInput, BitsThatNameNoModifierAreDropped) {
    // Two bits above NumLock belong to nothing. Keeping them would make a set
    // that no query can test for and that survives no round trip through the
    // native flags, so it would compare unequal to itself.
    const Modifiers everything = Modifiers::fromBits(0xFF);
    const Modifiers defined = Modifier::Shift | Modifier::Control | Modifier::Alt |
                              Modifier::Super | Modifier::CapsLock | Modifier::NumLock;

    EXPECT_EQ(everything, defined);
    EXPECT_EQ(ysq::modifiersFromNative(ysq::nativeFromModifiers(everything)), everything);
}

TEST(PlatformInput, PressIsAnEdgeAndDownIsAState) {
    InputState input;
    EXPECT_FALSE(input.keyDown(Key::W));

    press(input, Key::W);
    EXPECT_TRUE(input.keyDown(Key::W));
    EXPECT_TRUE(input.keyPressed(Key::W));
    EXPECT_FALSE(input.keyReleased(Key::W));

    input.newFrame();
    EXPECT_TRUE(input.keyDown(Key::W)) << "holding a key must survive the frame";
    EXPECT_FALSE(input.keyPressed(Key::W)) << "a press is an edge, not a state";

    release(input, Key::W);
    EXPECT_FALSE(input.keyDown(Key::W));
    EXPECT_TRUE(input.keyReleased(Key::W));

    input.newFrame();
    EXPECT_FALSE(input.keyReleased(Key::W));
}

TEST(PlatformInput, TapWithinOneFrameReportsBothEdges) {
    InputState input;
    press(input, Key::Space);
    release(input, Key::Space);

    EXPECT_TRUE(input.keyPressed(Key::Space)) << "a fast tap must not be lost";
    EXPECT_TRUE(input.keyReleased(Key::Space));
    EXPECT_FALSE(input.keyDown(Key::Space));
}

TEST(PlatformInput, AutoRepeatIsNotAFreshPress) {
    InputState input;
    press(input, Key::D);
    input.newFrame();

    input.onKey(Key::D, ButtonAction::Repeat, {});
    EXPECT_TRUE(input.keyDown(Key::D));
    EXPECT_FALSE(input.keyPressed(Key::D))
        << "repeat would fire a once-per-press action at the keyboard repeat rate";
}

TEST(PlatformInput, RepeatArrivingWithoutAPressStillCountsAsHeld) {
    // The window system can deliver a repeat for a key that went down before the
    // window had focus. Ignoring it would leave a visibly held key reading up.
    InputState input;
    input.onKey(Key::D, ButtonAction::Repeat, {});
    EXPECT_TRUE(input.keyDown(Key::D));
    EXPECT_FALSE(input.keyPressed(Key::D));
}

TEST(PlatformInput, MouseButtonsFollowTheSameEdgeRules) {
    InputState input;
    input.onMouseButton(MouseButton::Left, ButtonAction::Press, {});
    EXPECT_TRUE(input.mouseButtonDown(MouseButton::Left));
    EXPECT_TRUE(input.mouseButtonPressed(MouseButton::Left));
    EXPECT_FALSE(input.mouseButtonDown(MouseButton::Right));

    input.newFrame();
    EXPECT_TRUE(input.mouseButtonDown(MouseButton::Left));
    EXPECT_FALSE(input.mouseButtonPressed(MouseButton::Left));

    input.onMouseButton(MouseButton::Left, ButtonAction::Release, {});
    EXPECT_FALSE(input.mouseButtonDown(MouseButton::Left));
    EXPECT_TRUE(input.mouseButtonReleased(MouseButton::Left));
}

TEST(PlatformInput, ModifiersTrackTheLatestEvent) {
    InputState input;
    press(input, Key::S, Modifier::Control);
    EXPECT_TRUE(input.modifiers().has(Modifier::Control));

    release(input, Key::S, Modifiers{});
    EXPECT_TRUE(input.modifiers().none());
}

TEST(PlatformInput, OutOfRangeKeysAreIgnoredRatherThanScribbling) {
    InputState input;
    press(input, Key::Count);
    press(input, static_cast<Key>(9999));

    EXPECT_FALSE(input.keyDown(Key::Count));
    EXPECT_FALSE(input.keyPressed(static_cast<Key>(9999)));
    for (std::size_t i = 0; i < kKeyCount; ++i) {
        EXPECT_FALSE(input.keyDown(keyAt(i))) << "key index " << i << " was scribbled on";
    }
}

TEST(PlatformInput, FirstCursorSightingIsAPositionNotAMovement) {
    InputState input;
    EXPECT_FALSE(input.hasCursor());
    EXPECT_EQ(input.cursorDelta(), ysq::CursorPosition{});

    input.onCursorPosition(400.0, 300.0);
    EXPECT_TRUE(input.hasCursor());
    EXPECT_EQ(input.cursor(), (ysq::CursorPosition{400.0, 300.0}));
    EXPECT_EQ(input.cursorDelta(), ysq::CursorPosition{})
        << "the first sample is where the cursor is, not how far it moved";
}

TEST(PlatformInput, CursorDeltaMeasuresFromTheFrameStart) {
    InputState input;
    input.onCursorPosition(100.0, 100.0);

    input.newFrame();
    input.onCursorPosition(110.0, 95.0);
    EXPECT_EQ(input.cursorDelta(), (ysq::CursorPosition{10.0, -5.0}));

    // Several samples in one frame are one movement, not several.
    input.onCursorPosition(120.0, 90.0);
    EXPECT_EQ(input.cursorDelta(), (ysq::CursorPosition{20.0, -10.0}));

    input.newFrame();
    EXPECT_EQ(input.cursorDelta(), ysq::CursorPosition{});
}

TEST(PlatformInput, ScrollAccumulatesWithinAFrame) {
    InputState input;
    input.onScroll(0.0, 1.0);
    input.onScroll(0.0, 2.0);
    input.onScroll(-1.0, 0.0);

    EXPECT_EQ(input.scrollDelta(), (ysq::ScrollOffset{-1.0, 3.0}))
        << "a wheel notch can arrive as several events";

    input.newFrame();
    EXPECT_EQ(input.scrollDelta(), ysq::ScrollOffset{});
}

TEST(PlatformInput, LosingFocusReleasesEverythingHeld) {
    InputState input;
    press(input, Key::W, Modifier::Shift);
    input.onMouseButton(MouseButton::Right, ButtonAction::Press, Modifier::Shift);
    input.newFrame();
    ASSERT_TRUE(input.keyDown(Key::W));

    input.onFocusLost();

    EXPECT_FALSE(input.keyDown(Key::W))
        << "a key held across a focus change never delivers its release";
    EXPECT_TRUE(input.keyReleased(Key::W)) << "the release must still be observable";
    EXPECT_FALSE(input.mouseButtonDown(MouseButton::Right));
    EXPECT_TRUE(input.mouseButtonReleased(MouseButton::Right));
    EXPECT_TRUE(input.modifiers().none());
}

TEST(PlatformInput, ReturningFromFocusLossDoesNotReadAsOneHugeMovement) {
    InputState input;
    input.onCursorPosition(100.0, 100.0);
    input.newFrame();

    input.onFocusLost();
    EXPECT_FALSE(input.hasCursor());

    // The cursor comes back somewhere else entirely.
    input.onCursorPosition(900.0, 20.0);
    EXPECT_EQ(input.cursorDelta(), ysq::CursorPosition{});
    EXPECT_EQ(input.cursor(), (ysq::CursorPosition{900.0, 20.0}));

    input.newFrame();
    input.onCursorPosition(905.0, 20.0);
    EXPECT_EQ(input.cursorDelta(), (ysq::CursorPosition{5.0, 0.0}));
}

TEST(PlatformInput, ResetReturnsToTheInitialState) {
    InputState input;
    press(input, Key::W);
    input.onCursorPosition(10.0, 10.0);
    input.onScroll(1.0, 1.0);

    input.reset();

    EXPECT_FALSE(input.keyDown(Key::W));
    EXPECT_FALSE(input.keyPressed(Key::W));
    EXPECT_FALSE(input.hasCursor());
    EXPECT_EQ(input.cursor(), ysq::CursorPosition{});
    EXPECT_EQ(input.scrollDelta(), ysq::ScrollOffset{});
}

TEST(PlatformInput, SuppressedMouseReadsAsNothingHappening) {
    InputState input;
    input.onCursorPosition(100.0, 100.0);
    input.onMouseButton(MouseButton::Left, ButtonAction::Press, {});
    input.newFrame();
    input.onCursorPosition(110.0, 95.0);
    input.onScroll(0.0, 2.0);
    ASSERT_TRUE(input.mouseButtonDown(MouseButton::Left));
    ASSERT_EQ(input.cursorDelta(), (ysq::CursorPosition{10.0, -5.0}));
    ASSERT_EQ(input.scrollDelta(), (ysq::ScrollOffset{0.0, 2.0}));

    input.suppressMouseThisFrame();

    EXPECT_FALSE(input.mouseButtonDown(MouseButton::Left));
    EXPECT_FALSE(input.mouseButtonPressed(MouseButton::Left));
    EXPECT_FALSE(input.mouseButtonReleased(MouseButton::Left));
    EXPECT_EQ(input.cursorDelta(), ysq::CursorPosition{});
    EXPECT_EQ(input.scrollDelta(), ysq::ScrollOffset{});
}

TEST(PlatformInput, SuppressingMouseLeavesKeysAlone) {
    InputState input;
    press(input, Key::W);

    input.suppressMouseThisFrame();

    EXPECT_TRUE(input.keyDown(Key::W))
        << "mouse suppression must not touch keyboard state";
}

TEST(PlatformInput, MouseSuppressionDoesNotMutateTheUnderlyingState) {
    InputState input;
    input.onMouseButton(MouseButton::Left, ButtonAction::Press, {});
    input.suppressMouseThisFrame();
    ASSERT_FALSE(input.mouseButtonDown(MouseButton::Left)) << "suppressed for this frame";

    input.newFrame();

    EXPECT_TRUE(input.mouseButtonDown(MouseButton::Left))
        << "the real button state must have survived suppression, unaffected";
}

TEST(PlatformInput, NewFrameLiftsSuppressionFromThePreviousFrame) {
    InputState input;
    input.onCursorPosition(0.0, 0.0);
    input.suppressMouseThisFrame();

    input.newFrame();
    input.onCursorPosition(10.0, 0.0);

    EXPECT_EQ(input.cursorDelta(), (ysq::CursorPosition{10.0, 0.0}))
        << "suppression must not carry over into the next frame";
}
