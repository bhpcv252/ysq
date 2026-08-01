#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>

namespace ysq {

/// Physical keys, in the US layout positions GLFW reports.
///
/// The values are dense and start at zero so key state is a bitset and a
/// mapping table is an array. GLFW's own codes are sparse (32, 39, 44-96,
/// 161-162, 256-348); translating between the two is Input.cpp's job, and it is
/// the reason GLFW does not appear in this header at all.
///
/// A key names a position, not a character. `Key::Q` is the key where Q sits on
/// a US keyboard whatever the active layout prints on it, which is what a
/// movement binding wants. Layout-correct text entry is a separate problem and
/// is not solved here.
enum class Key : std::uint16_t {
    Unknown = 0,

    Space,
    Apostrophe,
    Comma,
    Minus,
    Period,
    Slash,

    D0,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,
    D9,

    Semicolon,
    Equal,

    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    LeftBracket,
    Backslash,
    RightBracket,
    GraveAccent,

    /// The extra keys on non-US layouts, which have no agreed name.
    World1,
    World2,

    Escape,
    Enter,
    Tab,
    Backspace,
    Insert,
    Delete,
    Right,
    Left,
    Down,
    Up,
    PageUp,
    PageDown,
    Home,
    End,

    CapsLock,
    ScrollLock,
    NumLock,
    PrintScreen,
    Pause,

    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,
    F25,

    Kp0,
    Kp1,
    Kp2,
    Kp3,
    Kp4,
    Kp5,
    Kp6,
    Kp7,
    Kp8,
    Kp9,
    KpDecimal,
    KpDivide,
    KpMultiply,
    KpSubtract,
    KpAdd,
    KpEnter,
    KpEqual,

    LeftShift,
    LeftControl,
    LeftAlt,
    LeftSuper,
    RightShift,
    RightControl,
    RightAlt,
    RightSuper,
    Menu,

    Count
};

enum class MouseButton : std::uint8_t {
    Left = 0,
    Right,
    Middle,
    Button4,
    Button5,
    Button6,
    Button7,
    Button8,

    Count
};

/// What an event did to a button. Repeat is the auto-repeat the window system
/// generates while a key is held; it is not a fresh press.
enum class ButtonAction : std::uint8_t { Release, Press, Repeat };

enum class Modifier : std::uint8_t {
    Shift = 1u << 0,
    Control = 1u << 1,
    Alt = 1u << 2,
    Super = 1u << 3,
    CapsLock = 1u << 4,
    NumLock = 1u << 5,
};

/// A set of Modifier flags.
///
/// A class rather than a bitmask over the enum itself, so that `Modifier::Shift`
/// alone is a valid set, an empty set is spellable, and `|` cannot silently
/// produce a value that is not a valid Modifier.
class Modifiers {
public:
    constexpr Modifiers() noexcept = default;
    constexpr Modifiers(Modifier modifier) noexcept  // NOLINT: implicit by design
        : m_bits(static_cast<std::uint8_t>(modifier)) {}

    /// Bits with no Modifier are dropped, so a set cannot carry a flag that
    /// nothing can test for and that would survive no round trip.
    [[nodiscard]] static constexpr Modifiers fromBits(std::uint8_t bits) noexcept {
        Modifiers set;
        set.m_bits = static_cast<std::uint8_t>(bits & kDefinedBits);
        return set;
    }

    [[nodiscard]] constexpr std::uint8_t bits() const noexcept { return m_bits; }
    [[nodiscard]] constexpr bool none() const noexcept { return m_bits == 0; }

    [[nodiscard]] constexpr bool has(Modifier modifier) const noexcept {
        const auto flag = static_cast<std::uint8_t>(modifier);
        return (m_bits & flag) == flag;
    }

    /// True when every modifier in `other` is set. An empty set is contained in
    /// anything, so a binding with no modifiers matches regardless of them.
    [[nodiscard]] constexpr bool contains(Modifiers other) const noexcept {
        return (m_bits & other.m_bits) == other.m_bits;
    }

    constexpr Modifiers& operator|=(Modifiers other) noexcept {
        m_bits = static_cast<std::uint8_t>(m_bits | other.m_bits);
        return *this;
    }

    constexpr Modifiers& operator&=(Modifiers other) noexcept {
        m_bits = static_cast<std::uint8_t>(m_bits & other.m_bits);
        return *this;
    }

    [[nodiscard]] friend constexpr Modifiers operator|(Modifiers a,
                                                       Modifiers b) noexcept {
        return fromBits(static_cast<std::uint8_t>(a.m_bits | b.m_bits));
    }

    [[nodiscard]] friend constexpr Modifiers operator&(Modifiers a,
                                                       Modifiers b) noexcept {
        return fromBits(static_cast<std::uint8_t>(a.m_bits & b.m_bits));
    }

    [[nodiscard]] friend constexpr bool operator==(Modifiers,
                                                   Modifiers) noexcept = default;

private:
    /// Every bit that names a Modifier. The two above them are not ours.
    static constexpr std::uint8_t kDefinedBits =
        static_cast<std::uint8_t>(Modifier::Shift) |
        static_cast<std::uint8_t>(Modifier::Control) |
        static_cast<std::uint8_t>(Modifier::Alt) |
        static_cast<std::uint8_t>(Modifier::Super) |
        static_cast<std::uint8_t>(Modifier::CapsLock) |
        static_cast<std::uint8_t>(Modifier::NumLock);

    std::uint8_t m_bits = 0;
};

[[nodiscard]] constexpr Modifiers operator|(Modifier a, Modifier b) noexcept {
    return Modifiers{a} | Modifiers{b};
}

struct CursorPosition {
    double x = 0.0;
    double y = 0.0;

    [[nodiscard]] friend constexpr bool operator==(CursorPosition,
                                                   CursorPosition) noexcept = default;
};

struct ScrollOffset {
    double x = 0.0;
    double y = 0.0;

    [[nodiscard]] friend constexpr bool operator==(ScrollOffset,
                                                   ScrollOffset) noexcept = default;
};

/// Unrecognised native codes map to Key::Unknown, and Key::Unknown maps back to
/// the native "unknown" code. Every other key round-trips.
[[nodiscard]] Key keyFromNative(int nativeCode) noexcept;
[[nodiscard]] int nativeFromKey(Key key) noexcept;

/// There is no unknown mouse button to name, so an unrecognised code yields
/// MouseButton::Count and an unrecognised button yields -1. Neither matches any
/// query, which is the point: they cannot be mistaken for Left.
[[nodiscard]] MouseButton mouseButtonFromNative(int nativeCode) noexcept;
[[nodiscard]] int nativeFromMouseButton(MouseButton button) noexcept;

/// An unrecognised action is reported as Release, which is the safe reading:
/// a button state that cannot be interpreted must not stay stuck down.
[[nodiscard]] ButtonAction buttonActionFromNative(int nativeAction) noexcept;
[[nodiscard]] int nativeFromButtonAction(ButtonAction action) noexcept;

[[nodiscard]] Modifiers modifiersFromNative(int nativeBits) noexcept;
[[nodiscard]] int nativeFromModifiers(Modifiers modifiers) noexcept;

/// Keyboard and mouse state for one window, sampled per frame.
///
/// The on* methods take events and the query methods answer questions about the
/// frame in progress. Window feeds this from its GLFW callbacks; a test feeds it
/// directly, which is why nothing here knows what GLFW is.
///
///     input.newFrame();          // clears the edges, snapshots the cursor
///     Platform::pollEvents();    // callbacks land here
///     if (input.keyPressed(Key::Space)) { ... }
///
/// Three questions per button, and they differ:
///
///   keyDown      held right now
///   keyPressed   went down during this frame
///   keyReleased  came up during this frame
///
/// A key pressed and released inside one frame reports both edges and is not
/// down, so a fast tap is never missed. Auto-repeat sets neither edge: the key
/// was already down, and treating repeat as a fresh press would fire a
/// once-per-press action at the keyboard's repeat rate.
///
/// Not thread-safe. Window events arrive on the thread that polls.
class InputState {
public:
    void onKey(Key key, ButtonAction action, Modifiers modifiers) noexcept;
    void onMouseButton(MouseButton button, ButtonAction action,
                       Modifiers modifiers) noexcept;
    void onCursorPosition(double x, double y) noexcept;
    void onScroll(double x, double y) noexcept;

    /// Everything held is released and the cursor baseline is dropped.
    ///
    /// A key held while the window loses focus never delivers its release, so
    /// without this it stays down forever. The synthetic releases are reported
    /// as edges, so a handler watching for a release still sees one. Dropping
    /// the baseline stops the jump back into the window from reading as one
    /// enormous mouse movement.
    void onFocusLost() noexcept;

    /// Call once per frame, before polling. Clears the press and release edges
    /// and the scroll accumulator, and takes the cursor snapshot that
    /// cursorDelta() measures from.
    void newFrame() noexcept;

    /// Makes every mouse button/cursor-delta/scroll query answer as if
    /// nothing were happening, for the rest of the current frame -- call
    /// once per frame, before driving anything that reads mouse input, when
    /// something else (a UI layer) has already claimed the mouse. A
    /// query-time override, not a mutation of the real tracked state: the
    /// underlying button-down/cursor/scroll state is untouched and reads
    /// correctly again as soon as newFrame() resets this on the next frame.
    /// Keyboard queries are unaffected.
    void suppressMouseThisFrame() noexcept { m_mouseSuppressed = true; }

    /// Back to the state of a freshly constructed InputState.
    void reset() noexcept;

    [[nodiscard]] bool keyDown(Key key) const noexcept;
    [[nodiscard]] bool keyPressed(Key key) const noexcept;
    [[nodiscard]] bool keyReleased(Key key) const noexcept;

    /// False while suppressMouseThisFrame() is in effect, regardless of the
    /// real underlying state.
    [[nodiscard]] bool mouseButtonDown(MouseButton button) const noexcept;
    [[nodiscard]] bool mouseButtonPressed(MouseButton button) const noexcept;
    [[nodiscard]] bool mouseButtonReleased(MouseButton button) const noexcept;

    /// As of the most recent key or button event. Modifier keys held without any
    /// other key generate events of their own, so this tracks them too.
    [[nodiscard]] Modifiers modifiers() const noexcept { return m_modifiers; }

    [[nodiscard]] CursorPosition cursor() const noexcept { return m_cursor; }

    /// Movement since the last newFrame(), in the same coordinates as cursor().
    /// Zero until the cursor has been seen at least once, or while
    /// suppressMouseThisFrame() is in effect.
    [[nodiscard]] CursorPosition cursorDelta() const noexcept;

    /// True once a cursor position has been received, so a first frame can tell
    /// "at the origin" from "not seen yet".
    [[nodiscard]] bool hasCursor() const noexcept { return m_hasCursor; }

    /// Scroll accumulated during this frame. A wheel emits several events per
    /// notch on some drivers, so these are summed rather than replaced. Zero
    /// while suppressMouseThisFrame() is in effect.
    [[nodiscard]] ScrollOffset scrollDelta() const noexcept {
        return m_mouseSuppressed ? ScrollOffset{} : m_scroll;
    }

private:
    static constexpr std::size_t kKeyCount = static_cast<std::size_t>(Key::Count);
    static constexpr std::size_t kButtonCount =
        static_cast<std::size_t>(MouseButton::Count);

    std::bitset<kKeyCount> m_keysDown;
    std::bitset<kKeyCount> m_keysPressed;
    std::bitset<kKeyCount> m_keysReleased;

    std::bitset<kButtonCount> m_buttonsDown;
    std::bitset<kButtonCount> m_buttonsPressed;
    std::bitset<kButtonCount> m_buttonsReleased;

    Modifiers m_modifiers{};
    CursorPosition m_cursor{};
    CursorPosition m_previousCursor{};
    ScrollOffset m_scroll{};
    bool m_hasCursor = false;
    bool m_mouseSuppressed = false;
};

}  // namespace ysq
