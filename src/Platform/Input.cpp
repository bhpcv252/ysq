#include <Platform/Input.hpp>

#define GLFW_INCLUDE_NONE  // GLAD provides the GL headers, not GLFW
#include <GLFW/glfw3.h>

#include <array>

namespace ysq {

namespace {

constexpr std::size_t kKeyCount = static_cast<std::size_t>(Key::Count);
constexpr std::size_t kButtonCount = static_cast<std::size_t>(MouseButton::Count);

/// Indexed by Key, so this is the one place the two orderings meet. Adding a key
/// means adding a row here in the same position; the static_assert below catches
/// a row forgotten, and the round-trip test in tests/unit/platform_input.cpp
/// catches one in the wrong place.
constexpr std::array<int, kKeyCount> kNativeKeys{
    GLFW_KEY_UNKNOWN,

    GLFW_KEY_SPACE,
    GLFW_KEY_APOSTROPHE,
    GLFW_KEY_COMMA,
    GLFW_KEY_MINUS,
    GLFW_KEY_PERIOD,
    GLFW_KEY_SLASH,

    GLFW_KEY_0,
    GLFW_KEY_1,
    GLFW_KEY_2,
    GLFW_KEY_3,
    GLFW_KEY_4,
    GLFW_KEY_5,
    GLFW_KEY_6,
    GLFW_KEY_7,
    GLFW_KEY_8,
    GLFW_KEY_9,

    GLFW_KEY_SEMICOLON,
    GLFW_KEY_EQUAL,

    GLFW_KEY_A,
    GLFW_KEY_B,
    GLFW_KEY_C,
    GLFW_KEY_D,
    GLFW_KEY_E,
    GLFW_KEY_F,
    GLFW_KEY_G,
    GLFW_KEY_H,
    GLFW_KEY_I,
    GLFW_KEY_J,
    GLFW_KEY_K,
    GLFW_KEY_L,
    GLFW_KEY_M,
    GLFW_KEY_N,
    GLFW_KEY_O,
    GLFW_KEY_P,
    GLFW_KEY_Q,
    GLFW_KEY_R,
    GLFW_KEY_S,
    GLFW_KEY_T,
    GLFW_KEY_U,
    GLFW_KEY_V,
    GLFW_KEY_W,
    GLFW_KEY_X,
    GLFW_KEY_Y,
    GLFW_KEY_Z,

    GLFW_KEY_LEFT_BRACKET,
    GLFW_KEY_BACKSLASH,
    GLFW_KEY_RIGHT_BRACKET,
    GLFW_KEY_GRAVE_ACCENT,

    GLFW_KEY_WORLD_1,
    GLFW_KEY_WORLD_2,

    GLFW_KEY_ESCAPE,
    GLFW_KEY_ENTER,
    GLFW_KEY_TAB,
    GLFW_KEY_BACKSPACE,
    GLFW_KEY_INSERT,
    GLFW_KEY_DELETE,
    GLFW_KEY_RIGHT,
    GLFW_KEY_LEFT,
    GLFW_KEY_DOWN,
    GLFW_KEY_UP,
    GLFW_KEY_PAGE_UP,
    GLFW_KEY_PAGE_DOWN,
    GLFW_KEY_HOME,
    GLFW_KEY_END,

    GLFW_KEY_CAPS_LOCK,
    GLFW_KEY_SCROLL_LOCK,
    GLFW_KEY_NUM_LOCK,
    GLFW_KEY_PRINT_SCREEN,
    GLFW_KEY_PAUSE,

    GLFW_KEY_F1,
    GLFW_KEY_F2,
    GLFW_KEY_F3,
    GLFW_KEY_F4,
    GLFW_KEY_F5,
    GLFW_KEY_F6,
    GLFW_KEY_F7,
    GLFW_KEY_F8,
    GLFW_KEY_F9,
    GLFW_KEY_F10,
    GLFW_KEY_F11,
    GLFW_KEY_F12,
    GLFW_KEY_F13,
    GLFW_KEY_F14,
    GLFW_KEY_F15,
    GLFW_KEY_F16,
    GLFW_KEY_F17,
    GLFW_KEY_F18,
    GLFW_KEY_F19,
    GLFW_KEY_F20,
    GLFW_KEY_F21,
    GLFW_KEY_F22,
    GLFW_KEY_F23,
    GLFW_KEY_F24,
    GLFW_KEY_F25,

    GLFW_KEY_KP_0,
    GLFW_KEY_KP_1,
    GLFW_KEY_KP_2,
    GLFW_KEY_KP_3,
    GLFW_KEY_KP_4,
    GLFW_KEY_KP_5,
    GLFW_KEY_KP_6,
    GLFW_KEY_KP_7,
    GLFW_KEY_KP_8,
    GLFW_KEY_KP_9,
    GLFW_KEY_KP_DECIMAL,
    GLFW_KEY_KP_DIVIDE,
    GLFW_KEY_KP_MULTIPLY,
    GLFW_KEY_KP_SUBTRACT,
    GLFW_KEY_KP_ADD,
    GLFW_KEY_KP_ENTER,
    GLFW_KEY_KP_EQUAL,

    GLFW_KEY_LEFT_SHIFT,
    GLFW_KEY_LEFT_CONTROL,
    GLFW_KEY_LEFT_ALT,
    GLFW_KEY_LEFT_SUPER,
    GLFW_KEY_RIGHT_SHIFT,
    GLFW_KEY_RIGHT_CONTROL,
    GLFW_KEY_RIGHT_ALT,
    GLFW_KEY_RIGHT_SUPER,
    GLFW_KEY_MENU,
};

static_assert(kNativeKeys.size() == kKeyCount,
              "kNativeKeys must have one entry per Key");
static_assert(kNativeKeys[0] == GLFW_KEY_UNKNOWN,
              "Key::Unknown must be first, so a zeroed lookup table means unknown");

/// Native codes are sparse and stop at GLFW_KEY_LAST, so the reverse direction
/// is a direct-indexed table rather than a search. It is built from kNativeKeys
/// at compile time, which is what keeps the two directions from drifting apart.
constexpr auto kKeysByNative = [] {
    std::array<Key, static_cast<std::size_t>(GLFW_KEY_LAST) + 1> table{};  // Key::Unknown
    for (std::size_t i = 0; i < kKeyCount; ++i) {
        const int native = kNativeKeys[i];
        if (native >= 0) {
            table[static_cast<std::size_t>(native)] = static_cast<Key>(i);
        }
    }
    return table;
}();

constexpr std::array<int, kButtonCount> kNativeButtons{
    GLFW_MOUSE_BUTTON_LEFT, GLFW_MOUSE_BUTTON_RIGHT, GLFW_MOUSE_BUTTON_MIDDLE,
    GLFW_MOUSE_BUTTON_4,    GLFW_MOUSE_BUTTON_5,     GLFW_MOUSE_BUTTON_6,
    GLFW_MOUSE_BUTTON_7,    GLFW_MOUSE_BUTTON_8,
};

static_assert(kNativeButtons.size() == kButtonCount,
              "kNativeButtons must have one entry per MouseButton");

constexpr std::size_t indexOf(Key key) noexcept {
    return static_cast<std::size_t>(key);
}

constexpr std::size_t indexOf(MouseButton button) noexcept {
    return static_cast<std::size_t>(button);
}

}  // namespace

Key keyFromNative(int nativeCode) noexcept {
    if (nativeCode < 0 || nativeCode > GLFW_KEY_LAST) {
        return Key::Unknown;
    }
    return kKeysByNative[static_cast<std::size_t>(nativeCode)];
}

int nativeFromKey(Key key) noexcept {
    const std::size_t index = indexOf(key);
    if (index >= kKeyCount) {
        return GLFW_KEY_UNKNOWN;
    }
    return kNativeKeys[index];
}

MouseButton mouseButtonFromNative(int nativeCode) noexcept {
    // Dense and identical to ours, but still translated rather than cast: the
    // cast would be a silent bug the day either side gains a button.
    for (std::size_t i = 0; i < kButtonCount; ++i) {
        if (kNativeButtons[i] == nativeCode) {
            return static_cast<MouseButton>(i);
        }
    }
    return MouseButton::Count;
}

int nativeFromMouseButton(MouseButton button) noexcept {
    const std::size_t index = indexOf(button);
    if (index >= kButtonCount) {
        return -1;
    }
    return kNativeButtons[index];
}

ButtonAction buttonActionFromNative(int nativeAction) noexcept {
    switch (nativeAction) {
        case GLFW_PRESS:
            return ButtonAction::Press;
        case GLFW_REPEAT:
            return ButtonAction::Repeat;
        default:
            return ButtonAction::Release;
    }
}

int nativeFromButtonAction(ButtonAction action) noexcept {
    switch (action) {
        case ButtonAction::Press:
            return GLFW_PRESS;
        case ButtonAction::Repeat:
            return GLFW_REPEAT;
        case ButtonAction::Release:
            break;
    }
    return GLFW_RELEASE;
}

Modifiers modifiersFromNative(int nativeBits) noexcept {
    Modifiers set;
    if (nativeBits & GLFW_MOD_SHIFT) {
        set |= Modifier::Shift;
    }
    if (nativeBits & GLFW_MOD_CONTROL) {
        set |= Modifier::Control;
    }
    if (nativeBits & GLFW_MOD_ALT) {
        set |= Modifier::Alt;
    }
    if (nativeBits & GLFW_MOD_SUPER) {
        set |= Modifier::Super;
    }
    if (nativeBits & GLFW_MOD_CAPS_LOCK) {
        set |= Modifier::CapsLock;
    }
    if (nativeBits & GLFW_MOD_NUM_LOCK) {
        set |= Modifier::NumLock;
    }
    return set;
}

int nativeFromModifiers(Modifiers modifiers) noexcept {
    int bits = 0;
    if (modifiers.has(Modifier::Shift)) {
        bits |= GLFW_MOD_SHIFT;
    }
    if (modifiers.has(Modifier::Control)) {
        bits |= GLFW_MOD_CONTROL;
    }
    if (modifiers.has(Modifier::Alt)) {
        bits |= GLFW_MOD_ALT;
    }
    if (modifiers.has(Modifier::Super)) {
        bits |= GLFW_MOD_SUPER;
    }
    if (modifiers.has(Modifier::CapsLock)) {
        bits |= GLFW_MOD_CAPS_LOCK;
    }
    if (modifiers.has(Modifier::NumLock)) {
        bits |= GLFW_MOD_NUM_LOCK;
    }
    return bits;
}

void InputState::onKey(Key key, ButtonAction action, Modifiers modifiers) noexcept {
    m_modifiers = modifiers;

    const std::size_t index = indexOf(key);
    if (index >= kKeyCount) {
        return;
    }

    switch (action) {
        case ButtonAction::Press:
            m_keysDown.set(index);
            m_keysPressed.set(index);
            break;
        case ButtonAction::Release:
            m_keysDown.reset(index);
            m_keysReleased.set(index);
            break;
        case ButtonAction::Repeat:
            // Already down, and not a new press. Auto-repeat exists for text
            // entry and would otherwise fire a per-press action at the
            // keyboard's repeat rate.
            m_keysDown.set(index);
            break;
    }
}

void InputState::onMouseButton(MouseButton button, ButtonAction action,
                               Modifiers modifiers) noexcept {
    m_modifiers = modifiers;

    const std::size_t index = indexOf(button);
    if (index >= kButtonCount) {
        return;
    }

    switch (action) {
        case ButtonAction::Press:
            m_buttonsDown.set(index);
            m_buttonsPressed.set(index);
            break;
        case ButtonAction::Release:
            m_buttonsDown.reset(index);
            m_buttonsReleased.set(index);
            break;
        case ButtonAction::Repeat:
            m_buttonsDown.set(index);
            break;
    }
}

void InputState::onCursorPosition(double x, double y) noexcept {
    m_cursor = CursorPosition{x, y};
    if (!m_hasCursor) {
        // First sighting is a position, not a movement. Without this the delta
        // on that frame is the whole distance from the origin.
        m_previousCursor = m_cursor;
        m_hasCursor = true;
    }
}

void InputState::onScroll(double x, double y) noexcept {
    m_scroll.x += x;
    m_scroll.y += y;
}

void InputState::onFocusLost() noexcept {
    m_keysReleased |= m_keysDown;
    m_keysDown.reset();
    m_buttonsReleased |= m_buttonsDown;
    m_buttonsDown.reset();
    m_modifiers = Modifiers{};

    m_hasCursor = false;
    m_previousCursor = m_cursor;
}

void InputState::newFrame() noexcept {
    m_keysPressed.reset();
    m_keysReleased.reset();
    m_buttonsPressed.reset();
    m_buttonsReleased.reset();
    m_previousCursor = m_cursor;
    m_scroll = ScrollOffset{};
}

void InputState::reset() noexcept {
    *this = InputState{};
}

bool InputState::keyDown(Key key) const noexcept {
    const std::size_t index = indexOf(key);
    return index < kKeyCount && m_keysDown.test(index);
}

bool InputState::keyPressed(Key key) const noexcept {
    const std::size_t index = indexOf(key);
    return index < kKeyCount && m_keysPressed.test(index);
}

bool InputState::keyReleased(Key key) const noexcept {
    const std::size_t index = indexOf(key);
    return index < kKeyCount && m_keysReleased.test(index);
}

bool InputState::mouseButtonDown(MouseButton button) const noexcept {
    const std::size_t index = indexOf(button);
    return index < kButtonCount && m_buttonsDown.test(index);
}

bool InputState::mouseButtonPressed(MouseButton button) const noexcept {
    const std::size_t index = indexOf(button);
    return index < kButtonCount && m_buttonsPressed.test(index);
}

bool InputState::mouseButtonReleased(MouseButton button) const noexcept {
    const std::size_t index = indexOf(button);
    return index < kButtonCount && m_buttonsReleased.test(index);
}

CursorPosition InputState::cursorDelta() const noexcept {
    if (!m_hasCursor) {
        return CursorPosition{};
    }
    return CursorPosition{m_cursor.x - m_previousCursor.x,
                          m_cursor.y - m_previousCursor.y};
}

}  // namespace ysq
