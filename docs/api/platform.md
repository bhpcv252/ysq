# Platform API reference

Every public type and function in `Platform`: the windowing system, the
OpenGL context, and input. Start with
[docs/platform.md](../platform.md) for how the pieces fit together;
[src/Platform/README.md](../../src/Platform/README.md) has the design notes.
`Platform` depends only on `Core`. Not thread-safe; on macOS the windowing
system additionally requires everything here run on the main thread.

## `Platform/Platform.hpp`

The windowing system itself: reference-counted, backend-selecting, headless
where needed.

```cpp
enum class PlatformBackend { Win32, Cocoa, Wayland, X11, Null };

struct PlatformError { int code = 0; std::string message; };
struct PlatformSettings { std::optional<PlatformBackend> backend{}; };  // unset auto-selects

class Platform {
public:
    static std::optional<Platform> initialize(const PlatformSettings& settings = {},
                                               PlatformError* error = nullptr);
    static std::optional<Platform> current() noexcept;   // handle on what's already up, retains

    // copyable and movable; copying retains a reference count
    ~Platform();

    static bool initialized() noexcept;
    static std::optional<PlatformBackend> backend() noexcept;
    static bool backendAvailable(PlatformBackend backend) noexcept;

    static void pollEvents() noexcept;                 // delivers everything waiting, returns
    static void waitEvents() noexcept;                  // sleeps until an event arrives
    static void waitEventsFor(double timeoutSeconds) noexcept;
};

std::string_view toString(PlatformBackend backend) noexcept;
```

| Member | Description |
| --- | --- |
| `initialize(settings, error)` | First live handle initializes the backend; the last one to die shuts it down. Initializing twice with a *different* backend fails rather than silently keeping the first. |
| `current()` | A handle on the already-initialized system, or `nullopt`. Retains like a copy; does not initialize. |
| `backendAvailable(backend)` | Whether support was **compiled in**: a build-time fact, `false` doesn't mean "no display right now." |
| `pollEvents()` | For a simulation loop: delivers what's waiting and returns immediately. |
| `waitEvents()` / `waitEventsFor(timeout)` | For an idle editor/viewer that should cost nothing until something happens. A non-positive or non-finite timeout returns immediately rather than blocking forever. |

```cpp
const auto platform = ysq::Platform::initialize();
if (!platform) { /* headless machine, or no display */ }
```

`PlatformBackend::Null` gets its OpenGL contexts from OSMesa (a software
rasterizer loaded at runtime) rather than a real display server, which is
what makes the offscreen path testable on a machine or CI runner with no
display. `Null` is never auto-selected; it has to be asked for by name.

## `Platform/Window.hpp`

A window and its OpenGL context (or, with `visible = false`, a context on
its own: an offscreen context is just a window with `visible` off, so
there's one code path rather than two).

```cpp
struct GLVersion { int major, minor; };            // ordered, comparable
struct Extent { int width, height; };
struct ContentScale { float x = 1.0f, y = 1.0f; };  // framebuffer pixels per window coordinate

enum class CursorMode { Normal, Hidden, Disabled };  // Disabled: unbounded, for mouse-look

enum class WindowErrorCode {
    PlatformNotInitialized, InvalidSettings, CreationFailed, LoaderFailed, VersionBelowRequest
};
struct WindowError { WindowErrorCode code; std::string message; GLVersion loaded{}; };

inline constexpr GLVersion kMaxSupportedGL{4, 6};   // ceiling this build's GLAD was generated for

struct ContextSettings {
    int versionMajor = 4, versionMinor = 1;   // 4.1: the macOS ceiling
    bool coreProfile = true, forwardCompatible = true, debugContext = false;
    int depthBits = 24, stencilBits = 8, samples = 0;   // samples: MSAA, 0 disables
};

ContextSettings effectiveContext(const ContextSettings& requested, PlatformBackend backend) noexcept;

struct WindowSettings {
    std::string title = "YSQ";
    int width = 1280, height = 720;
    bool visible = true, resizable = true, decorated = true, vsync = true;
    ContextSettings context{};
};
```

```cpp
class Window {
public:
    static std::optional<Window> create(const WindowSettings&, WindowError* error = nullptr);
    static std::optional<Window> createOffscreen(int width, int height,
                                                  const ContextSettings& context = {},
                                                  WindowError* error = nullptr);
    // move-only; ~Window() destroys

    void makeContextCurrent();          // reloads GL entry points if the context changed
    static void clearCurrentContext();
    bool isContextCurrent() const;

    void swapBuffers();
    void setVSync(bool enabled);        // makes the context current first
    bool vsync() const noexcept;

    bool shouldClose() const;
    void setShouldClose(bool close);

    void setTitle(std::string_view title);
    void setCursorMode(CursorMode mode);
    CursorMode cursorMode() const noexcept;

    void show(); void hide(); bool visible() const;

    Extent size() const;                 // window coordinates, not pixels on HiDPI
    Extent framebufferSize() const;      // pixels; what glViewport wants
    ContentScale contentScale() const;
    GLVersion glVersion() const noexcept;              // what the context actually reports
    const ContextSettings& contextSettings() const noexcept;  // what was asked, post-effectiveContext

    InputState& input() noexcept;
    void setFramebufferSizeCallback(std::function<void(Extent)> callback);
    void setCloseCallback(std::function<void()> callback);  // clear shouldClose() in the handler to refuse

    GLFWwindow* nativeHandle() const noexcept;   // escape hatch for ImGui's GLFW backend
};
```

| Member | Description |
| --- | --- |
| `create` / `createOffscreen` | `createOffscreen` is `create` with `visible = false` under the hood; on the `Null` backend every window is offscreen regardless. |
| `makeContextCurrent()` | GLAD's entry points are **process-wide globals** describing whichever context was made current last. Reloading costs a few hundred symbol lookups; switch contexts at setup time, not per frame. |
| `setFramebufferSizeCallback` / `setCloseCallback` | Fired during `Platform::pollEvents()`. The close flag is already set when the close callback fires; clear it there to refuse the close. |
| `input()` | Per-window `InputState`; see below. With more than one window, call `newFrame()` on every window before the single shared `pollEvents()`. |

```cpp
const auto platform = ysq::Platform::initialize();
auto window = ysq::Window::create({.title = "YSQ"});
while (!window->shouldClose()) {
    window->input().newFrame();
    ysq::Platform::pollEvents();
    // draw
    window->swapBuffers();
}
```

`ContextSettings::versionMajor/Minor` defaults to 4.1 because that's the
macOS ceiling; `Compute`'s OpenGL backend asks for 4.3 (where compute
shaders start) and gets a clean `nullptr`/failure where that isn't
available, rather than a half-working context.

## `Platform/Input.hpp`

Keyboard and mouse state for one window, sampled per frame. Physical keys
(US layout positions, not characters: `Key::Q` names where Q sits, not what
the active layout prints there).

```cpp
enum class Key : std::uint16_t { Unknown = 0, Space, /* ... */, Z, /* ... */, Menu, Count };
enum class MouseButton : std::uint8_t { Left = 0, Right, Middle, Button4, /* ... */, Count };
enum class ButtonAction : std::uint8_t { Release, Press, Repeat };
enum class Modifier : std::uint8_t { Shift = 1<<0, Control = 1<<1, Alt = 1<<2, Super = 1<<3, CapsLock = 1<<4, NumLock = 1<<5 };

class Modifiers {
public:
    constexpr Modifiers() noexcept = default;
    constexpr Modifiers(Modifier modifier) noexcept;         // implicit by design
    static constexpr Modifiers fromBits(std::uint8_t bits) noexcept;   // undefined bits dropped

    constexpr std::uint8_t bits() const noexcept;
    constexpr bool none() const noexcept;
    constexpr bool has(Modifier modifier) const noexcept;
    constexpr bool contains(Modifiers other) const noexcept;   // empty set contained in anything
    // |= &= | & ==
};

struct CursorPosition { double x = 0.0, y = 0.0; };
struct ScrollOffset { double x = 0.0, y = 0.0; };

Key keyFromNative(int nativeCode) noexcept;
int nativeFromKey(Key key) noexcept;
MouseButton mouseButtonFromNative(int nativeCode) noexcept;
int nativeFromMouseButton(MouseButton button) noexcept;
ButtonAction buttonActionFromNative(int nativeAction) noexcept;
int nativeFromButtonAction(ButtonAction action) noexcept;
Modifiers modifiersFromNative(int nativeBits) noexcept;
int nativeFromModifiers(Modifiers modifiers) noexcept;
```

| Function | Notes |
| --- | --- |
| `keyFromNative`/`nativeFromKey` | Unrecognized native code -> `Key::Unknown`; `Key::Unknown` round-trips back to the native "unknown" code. Every other key round-trips. |
| `mouseButtonFromNative`/reverse | No "unknown" mouse button exists: an unrecognized code yields `MouseButton::Count` (native reverse yields `-1`), neither of which matches any query, so it can't be mistaken for `Left`. |
| `buttonActionFromNative` | An unrecognized action is reported as `Release`, the safe reading, since an uninterpretable state must not read as stuck-down. |

```cpp
class InputState {
public:
    void onKey(Key, ButtonAction, Modifiers) noexcept;
    void onMouseButton(MouseButton, ButtonAction, Modifiers) noexcept;
    void onCursorPosition(double x, double y) noexcept;
    void onScroll(double x, double y) noexcept;
    void onFocusLost() noexcept;    // releases everything held; drops the cursor baseline

    void newFrame() noexcept;        // call once per frame, before polling
    void reset() noexcept;

    bool keyDown(Key) const noexcept;       // held right now
    bool keyPressed(Key) const noexcept;    // went down this frame
    bool keyReleased(Key) const noexcept;   // came up this frame

    bool mouseButtonDown/Pressed/Released(MouseButton) const noexcept;

    Modifiers modifiers() const noexcept;    // as of the most recent key/button event
    CursorPosition cursor() const noexcept;
    CursorPosition cursorDelta() const noexcept;   // movement since last newFrame(); zero until first seen
    bool hasCursor() const noexcept;               // true once a cursor position has ever arrived
    ScrollOffset scrollDelta() const noexcept;      // accumulated this frame (summed, not replaced)
};
```

```cpp
input.newFrame();          // clears edges, snapshots the cursor
ysq::Platform::pollEvents();    // callbacks land here
if (input.keyPressed(ysq::Key::Space)) { jump(); }
```

`keyDown`/`keyPressed`/`keyReleased` genuinely differ: a key pressed **and**
released within one frame reports both edges while `keyDown` is false for
that frame, so a fast tap is never missed. Auto-repeat sets neither edge
(the key was already down); treating repeat as a fresh press would fire a
once-per-press action at the keyboard's repeat rate instead.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/platform)
and let us know.
