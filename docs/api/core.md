# Core API reference

Every public class and function in `Core`, with signatures, parameters and a
usage snippet. Start with [docs/core.md](../core.md) if you haven't met the
module yet; that page explains why each piece exists. For the reasoning
behind a specific design choice, [src/Core/README.md](../../src/Core/README.md)
is the authoritative source.

Nothing here is thread-safe unless stated otherwise. `Logger` is; everything
else assumes a single owning thread, which for `Clock` and `EventBus` is the
simulation loop.

## `Core/Version.hpp`

Engine version, generated from the top-level CMake project version.

```cpp
struct Version {
    int major;
    int minor;
    int patch;
};

constexpr Version version() noexcept;
std::string versionString();  // "major.minor.patch"
```

```cpp
#include <Core/Version.hpp>

ysq::version();        // Version{0, 1, 0}
ysq::versionString();  // "0.1.0"
```

The header is generated (`build/generated/Core/Version.hpp`) but included the
same way as any other. Never edit it directly; edit `Version.hpp.in`.

## `Core/Logger.hpp`

Logging facade over spdlog. spdlog is not named in the header, so nothing
that includes this compiles spdlog or depends on the backend staying spdlog.

```cpp
enum class LogLevel { Trace, Debug, Info, Warn, Error, Critical, Off };

struct LogSettings {
    std::string name = "ysq";
    LogLevel level = LogLevel::Info;
    std::string pattern = "[%T.%e] [%n] [%l] %v";
    bool console = true;
    std::optional<std::filesystem::path> file{};
    std::ostream* stream = nullptr;  // not owned, must outlive the logger
};

class Logger {
public:
    static void init(const LogSettings& settings = {});
    static void shutdown();
    static void setLevel(LogLevel level);
    static LogLevel level() noexcept;
    static bool enabled(LogLevel level) noexcept;
};

namespace logging {
    template <class... Args> void trace(std::format_string<Args...> fmt, const Args&... args);
    template <class... Args> void debug(std::format_string<Args...> fmt, const Args&... args);
    template <class... Args> void info(std::format_string<Args...> fmt, const Args&... args);
    template <class... Args> void warn(std::format_string<Args...> fmt, const Args&... args);
    template <class... Args> void error(std::format_string<Args...> fmt, const Args&... args);
    template <class... Args> void critical(std::format_string<Args...> fmt, const Args&... args);
}
```

| Member | Description |
| --- | --- |
| `Logger::init(settings)` | Optional. Configures sinks and level. If skipped, the first log call initializes defaults (`Info`, console on). |
| `Logger::shutdown()` | Drops the logger and its sinks. Closes any file sink; a file sink kept alive past this holds the file open and blocks rotation/deletion on Windows. |
| `Logger::setLevel(level)` / `level()` | Change or read the active level at runtime. |
| `Logger::enabled(level)` | Cheap check against an atomic mirror of the level, used internally so a filtered call formats nothing. |
| `logging::trace/debug/info/warn/error/critical(fmt, args...)` | Emit a line via `std::format` syntax. A filtered-out call costs one atomic load and returns; it never reaches spdlog. |

```cpp
ysq::LogSettings settings;
settings.level = ysq::LogLevel::Debug;
settings.file = "ysq.log";
ysq::Logger::init(settings);

ysq::logging::info("{} bodies at t={}", count, t);
ysq::Logger::setLevel(ysq::LogLevel::Warn);
ysq::Logger::shutdown();
```

`LogSettings::stream` is a non-owning pointer; whatever it points at must
outlive the logger, since `shutdown()` flushes into it. This is how tests
capture log output without a file.

## `Core/Timer.hpp`

Wall-clock stopwatch, separate from `Clock`'s simulation time.

```cpp
template <class ClockT = std::chrono::steady_clock>
class BasicTimer {
public:
    using Seconds = std::chrono::duration<double>;

    BasicTimer();
    void reset();
    void stop();
    void resume();
    bool running() const noexcept;
    Seconds elapsed() const;
    double elapsedSeconds() const;
    Seconds lap();  // elapsed since the last lap, then restart
};

using Timer = BasicTimer<std::chrono::steady_clock>;

class ScopedTimer {
public:
    explicit ScopedTimer(std::string label, LogLevel level = LogLevel::Debug);
    // logs "{label} took {ms} ms" at `level` on destruction
};
```

| Member | Description |
| --- | --- |
| `reset()` | Restart from zero, running. |
| `stop()` / `resume()` | Pause/unpause; `elapsed()` freezes while stopped. |
| `elapsed()` / `elapsedSeconds()` | Total elapsed time, including the current run if still running. |
| `lap()` | Time since the last `lap()` or `reset()`, then resets. The frame-loop idiom. |

```cpp
ysq::Timer frame;
// ... later, once per frame:
clock.advance(frame.lap().count());
```

`BasicTimer` is templated on the clock (`Timer` is the `steady_clock` alias)
so a test can drive it with a manual clock type and assert exact values
instead of sleeping. `ScopedTimer` logs the lifetime of its scope; construct
one at the top of a block to time it.

## `Core/Clock.hpp`

Fixed-step simulation time. Never reads a wall clock itself; the host loop
feeds it real elapsed time and it hands back whole fixed steps to run, so
integration stays deterministic: the same sequence of deltas always produces
the same trajectory.

```cpp
class Clock {
public:
    struct Settings {
        double fixedStep = 1.0 / 60.0;   // simulation seconds per step
        double timeScale = 1.0;          // simulation seconds per real second
        int maxStepsPerAdvance = 8;       // clamp on a single advance()
        bool paused = false;
    };

    Clock();
    explicit Clock(Settings settings);

    int advance(double realDeltaSeconds);
    bool consumeStep();
    int stepsDue() const noexcept;
    void stepOnce();

    void pause() noexcept;
    void resume() noexcept;
    bool paused() const noexcept;

    void setTimeScale(double scale) noexcept;
    double timeScale() const noexcept;
    void setFixedStep(double seconds) noexcept;
    double fixedStep() const noexcept;

    void reset() noexcept;

    double simulationTime() const noexcept;
    double realTime() const noexcept;
    std::uint64_t stepCount() const noexcept;
    double alpha() const noexcept;  // unconsumed fraction of a step, [0, 1)
};
```

| Member | Description |
| --- | --- |
| `advance(realDeltaSeconds)` | Feed one frame of real elapsed time. Returns how many fixed steps are now due. Negative/non-finite deltas are ignored and leave the due count untouched. While paused, real time still accrues but the due count doesn't change. Otherwise the due count is recomputed from scratch, so steps left unconsumed from the previous `advance()` are dropped. |
| `consumeStep()` | Takes one due step, advancing `simulationTime()` by `fixedStep()`. Returns `false` once the steps from the last `advance()` are spent. |
| `stepOnce()` | Advances one step regardless of pause state or steps due. The single-step debugging control. |
| `pause()` / `resume()` | Freezes/unfreezes simulation time; real time keeps accruing while paused. |
| `setTimeScale(scale)` | Ignored if non-finite or negative. |
| `setFixedStep(seconds)` | Ignored if non-finite or non-positive. |
| `reset()` | Zeroes simulation time, real time, step count and the accumulator. Settings are kept. |
| `alpha()` | Leftover fraction of a step not yet consumed, in `[0, 1)`. The blend factor for interpolating a render between the last two simulation states. |

```cpp
ysq::Clock clock;
ysq::Timer frame;

while (running) {
    clock.advance(frame.lap().count());
    while (clock.consumeStep()) {
        world.integrate(clock.fixedStep());
        // clock.simulationTime() is the time at the end of this step
    }
    renderer.draw(clock.alpha());
}
```

`maxStepsPerAdvance` bounds how many steps one `advance()` can ask for; when
it trips, the backlog beyond the clamp is discarded rather than carried, so a
long stall makes simulation time fall behind real time instead of making the
next stall worse. Times are `double` seconds; there is no `Units::Time`
integration, since `Core` sits beside `Units`, not above it.

## `Core/UUID.hpp`

Random (RFC 4122 v4) identifiers.

```cpp
class UUID {
public:
    using Bytes = std::array<std::uint8_t, 16>;

    constexpr UUID() noexcept;                       // nil UUID
    explicit constexpr UUID(const Bytes& bytes) noexcept;

    static UUID generate();
    static std::optional<UUID> parse(std::string_view text);

    std::string toString() const;         // lowercase 8-4-4-4-12
    constexpr const Bytes& bytes() const noexcept;
    constexpr bool isNil() const noexcept;

    friend bool operator==(const UUID&, const UUID&) = default;
    friend auto operator<=>(const UUID&, const UUID&) = default;
};

class UuidGenerator {
public:
    explicit UuidGenerator(std::uint64_t seed);
    UUID operator()();
};
```

| Member | Description |
| --- | --- |
| `UUID::generate()` | Draws from a thread-local, non-reproducible RNG. |
| `UUID::parse(text)` | Strict: exactly 8-4-4-4-12 hex, either case, nothing else. Returns `std::nullopt` otherwise. |
| `toString()` | Lowercase canonical form. |
| `isNil()` | True for a default-constructed `UUID`. |
| `UuidGenerator{seed}` | Explicitly seeded generator; the sequence depends only on the seed, so a run using it replays identically. |

```cpp
const ysq::UUID id = ysq::UUID::generate();
id.toString();                          // "1b4e28ba-2fa1-4d9b-b5f9-2ff3a2c1e4f7"
ysq::UUID::parse(text);                 // std::optional<UUID>

ysq::UuidGenerator gen{42};
const ysq::UUID reproducible = gen();   // same for every run seeded with 42
```

`UUID` is comparable, ordered and hashable (`std::hash<ysq::UUID>` is
specialized), so it drops directly into `std::unordered_map`/`std::set`.

## `Core/Event.hpp`

Synchronous, type-keyed event bus. Any copyable type is an event; there's no
base class to inherit and no enum to extend.

```cpp
class Subscription {
public:
    Subscription() noexcept;
    void unsubscribe() noexcept;
    void release() noexcept;    // detach without unsubscribing
    bool active() const noexcept;
};

class EventBus {
public:
    EventBus();

    template <class E>
    Subscription subscribe(std::function<void(const E&)> handler);

    template <class E>
    void publish(const E& event);      // delivered now

    template <class E>
    void enqueue(E event);             // delivered on next dispatchQueued()

    void dispatchQueued();
    template <class E> std::size_t subscriberCount() const;
    std::size_t queuedCount() const noexcept;
    void clear();
};
```

| Member | Description |
| --- | --- |
| `subscribe<E>(handler)` | Registers a handler for event type `E`. Returns a move-only `Subscription`. |
| `publish<E>(event)` | Delivers to every current subscriber of `E`, in subscription order, synchronously. |
| `enqueue<E>(event)` | Stores a copy; delivered on the next `dispatchQueued()` rather than now. |
| `dispatchQueued()` | Drains everything queued so far. Events enqueued by a handler during this call land in the *next* drain, not this one. |
| `Subscription::unsubscribe()` | Also happens automatically on destruction. |
| `Subscription::release()` | Detaches the handle without unsubscribing; the handler then lives as long as the bus. |
| `clear()` | Drops every handler and queued event. **Not valid from inside a handler**: it destroys the list a dispatch is walking. |

```cpp
struct StepCompleted { std::uint64_t index; double simulationTime; };

ysq::EventBus bus;
const ysq::Subscription handle =
    bus.subscribe<StepCompleted>([](const StepCompleted& e) {
        ysq::logging::debug("step {} at t={}", e.index, e.simulationTime);
    });

bus.publish(StepCompleted{n, t});
bus.enqueue(StepCompleted{n, t});  // delivered next dispatchQueued()
bus.dispatchQueued();
```

A `Subscription` that outlives its `EventBus` goes inert (it holds a weak
reference) rather than writing through a dangling pointer. A handler added
during a dispatch does not receive the event that triggered its own
registration. Each `publish` costs a hash lookup plus one indirect call per
handler, which suits frame- or step-level events, not per-body traffic; a
hot loop should call its collaborator directly instead.

## `Core/Config.hpp`

Flat key/value configuration, dotted keys, typed access, an INI-like text
form.

```cpp
struct ConfigError {
    std::size_t line = 0;   // 1-based; 0 if not tied to a line
    std::string message;
};

class Config {
public:
    template <class T> std::optional<T> tryGet(std::string_view key) const;
    template <class T> T get(std::string_view key, const T& fallback) const;
    std::string get(std::string_view key, const char* fallback) const;

    template <class T> bool set(std::string_view key, const T& value);
    bool set(std::string_view key, std::string_view value);

    bool has(std::string_view key) const;
    bool erase(std::string_view key);
    void clear();
    std::size_t size() const noexcept;
    std::vector<std::string> keys() const;      // sorted

    void merge(const Config& overrides);          // overrides wins
    std::string toString() const;

    static constexpr std::uintmax_t kDefaultMaxFileBytes = 16u * 1024u * 1024u;
    static std::optional<Config> parse(std::string_view text, ConfigError* error = nullptr);
    static std::optional<Config> load(const std::filesystem::path& path,
                                       ConfigError* error = nullptr,
                                       std::uintmax_t maxBytes = kDefaultMaxFileBytes);
    bool save(const std::filesystem::path& path) const;
};
```

Supported `T`: `bool`, any integral type except `bool`/character types,
floating-point, and `std::string`.

| Member | Description |
| --- | --- |
| `tryGet<T>(key)` | `std::nullopt` if the key is missing or the value won't parse/fit as `T`. Every call is a fresh map lookup and parse; read once into whatever owns the value. |
| `get<T>(key, fallback)` | `tryGet<T>(key).value_or(fallback)`. Never throws. |
| `set<T>(key, value)` | Returns `false` if the key is malformed or the value can't survive a round trip (e.g. contains a newline); otherwise stores it. |
| `load(path, error, maxBytes)` | Refuses a file over `maxBytes` (default 16 MiB) rather than allocating whatever it's pointed at. Parse failures report a line number through `ConfigError`. |
| `merge(overrides)` | Keys present in `overrides` replace this config's; keys only in `this` are kept. |
| `toString()` | A fixed point: parsing its output and emitting again yields the same bytes. |

```ini
# comment
timeScale = 1.0

[physics]
integrator = rk4
timestep   = 0.001
```

```cpp
ysq::ConfigError error;
const std::optional<ysq::Config> config = ysq::Config::load("sim.ini", &error);
const double timestep = config->get<double>("physics.timestep", 1e-3);
const std::string integrator = config->get("physics.integrator", "rk4");
```

`[section]` headers are pure sugar: `[physics]` followed by `timestep = ...`
is the same key as `physics.timestep` written flat. Reads are total: a
missing file, a missing key, or an unparsable value all fall back to the
default you gave rather than throwing, so a malformed settings file degrades
to defaults instead of crashing startup.

## `Core/Csv.hpp`

A CSV table, header row plus typed data rows, for data a consumer downloaded
or hand-curated. Round-tripping is not a goal the way it is for `Config`.

```cpp
struct CsvError {
    std::size_t line = 0;   // 1-based; 0 if not tied to a line
    std::string message;
};

class Csv {
public:
    class Row {
    public:
        template <class T> std::optional<T> tryGet(std::string_view column) const;
        template <class T> T get(std::string_view column, const T& fallback) const;
        std::string get(std::string_view column, const char* fallback) const;
        bool has(std::string_view column) const;
        std::size_t lineNumber() const noexcept;   // 1-based source line
    };

    const std::vector<std::string>& columns() const noexcept;
    std::size_t columnCount() const noexcept;
    std::size_t rowCount() const noexcept;
    bool hasColumn(std::string_view column) const;

    Row row(std::size_t index) const;
    RowIterator begin() const noexcept;   // range-for support
    RowIterator end() const noexcept;

    static constexpr std::uintmax_t kDefaultMaxFileBytes = 64u * 1024u * 1024u;
    static std::optional<Csv> parse(std::string_view text, CsvError* error = nullptr);
    static std::optional<Csv> load(const std::filesystem::path& path,
                                    CsvError* error = nullptr,
                                    std::uintmax_t maxBytes = kDefaultMaxFileBytes);
};
```

Supported `T`: the same set `Config` supports -- `bool`, any integral type
except `bool`/character types, floating-point, and `std::string`.

| Member | Description |
| --- | --- |
| `Row::tryGet<T>(column)` | `std::nullopt` if the column is missing or the field won't parse/fit as `T`. |
| `Row::get<T>(column, fallback)` | `tryGet<T>(column).value_or(fallback)`. Never throws. |
| `Row::lineNumber()` | This row's own 1-based source line, for a caller's own error message about that row's data. |
| `parse(text, error)` / `load(path, error, maxBytes)` | Parse failures (a ragged row, an unterminated quote, a duplicate header column, and so on) report a line number through `CsvError`. `load` refuses a file over `maxBytes` (default 64 MiB). |
| `begin()` / `end()` | Row objects are views into the `Csv` that produced them and do not outlive it. |

```
# source: JPL SSD, epoch 2000-01-01.5 TDB
name, mass_kg, semi_major_axis_km
Io, 8.9319e22, 421800
"Europa, Jupiter II", 4.7998e22, 671100
```

```cpp
ysq::CsvError error;
const std::optional<ysq::Csv> table = ysq::Csv::load("moons.csv", &error);
for (const ysq::Csv::Row& row : *table) {
    const std::string name = row.get<std::string>("name", "");
    const double massKg = row.get<double>("mass_kg", 0.0);
}
```

The text form is RFC 4180 with two documented extensions: a line whose first
non-whitespace character is `#` is a comment (skipped, matching `Config`'s own
convention), and an unquoted field is trimmed of leading/trailing whitespace
while a quoted one is preserved exactly. A quoted field may contain a literal
comma or newline (`""` embeds a literal quote); a `"` outside a quoted field,
or content after a quoted field closes and before the next comma, is a parse
error naming its line. Every data row must have exactly as many fields as the
header has columns, and header names must be non-empty and unique -- both are
parse errors, not silently padded, truncated, or overwritten rows.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/core)
and let us know.
