# Core

Foundational services with no physics or math content: logging, timing, identity,
events, configuration. Every other module may depend on `Core`; `Core` depends on
nothing in the engine.

**Target:** `ysq::Core` (static)
**Depends on:** nothing in YSQ. Externally, spdlog, linked `PRIVATE`.

## Contents

| Header             | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `Core/Version.hpp` | Engine version, generated from the CMake project version |
| `Core/Logger.hpp`  | Logging facade over spdlog                               |
| `Core/Timer.hpp`   | Wall-clock stopwatch                                     |
| `Core/Clock.hpp`   | Simulation time: fixed steps, time scale, pause          |
| `Core/UUID.hpp`    | RFC 4122 version 4 identifiers                           |
| `Core/Event.hpp`   | Type-keyed event bus                                     |
| `Core/Config.hpp`  | Key/value configuration with an INI text form            |

Nothing here is thread-safe unless it says so. `Logger` is; the rest assume a
single owning thread, which for `Clock` and `EventBus` is the simulation loop.

## Version

`Version.hpp` is generated from `Version.hpp.in` by `configure_file`, so the
version lives in exactly one place: `project(YSQ VERSION ...)` in the top-level
`CMakeLists.txt`. The generated header lands in `build/generated/Core/Version.hpp`
and is included the same way as any checked-in header:

```cpp
#include <Core/Version.hpp>

ysq::version();        // Version{0, 1, 0}
ysq::versionString();  // "0.1.0"
```

Do not edit the generated file. Edit `Version.hpp.in`.

## Logger

```cpp
ysq::LogSettings settings;
settings.level = ysq::LogLevel::Debug;
settings.file = "ysq.log";
ysq::Logger::init(settings);

ysq::log::info("{} bodies at t={}", count, t);
ysq::Logger::setLevel(ysq::LogLevel::Warn);
```

Levels are `Trace`, `Debug`, `Info`, `Warn`, `Error`, `Critical`, `Off`. The sinks
are a colour console, a file, and an arbitrary `std::ostream`, each independent.
`init()` is optional: the first log call sets up the defaults.

`shutdown()` drops the logger and its sinks, which is what closes the log file.
That matters more than it sounds: a file sink kept alive past `shutdown()` holds
the file open, and Windows will not let anything rotate or delete it.

Keeping that guarantee costs a short lock around a `shared_ptr` copy on each
emitted line. A lock-free read would need the pointer to stay valid for a writer
that loaded it just before a swap, which without `std::atomic<std::shared_ptr>`
(unimplemented in libc++) means retaining every logger forever and losing the
release-on-shutdown guarantee. spdlog's sink already takes a mutex per line, so
the second uncontended acquire is the cheaper trade.

**spdlog does not appear in the header.** `Logger.hpp` includes `<format>` and
nothing else from outside the standard library, so no consumer of `Core` compiles
spdlog headers and nothing depends on the backend staying spdlog. Formatting
happens on our side and reaches spdlog as a finished string. That costs one
indirection per line actually emitted; the level check happens first, inline,
against an atomic mirror of the active level, so a filtered call formats nothing
and never leaves the header.

`LogSettings::stream` does not own what it points at. It has to outlive the
logger, because `shutdown()` flushes into it.

## Timer and Clock

Two different things, deliberately separated.

`Timer` is a wall-clock stopwatch: `elapsed()`, `lap()`, `stop()`, `resume()`. It
is templated on its clock (`BasicTimer<ClockT>`, with `Timer` as the
`steady_clock` alias) so tests can drive it with a manual clock and assert exact
values rather than sleeping. `ScopedTimer` logs how long its scope took.

`Clock` is simulation time, and it never reads a wall clock. The host loop feeds
it real elapsed time; it decides how many fixed steps are due.

```cpp
clock.advance(frame.lap().count());
while (clock.consumeStep()) {
    world.integrate(clock.fixedStep());
    // simulationTime() is the time at the end of this step
}
renderer.draw(clock.alpha());  // alpha is the leftover fraction of a step
```

Deciding and executing are split so the loop body can see the simulation time of
the step it is running, not only of the frame. A normal `advance()` recomputes the
due count, so steps left unconsumed are dropped; an `advance()` that refuses its
delta (negative, NaN, infinite) or one made while paused leaves the pending count
alone, so a bad timing sample cannot swallow a frame of simulation.

Fixed steps rather than a variable `dt` because integration has to be
reproducible: the same sequence of deltas must produce the same trajectory, or a
conservation test means nothing. Feeding deltas in rather than reading a clock is
the other half of that, and it is why the clock's tests need no sleeping.

`timeScale` scales simulation seconds per real second, `pause()` freezes
simulation time while real time keeps accruing, and `stepOnce()` is the
single-step debugging control. `maxStepsPerAdvance` bounds what one frame can ask
for; when it trips, the backlog is discarded rather than carried, so simulation
time falls behind real time instead of each stall making the next one worse.

Times are `double` seconds. `Units::Time` will layer on top later; `Core` sits
beside `Units`, not above it, so it cannot use it.

## UUID

```cpp
const ysq::UUID id = ysq::UUID::generate();
id.toString();             // "1b4e28ba-2fa1-4d9b-b5f9-2ff3a2c1e4f7"
ysq::UUID::parse(text);    // std::optional<UUID>, strict 8-4-4-4-12
```

Version 4, random. Comparable, ordered, hashable. `generate()` draws from a
thread-local `mt19937_64` seeded from `random_device` mixed with a clock reading
and a counter, because `random_device` is deterministic on some MinGW builds.
`UuidGenerator{seed}` is the explicitly seeded form, for runs that have to replay
identically.

## Event

Any copyable type is an event. There is no base class to inherit and no enum to
extend; the bus keys handlers by `std::type_index`.

```cpp
struct StepCompleted { std::uint64_t index; double simulationTime; };

const ysq::Subscription handle =
    bus.subscribe<StepCompleted>([](const StepCompleted& e) { ... });

bus.publish(StepCompleted{n, t});   // delivered now
bus.enqueue(StepCompleted{n, t});   // delivered on the next dispatchQueued()
```

Each publish costs a hash lookup on the event type plus one indirect call per
handler, which suits frame- and step-level events and not per-body traffic. A hot
loop should call its collaborator directly and leave the bus for events something
else genuinely needs to observe.

`Subscription` is a move-only handle that unsubscribes when destroyed, so a
handler cannot outlive what it captured by accident; `release()` detaches it for
handlers meant to live as long as the bus. It holds a weak reference to the bus,
so a subscription that outlives its bus goes inert rather than writing through a
dangling pointer.

Handlers may subscribe, unsubscribe and publish from inside a dispatch. Removals
are deferred: a slot is marked dead and swept later, so the loop that is running
is never disturbed and the `noexcept` unsubscribe path never has to move a
`std::function`. The handler itself is cleared at once, so whatever it captured is
released when the subscription ends rather than when the slot is reclaimed.

A handler added during a dispatch does not receive the event that created it.
`clear()` is the one thing that is not valid from inside a handler, because it
destroys the list being walked.

## Config

Flat key/value store with typed access. Keys are dotted paths; a `[section]`
header in the text form is just a prefix.

```ini
# comment
timeScale = 1.0

[physics]
integrator = rk4
timestep   = 0.001
```

```cpp
const std::optional<ysq::Config> config = ysq::Config::load("sim.ini", &error);
config->get<double>("physics.timestep", 1e-3);   // fallback if missing or unparsable
config->get<std::string>("physics.integrator", "rk4");
```

Supported types are `bool`, integral, floating-point and `std::string`. Reads
never throw: a missing key, a value that will not parse as `T`, and a value that
will not fit in `T` all yield the fallback. The range check covers floating-point
as well as integers, so reading a double-sized value as `float` reports failure
rather than quietly returning `inf`; an infinity stored as an infinity still reads
back as one. `tryGet<T>` returns `std::optional` where the difference matters, and
`get(key, "default")` needs no explicit type. Parse failures report a line number
through `ConfigError`.

Every read is a map lookup and a fresh parse. Read what a run needs once, into
whatever owns it. `load()` refuses a file larger than `kDefaultMaxFileBytes`
(16 MiB, overridable per call), since this is the one place `Core` reads a file it
did not write.

Round-tripping is total rather than nearly total, which costs a few rules:

- Valid key characters are alphanumerics, `_`, `-` and `.`. Segments must be
  non-empty, so no leading, trailing or doubled dots.
- Values are trimmed when stored, so surrounding whitespace is not preserved, and
  a value cannot contain a newline. `set` returns `false` rather than storing
  something that would not survive a write followed by a read.
- Comments start a line. A `#` inside a value is data.
- Booleans write as `true` or `false`, and read `true`, `false`, `1` or `0` in any
  case.
- Doubles are written with `std::format`, whose default is the shortest form that
  reads back bit-identical, and parsed with `std::from_chars`. Both are
  locale-independent, so the file format cannot start depending on the host's
  `LC_NUMERIC`. Denormals and infinities survive.

`toString()` is a fixed point: parsing its output and emitting again gives the
same bytes, so a config file does not churn every time something rewrites it.

There is no JSON or TOML dependency. The format is small enough to own, and the
dependency table in the root `README.md` stays as short as it is.
