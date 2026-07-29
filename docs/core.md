# Core: infrastructure

The plumbing every simulation needs regardless of what it's simulating: when
did this step happen, what happened, what's configured. No physics or math
content at all.

## The idea

Every other module in YSQ is about a specific kind of thing: vectors, forces,
metrics, pixels. `Core` isn't about anything domain-specific; it's the
handful of concerns that show up in literally any running program and are
worth getting right once instead of reinventing per module: writing out what
happened for later debugging (logging), knowing how much simulated time has
passed versus how much real time has passed (timing), giving something a
unique name (identity), letting parts of the program that don't know about
each other still communicate (events), and reading settings from a file
(configuration). `Core` depends on nothing else in the engine; everything
else may depend on it.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Core/Logger.hpp` | Logging: levels, a console/file/stream sink, no spdlog in the header |
| `Core/Timer.hpp` | A wall-clock stopwatch |
| `Core/Clock.hpp` | Simulation time: fixed steps, time scale, pause |
| `Core/UUID.hpp` | Random, unique identifiers |
| `Core/Event.hpp` | A type-keyed event bus for anything copyable |
| `Core/Config.hpp` | Key/value settings, read from a small INI-like text format |
| `Core/Version.hpp` | The engine's own version, generated from the CMake project version |

The one to understand first is `Clock`, because its shape is the shape of
every YSQ simulation's main loop.

## Using it

`Clock` never reads a wall clock itself; the host loop feeds it real elapsed
time, and it decides how many fixed-size simulation steps are due:

```cpp
#include <Core/Clock.hpp>
#include <Core/Timer.hpp>

clock.advance(frame.lap().count());
while (clock.consumeStep()) {
    world.integrate(clock.fixedStep());
    // clock.simulationTime() is the time at the end of this step
}
renderer.draw(clock.alpha());  // the leftover fraction of a step, for smoothing
```

Fixed steps rather than a variable one, because a physics integrator has to
be reproducible: feed it the same sequence of step sizes and it must produce
the same trajectory every time, or a test that checks energy conservation
means nothing. `timeScale` speeds up or slows down simulation time relative
to real time, `pause()` freezes it, `stepOnce()` is the single-step
debugging control.

Logging and configuration are the other two you'll reach for immediately in
any new `Applications/` program:

```cpp
#include <Core/Logger.hpp>
#include <Core/Config.hpp>

ysq::LogSettings settings;
settings.level = ysq::LogLevel::Debug;
ysq::Logger::init(settings);
ysq::logging::info("{} bodies at t={}", count, t);

const std::optional<ysq::Config> config = ysq::Config::load("sim.ini", &error);
const double timestep = config->get<double>("physics.timestep", 1e-3);
```

`Config::load` and every `get<T>` are total: a missing file, a missing key,
or a value that won't parse as `T` all fall back to the default you gave
rather than throwing, so a malformed settings file degrades to defaults
instead of crashing your simulation on startup.

## Go deeper

[docs/api/core.md](api/core.md) has every signature: parameters, return
values, and the exact edge-case behavior (what an invalid `advance()` delta
does, what `Config::get` falls back to, and so on).

[src/Core/README.md](../src/Core/README.md) has the full interface,
including `UUID` and `Event` (useful once a simulation has enough moving
parts that things need to be told apart or need to talk to each other
without being wired together directly), the exact `Config` file format and
its round-tripping guarantees, and why `Logger.hpp` never includes spdlog
even though it's built on it.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+core)
and let us know.
