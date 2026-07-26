# Core

Foundational services with no physics or math content: logging, timing, identity,
events, configuration. Every other module may depend on `Core`; `Core` depends on
nothing in the engine.

**Target:** `ysq::Core` (static)
**Depends on:** nothing in YSQ. Externally, spdlog (once `Logger` lands).

## Contents

| Header             | Purpose                                       |
| ------------------ | --------------------------------------------- |
| `Core/Version.hpp` | Engine version, generated from the CMake project version |

Planned, per the project layout: `Logger.hpp` (spdlog facade), `Timer.hpp`,
`Clock.hpp` (simulation time vs. wall-clock), `UUID.hpp`, `Event.hpp`,
`Config.hpp`.

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
