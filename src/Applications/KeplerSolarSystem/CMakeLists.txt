# KeplerSolarSystem: the real Sun, all 8 planets, every moon JPL SSD
# publishes orbital elements for, the 5 IAU dwarf planets, and procedural
# asteroid-belt/Kuiper-belt populations -- every body's position evaluated
# directly from its own Kepler orbital elements at whatever simulation time
# is asked for, not integrated forward by a real n-body scheme. See
# src/Applications/KeplerSolarSystem/README.md for why this is a separate
# app from Applications/SolarSystem rather than a mode of it.
#
# Scenario construction is its own library, with no Platform/Renderer/UI
# dependency, so it builds and links under -DYSQ_BUILD_GRAPHICS=OFF too and
# tests/unit/kepler_solar_system_scenario.cpp can assert the shipped
# scenario's own invariants without pulling in graphics. The executable
# itself needs a window, so it is graphics-only.

add_library(ysq_kepler_solar_system_scenario STATIC
    Scenario.hpp
    Scenario.cpp)
add_library(ysq::KeplerSolarSystemScenario ALIAS ysq_kepler_solar_system_scenario)

target_include_directories(ysq_kepler_solar_system_scenario PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})
target_link_libraries(ysq_kepler_solar_system_scenario
    PUBLIC ysq::Physics ysq::ApplicationsHelper
    PRIVATE ysq::Core ysq::warnings)

# Scenario.cpp locates data/solar_system_bodies.csv by absolute source
# path, baked in at build time: robust to whatever directory the built
# executable is actually run from, unlike a path relative to the current
# working directory. This is KeplerSolarSystem's own copy of the data file,
# independent of Applications/SolarSystem/data's; see that file's own
# header comment.
target_compile_definitions(ysq_kepler_solar_system_scenario PRIVATE
    YSQ_KEPLER_SOLAR_SYSTEM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

if(YSQ_BUILD_GRAPHICS)
    add_executable(kepler-solar-system main.cpp)
    # Math, Units and Physics come in PUBLIC via KeplerSolarSystemScenario;
    # Platform comes in PUBLIC via both Renderer and UI. Core is used
    # directly (Logger/Timer/UUID/Config) and nothing already listed
    # exposes it PUBLIC, so it needs to be named here. Unlike
    # Applications::SolarSystem, this app runs no per-frame energy/momentum
    # reduction, so it has no need of ysq::Compute at all. See
    # src/Applications/README.md for the convention this follows.
    target_link_libraries(kepler-solar-system PRIVATE
        ysq::KeplerSolarSystemScenario
        ysq::Core
        ysq::Renderer
        ysq::UI
        ysq::warnings)
endif()
