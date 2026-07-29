# SolarSystem: Sun + five planets under Newtonian N-body gravity.
#
# Scenario construction is its own library, with no Platform/Renderer/UI
# dependency, so it builds and links under -DYSQ_BUILD_GRAPHICS=OFF too and
# tests/e2e/solar_system.cpp can assert the shipped scenario's physical
# invariants without pulling in graphics. The executable itself needs a
# window, so it is graphics-only.

add_library(ysq_solar_system_scenario STATIC
    Scenario.hpp
    Scenario.cpp)
add_library(ysq::SolarSystemScenario ALIAS ysq_solar_system_scenario)

target_include_directories(ysq_solar_system_scenario PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})
target_link_libraries(ysq_solar_system_scenario
    PUBLIC ysq::Physics
    PRIVATE ysq::warnings)

if(YSQ_BUILD_GRAPHICS)
    add_executable(solar-system main.cpp)
    # Math, Units and Physics come in PUBLIC via SolarSystemScenario; Platform
    # comes in PUBLIC via both Renderer and UI. Core and Compute are used
    # directly (Clock/Logger/Timer/UUID/Config/Event, ComputeBackend) and
    # nothing already listed exposes either PUBLIC, so both need to be named
    # here. See src/Applications/README.md for the convention this follows.
    target_link_libraries(solar-system PRIVATE
        ysq::SolarSystemScenario
        ysq::Core
        ysq::Compute
        ysq::Renderer
        ysq::UI
        ysq::warnings)
endif()
