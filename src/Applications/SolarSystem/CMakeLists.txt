# SolarSystem: the real Sun, all 8 planets, and every moon JPL SSD
# publishes orbital elements for, under Newtonian N-body gravity.
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
    PUBLIC ysq::Physics ysq::ApplicationsHelper
    PRIVATE ysq::Core ysq::warnings)

# Scenario.cpp looks for data/solar_system_bodies.csv next to its own
# running executable first (Core::executableDirectory), which is where a
# packaged release ships it (see this file's install(DIRECTORY ...) below);
# this compile-time source path is only the fallback for a local, uninstalled
# build (a plain `cmake --build`, or any test linking this library directly),
# where no such copy exists next to whatever executable is actually running.
target_compile_definitions(ysq_solar_system_scenario PRIVATE
    YSQ_SOLAR_SYSTEM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

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

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # A downloaded binary needs to run on whatever libstdc++ version the
        # user's own distro ships, not just the one this was built against.
        target_link_options(solar-system PRIVATE -static-libgcc -static-libstdc++)
    endif()

    # One app per top-level install directory: release packaging zips each
    # directory as that app's own downloadable archive, and the whole
    # install prefix together as the combined "every app" archive. See
    # src/Applications/README.md's convention section.
    install(TARGETS solar-system RUNTIME DESTINATION SolarSystem)
    # Ships data/ right next to the installed executable, exactly where
    # Scenario.cpp's own executable-relative lookup expects to find it.
    install(DIRECTORY data/ DESTINATION SolarSystem/data)
endif()
