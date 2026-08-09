# LunarEclipse: Sun, Earth, Moon, Jupiter under real n-body Newtonian
# gravity with Earth's J2 and rotation active, and the Moon's illumination
# during any eclipse computed from Optics/Illumination.hpp rather than
# scripted; see src/Applications/README.md.
#
# Scenario construction is its own library, with no Platform/Renderer/UI
# dependency, so it builds and links under -DYSQ_BUILD_GRAPHICS=OFF too and
# tests/e2e/lunar_eclipse.cpp can assert the shipped scenario's physical
# invariants without pulling in graphics. The executable itself needs a
# window, so it is graphics-only.

add_library(ysq_lunar_eclipse_scenario STATIC
    Scenario.hpp
    Scenario.cpp)
add_library(ysq::LunarEclipseScenario ALIAS ysq_lunar_eclipse_scenario)

target_include_directories(ysq_lunar_eclipse_scenario PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})
target_link_libraries(ysq_lunar_eclipse_scenario
    PUBLIC ysq::Physics
    PRIVATE ysq::ApplicationsHelper ysq::warnings)

if(YSQ_BUILD_GRAPHICS)
    add_executable(lunar-eclipse main.cpp)
    # Math, Units and Physics come in PUBLIC via LunarEclipseScenario;
    # Platform comes in PUBLIC via both Renderer and UI. Core and Compute
    # are used directly and nothing already listed exposes either PUBLIC,
    # so both need to be named here. See src/Applications/README.md.
    target_link_libraries(lunar-eclipse PRIVATE
        ysq::LunarEclipseScenario
        ysq::Core
        ysq::Compute
        ysq::Renderer
        ysq::UI
        ysq::warnings)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # A downloaded binary needs to run on whatever libstdc++ version the
        # user's own distro ships, not just the one this was built against.
        target_link_options(lunar-eclipse PRIVATE -static-libgcc -static-libstdc++)
    endif()

    # One app per top-level install directory: release packaging zips each
    # directory as that app's own downloadable archive, and the whole
    # install prefix together as the combined "every app" archive. See
    # src/Applications/README.md's convention section.
    install(TARGETS lunar-eclipse RUNTIME DESTINATION LunarEclipse)
endif()
