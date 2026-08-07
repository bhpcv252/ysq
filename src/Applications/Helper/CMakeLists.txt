# Shared scenario-setup code between applications: CSV loading, name/parent
# hierarchy resolution, and the RA/Dec pole convention for interpreting real
# catalog data. Not engine content -- it knows about named bodies, colors,
# and how a scenario's data file is laid out. See README.md. The orbital
# element and Kepler's-equation machinery this builds on lives in
# Physics/Gravity/Kepler.hpp instead, since that has no scenario knowledge
# of its own.

add_library(ysq_applications_helper STATIC
    Pole.hpp
    Pole.cpp
    BodyCatalog.hpp
    BodyCatalog.cpp
    KeplerPopulation.hpp
    KeplerPopulation.cpp)
add_library(ysq::ApplicationsHelper ALIAS ysq_applications_helper)

target_include_directories(ysq_applications_helper PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})
target_link_libraries(ysq_applications_helper
    PUBLIC ysq::Math ysq::Core ysq::Physics
    PRIVATE ysq::warnings)
