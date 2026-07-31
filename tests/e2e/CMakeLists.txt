# Whole applications, headless, asserting physical invariants. No graphics
# dependency here: each app's scenario-construction code is a separate,
# graphics-free library (see src/Applications/README.md), and that is what
# these link, not the windowed executable itself.

ysq_add_test(test_solar_system SOURCES solar_system.cpp
    LIBS ysq::SolarSystemScenario ysq::Physics ysq::Math)

ysq_add_test(test_lunar_eclipse SOURCES lunar_eclipse.cpp
    LIBS ysq::LunarEclipseScenario ysq::Physics ysq::Math)
