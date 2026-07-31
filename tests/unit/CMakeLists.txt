ysq_add_test(test_core_clock SOURCES core_clock.cpp LIBS ysq::Core)
ysq_add_test(test_core_config SOURCES core_config.cpp LIBS ysq::Core)
ysq_add_test(test_core_event SOURCES core_event.cpp LIBS ysq::Core)
ysq_add_test(test_core_logger SOURCES core_logger.cpp LIBS ysq::Core)
ysq_add_test(test_core_timer SOURCES core_timer.cpp LIBS ysq::Core)
ysq_add_test(test_core_uuid SOURCES core_uuid.cpp LIBS ysq::Core)
ysq_add_test(test_core_version SOURCES core_version.cpp LIBS ysq::Core)

ysq_add_test(test_math_vector SOURCES math_vector.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_matrix SOURCES math_matrix.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_intersection SOURCES math_intersection.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_quaternion SOURCES math_quaternion.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_complex SOURCES math_complex.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_dual SOURCES math_dual.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_tensor SOURCES math_tensor.cpp
    LIBS ysq::Math ysq::TestSupport)

# The version as CMake renders it, so the test compares two independent paths
# out of project(YSQ VERSION ...) rather than one value against itself.
target_compile_definitions(test_core_version PRIVATE
    YSQ_EXPECTED_VERSION_STRING="${PROJECT_VERSION}")
ysq_add_test(test_math_statistics SOURCES math_statistics.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_interpolation SOURCES math_interpolation.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_calculus SOURCES math_calculus.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_coordinates SOURCES math_coordinates.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_ode SOURCES math_ode.cpp
    LIBS ysq::Math ysq::TestSupport)
ysq_add_test(test_math_integrators SOURCES math_integrators.cpp
    LIBS ysq::Math ysq::TestSupport)

# Units links ysq::Units for the module itself and ysq::TestSupport only for
# the tests/ include root, which is what makes <support/UnitsApprox.hpp>
# resolve. TestSupport deliberately does not link Units: a Math-only test
# should not acquire a dependency on it.
ysq_add_test(test_units_dimensions SOURCES units_dimensions.cpp
    LIBS ysq::Units ysq::TestSupport)
ysq_add_test(test_units_quantity SOURCES units_quantity.cpp
    LIBS ysq::Units ysq::TestSupport)
ysq_add_test(test_units_conversions SOURCES units_conversions.cpp
    LIBS ysq::Units ysq::TestSupport)
ysq_add_test(test_units_vector SOURCES units_vector.cpp
    LIBS ysq::Units ysq::TestSupport)
ysq_add_test(test_units_format SOURCES units_format.cpp
    LIBS ysq::Units ysq::TestSupport)

# Platform exists only in a graphics build. Both tests link glfw as well as the
# module: the module keeps GLFW out of its headers, and these assert the mapping
# onto GLFW's own constants, so the test needs what the module hides.
if(YSQ_BUILD_GRAPHICS)
    ysq_add_test(test_platform_input SOURCES platform_input.cpp
        LIBS ysq::Platform glfw)
    ysq_add_test(test_platform_window SOURCES platform_window.cpp
        LIBS ysq::Platform glfw)
endif()

# Camera's projection/view factories and both controllers are pure functions
# over Math and Platform::InputState, so they need no GL context; see
# tests/integration/renderer_framebuffer.cpp for the part that does.
if(YSQ_BUILD_GRAPHICS)
    ysq_add_test(test_renderer_camera SOURCES renderer_camera.cpp
        LIBS ysq::Renderer ysq::TestSupport)
    # Pure atlas/UV logic, no GL context; see
    # tests/integration/renderer_text.cpp for the part that draws.
    ysq_add_test(test_renderer_font SOURCES renderer_font.cpp LIBS ysq::Renderer)
endif()

ysq_add_test(test_compute_cpu SOURCES compute_cpu.cpp LIBS ysq::Compute)
ysq_add_test(test_compute_backend SOURCES compute_backend.cpp LIBS ysq::Compute)

ysq_add_test(test_physics_body SOURCES physics_body.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_physics_mechanics SOURCES physics_mechanics.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_physics_rigidbody SOURCES physics_rigidbody.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_physics_gravity SOURCES physics_gravity.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_physics_barneshut SOURCES physics_barneshut.cpp
    LIBS ysq::Physics ysq::TestSupport)

ysq_add_test(test_spacetime_metric SOURCES spacetime_metric.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_spacetime_schwarzschild SOURCES spacetime_schwarzschild.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_spacetime_kerr SOURCES spacetime_kerr.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_spacetime_flrw SOURCES spacetime_flrw.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_spacetime_geodesic SOURCES spacetime_geodesic.cpp
    LIBS ysq::Physics ysq::TestSupport)

ysq_add_test(test_optics_propagation SOURCES optics_propagation.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_optics_refractivemedium SOURCES optics_refractivemedium.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_optics_rayleighscattering SOURCES optics_rayleighscattering.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_optics_illumination SOURCES optics_illumination.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_optics_lensing SOURCES optics_lensing.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_optics_frequencyshift SOURCES optics_frequencyshift.cpp
    LIBS ysq::Physics ysq::TestSupport)

ysq_add_test(test_em_field SOURCES em_field.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_em_lorentz SOURCES em_lorentz.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_em_maxwell SOURCES em_maxwell.cpp
    LIBS ysq::Physics ysq::TestSupport)

ysq_add_test(test_fluids_sph SOURCES fluids_sph.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_fluids_eulerian SOURCES fluids_eulerian.cpp
    LIBS ysq::Physics ysq::TestSupport)

ysq_add_test(test_thermodynamics SOURCES thermodynamics.cpp
    LIBS ysq::Physics ysq::TestSupport)
ysq_add_test(test_thermodynamics_heatequation SOURCES thermodynamics_heatequation.cpp
    LIBS ysq::Physics ysq::TestSupport)

# Applications' scenario-construction code, not the windowed executables
# themselves; see src/Applications/README.md. No graphics dependency.
ysq_add_test(test_solar_system_scenario SOURCES solar_system_scenario.cpp
    LIBS ysq::SolarSystemScenario ysq::Math)
