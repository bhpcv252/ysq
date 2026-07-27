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
