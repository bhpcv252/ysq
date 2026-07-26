# Cross-module behaviour.

ysq_add_test(test_core_runtime SOURCES core_runtime.cpp LIBS ysq::Core)

# Six Math headers on one physical problem. Each is unit tested on its own;
# this asks whether they add up.
ysq_add_test(test_math_kepler SOURCES math_kepler.cpp
    LIBS ysq::Math ysq::TestSupport)
