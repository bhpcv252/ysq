# Cross-module behaviour.

ysq_add_test(test_core_runtime SOURCES core_runtime.cpp LIBS ysq::Core)

# Six Math headers on one physical problem. Each is unit tested on its own;
# this asks whether they add up.
ysq_add_test(test_math_kepler SOURCES math_kepler.cpp
    LIBS ysq::Math ysq::TestSupport)

# Units meeting the Math integrators, which is the join that does not happen
# automatically: a dimensioned state cannot be handed to a stepper, and the
# conversion boundary is where Physics will live.
ysq_add_test(test_units_kinematics SOURCES units_kinematics.cpp
    LIBS ysq::Units ysq::Math ysq::TestSupport)
