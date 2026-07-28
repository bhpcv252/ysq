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

# A real OpenGL context, which is the one thing in the suite that depends on the
# machine rather than only on the code. It skips where no context can exist.
#
# A compile definition rather than an environment variable read at run time:
# std::getenv is deprecated by MSVC, and the project builds with warnings as
# errors, so reading it would cost either a suppression or a per-compiler
# wrapper for a value that is fixed for the whole run anyway.
if(YSQ_BUILD_GRAPHICS)
    ysq_add_test(test_platform_context SOURCES platform_context.cpp
        LIBS ysq::Platform glfw)
    target_compile_definitions(test_platform_context PRIVATE
        $<$<BOOL:${YSQ_REQUIRE_HEADLESS_GL}>:YSQ_REQUIRE_HEADLESS_GL>)
endif()

# Runs every backend that reports itself available against the CPU reference.
# On a machine with no GPU and no CUDA/Vulkan SDK this exercises CPU only and
# says so via GTEST_SKIP rather than passing on having checked nothing.
ysq_add_test(test_compute_backends_agree SOURCES compute_backends_agree.cpp
    LIBS ysq::Compute)
