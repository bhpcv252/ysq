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

# Gravity's Newtonian rung, and then Barnes-Hut, each under a symplectic
# integrator, checked against the physical invariants they are supposed to
# preserve rather than against a fixed expected trajectory.
ysq_add_test(test_orbit_stability SOURCES orbit_stability.cpp
    LIBS ysq::Physics ysq::Math)
ysq_add_test(test_nbody_energy SOURCES nbody_energy.cpp
    LIBS ysq::Physics ysq::Math)

# Spacetime + Optics: the deflection angle and the perihelion precession,
# validated against closed forms, with precession also cross-checked
# against Gravity's own 1PN prediction.
ysq_add_test(test_lensing_deflection SOURCES lensing_deflection.cpp
    LIBS ysq::Physics ysq::Math)

ysq_add_test(test_fluids_shocktube SOURCES fluids_shocktube.cpp
    LIBS ysq::Physics ysq::Math)

# A real OpenGL context, same as test_platform_context: geometry drawn to an
# offscreen framebuffer, and a fragment-shader ray trace with shadows and
# reflections, both read back and checked pixel by pixel. See
# support/GLContext.hpp.
if(YSQ_BUILD_GRAPHICS)
    ysq_add_test(test_renderer_framebuffer SOURCES renderer_framebuffer.cpp
        LIBS ysq::Renderer ysq::Platform ysq::TestSupport)
    ysq_add_test(test_renderer_raytracer SOURCES renderer_raytracer.cpp
        LIBS ysq::Renderer ysq::Platform ysq::TestSupport)

    # Links imgui directly, not just ysq::UI, because the test itself calls
    # ImGui:: to inspect draw data, the same reason test_graphics_link does.
    ysq_add_test(test_ui_imgui_layer SOURCES ui_imgui_layer.cpp
        LIBS ysq::UI ysq::Platform ysq::TestSupport imgui)
endif()
