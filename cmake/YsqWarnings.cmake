# Two warning levels, both INTERFACE targets meant to be linked PRIVATE so they
# apply to a module's own sources without leaking to its consumers.
#
#   ysq::warnings         every YSQ target
#   ysq::warnings_strict  engine core only (Math, Units, Physics, Compute)
#
# The strict set adds conversion warnings. In the engine core a silent
# double -> float narrowing is a physics bug: an energy accumulator or an
# integrator tolerance quietly loses precision and an invariant drifts. In the
# presentation layer (Renderer, UI, Platform) those same conversions are
# constant and deliberate, because OpenGL and ImGui are float/int APIs, so the
# strict set there would be noise that trains people to reach for static_cast.

add_library(ysq_warnings INTERFACE)
add_library(ysq::warnings ALIAS ysq_warnings)

add_library(ysq_warnings_strict INTERFACE)
add_library(ysq::warnings_strict ALIAS ysq_warnings_strict)

if(MSVC)
    # /external:W0 is what actually silences third-party headers. CMake emits
    # /external:I for SYSTEM includes but does not set their warning level, so
    # without this /W4 still fires on spdlog, ImGui and GoogleTest.
    set(_ysq_base /W4 /permissive- /external:W0)
    # C4242/C4244 narrowing, C4245/C4365 signed-unsigned mismatch, C4305 truncation.
    set(_ysq_strict ${_ysq_base} /w14242 /w14244 /w14245 /w14305 /w14365)
    set(_ysq_werror /WX)
else()
    set(_ysq_base -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
    # -Wsign-conversion is listed explicitly because Clang's -Wconversion implies
    # it for C++ and GCC's does not. Without it the two compilers disagree and a
    # build that is clean locally fails in CI.
    set(_ysq_strict ${_ysq_base} -Wconversion -Wsign-conversion -Wdouble-promotion)
    set(_ysq_werror -Werror)
endif()

if(YSQ_WARNINGS_AS_ERRORS)
    list(APPEND _ysq_base ${_ysq_werror})
    list(APPEND _ysq_strict ${_ysq_werror})
endif()

# Guarded by COMPILE_LANGUAGE so these never reach CUDA sources later.
target_compile_options(ysq_warnings INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:${_ysq_base}>")
target_compile_options(ysq_warnings_strict INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:${_ysq_strict}>")
