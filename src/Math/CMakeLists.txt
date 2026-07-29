# Header-only, so INTERFACE. There is no .cpp here and there is not meant to
# be: everything in Math is a template or constexpr.
add_library(ysq_math INTERFACE)
add_library(ysq::Math ALIAS ysq_math)

target_include_directories(ysq_math INTERFACE ${YSQ_SOURCE_INCLUDE_DIR})

# Deliberately no ysq::warnings_strict here. On an INTERFACE target the flags
# would propagate to every consumer's own sources, which would put
# -Wconversion -Wsign-conversion -Wdouble-promotion on Renderer, UI and
# Applications. The root README's Warnings section rules that out: those
# layers talk to OpenGL and ImGui, which are float/int APIs, so the strict
# set there is noise that trains people to reach for static_cast.
#
# The strict set is applied instead in tests/smoke/math_strict_warnings.cpp,
# which includes every header here and explicitly instantiates the templates so
# the warnings have something to fire on. An uninstantiated template is barely
# checked at all, so instantiating is what makes that test mean anything.
