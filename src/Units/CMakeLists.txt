# Header-only, so INTERFACE. Everything here is a template or constexpr, and a
# dimension has no representation at run time at all.
add_library(ysq_units INTERFACE)
add_library(ysq::Units ALIAS ysq_units)

target_include_directories(ysq_units INTERFACE ${YSQ_SOURCE_INCLUDE_DIR})
target_link_libraries(ysq_units INTERFACE ysq::Math)

# No ysq::warnings_strict here, for the same reason as src/Math/CMakeLists.txt:
# on an INTERFACE target the flags propagate to every consumer's own sources,
# which would put -Wconversion -Wsign-conversion -Wdouble-promotion on
# Renderer, UI and Applications. The root README's Warnings section rules
# that out.
#
# The strict set is applied instead in tests/smoke/units_strict_warnings.cpp,
# which includes every header here and instantiates the templates so the
# warnings have something to fire on.
