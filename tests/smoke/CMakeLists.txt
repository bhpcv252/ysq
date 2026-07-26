# Smoke tests prove the vendored dependencies are genuinely wired, not merely
# checked out. They assert nothing about physics.

ysq_add_test(test_spdlog_format SOURCES spdlog_format.cpp LIBS spdlog::spdlog)

# Not a dependency check: this one proves the engine core's own templates
# survive the strict warning set, which Math cannot apply to itself without
# leaking it to every consumer. See src/Math/CMakeLists.txt.
ysq_add_test(test_math_strict_warnings STRICT
    SOURCES math_strict_warnings.cpp
    LIBS ysq::Math)

if(YSQ_BUILD_GRAPHICS)
    ysq_add_test(test_graphics_link SOURCES graphics_link.cpp LIBS glad imgui)
endif()
