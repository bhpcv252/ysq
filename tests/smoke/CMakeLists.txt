# Smoke tests prove the vendored dependencies are genuinely wired, not merely
# checked out. They assert nothing about physics.

ysq_add_test(test_spdlog_format SOURCES spdlog_format.cpp LIBS spdlog::spdlog)

if(YSQ_BUILD_GRAPHICS)
    ysq_add_test(test_graphics_link SOURCES graphics_link.cpp LIBS glad imgui)
endif()
