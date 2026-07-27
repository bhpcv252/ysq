add_library(ysq_platform STATIC
    Input.cpp
    Platform.cpp
    Window.cpp
    Input.hpp
    Platform.hpp
    Window.hpp)
add_library(ysq::Platform ALIAS ysq_platform)

target_include_directories(ysq_platform PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})

# GLAD is PUBLIC: Platform owns the loader, and anything holding a live context
# has to call GL through the same process-wide entry points this module resolved.
# GLFW is PRIVATE, because no header here names it beyond a forward declaration.
# Core is PRIVATE for the same reason it is in every other module: logging is an
# implementation detail, not part of the interface.
target_link_libraries(ysq_platform
    PUBLIC glad
    PRIVATE glfw ysq::Core ysq::warnings)

# Deliberately not ysq::warnings_strict. docs/architecture.md keeps the
# conversion warnings on the engine core, where a silent narrowing is a physics
# bug; here every int/float conversion is one GLFW or OpenGL asked for.
