# Engine modules. Each is its own library target, added here as it lands.
# Dependencies flow one way: nothing lower depends on anything higher.

add_subdirectory(Core)
add_subdirectory(Math)
add_subdirectory(Units)

# Platform wraps GLFW and GLAD, so it exists only in a graphics build. Nothing
# in the simulation core may depend on it; that is what keeps a headless build
# a real configuration rather than a claim.
if(YSQ_BUILD_GRAPHICS)
    add_subdirectory(Platform)
endif()

# Compute is added after Platform: its OpenGL backend conditionally links
# ysq::Platform, which must already be a defined target when that happens. The
# CPU backend always builds regardless of every option above.
add_subdirectory(Compute)

# Physics builds on Compute (falling back to its CPU backend with no GPU) and
# on Math/Units below it; see docs/architecture.md.
add_subdirectory(Physics)

# Renderer and UI are the presentation layer: they depend only on Platform
# and Math, never Physics, so the engine stays domain-neutral. Graphics-only,
# same as Platform.
if(YSQ_BUILD_GRAPHICS)
    add_subdirectory(Renderer)
    add_subdirectory(UI)
endif()

# Applications sit on top of everything. Added unconditionally: an
# application's executable is graphics-only and says so itself, but its
# scenario-construction code is pure physics and tests/e2e/ links it headless.
add_subdirectory(Applications)
