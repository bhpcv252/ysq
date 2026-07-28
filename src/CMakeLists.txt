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
