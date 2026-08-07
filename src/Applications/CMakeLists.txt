# Applications: runnable programs built on the engine. One subdirectory each;
# see README.md for the convention. Each application's own executable is
# graphics-only, but this directory itself is added unconditionally, same as
# the outer src/CMakeLists.txt does for it.

# Helper first: SolarSystem, LunarEclipse and KeplerSolarSystem all link it.
add_subdirectory(Helper)
add_subdirectory(SolarSystem)
add_subdirectory(LunarEclipse)
add_subdirectory(KeplerSolarSystem)
