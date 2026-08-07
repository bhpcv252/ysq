cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed. Configure into a separate "
        "directory instead, e.g.: cmake -B build")
endif()

project(YSQ
    VERSION 0.1.0
    DESCRIPTION "Yotta Spacetime Quantities: a C++ engine for simulating and visualizing physical systems"
    LANGUAGES CXX C)

# C is enabled for GLAD's generated loader, the only C source in the tree.

# Single-config generators (Makefiles, Ninja) leave this empty with no flags
# of their own otherwise, which is neither Debug nor Release, just
# unoptimized. Multi-config generators (Xcode, Ninja Multi-Config) use
# CMAKE_CONFIGURATION_TYPES instead and ignore this entirely, so this must not
# touch them.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YSQ_BUILD_TESTS "Build the YSQ test suite" OFF)
option(YSQ_BUILD_COMPILE_FAIL_TESTS
    "Build the tests that assert something does not compile" ON)
option(YSQ_BUILD_GRAPHICS "Build graphics-dependent modules (GLFW, GLAD, Dear ImGui)" ON)
option(YSQ_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
# Whether an OpenGL context can be created is a property of the machine, so the
# context test skips where none can exist. A file of tests that always skip
# tests nothing, so on a machine known to have OSMesa this turns those skips
# into failures. CI sets it on the one job that qualifies.
option(YSQ_REQUIRE_HEADLESS_GL
    "Fail rather than skip when no headless OpenGL context can be created" OFF)
# CUDA and Vulkan are detected, not vendored: unlike third_party/, an absent
# toolkit or SDK is not an error, so these mean "use it if found" rather than
# "require it". See src/Compute/CMakeLists.txt.
option(YSQ_BUILD_COMPUTE_CUDA "Build the CUDA compute backend if the toolkit is found" ON)
option(YSQ_BUILD_COMPUTE_VULKAN "Build the Vulkan compute backend if the SDK is found" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# src/ is the include root, so every include is module-qualified:
#   #include <Core/Version.hpp>
# Generated headers mirror the same layout, so a configured header resolves
# exactly like a checked-in one.
set(YSQ_SOURCE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(YSQ_GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/generated)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(YsqWarnings)

add_subdirectory(third_party)
add_subdirectory(src)

if(YSQ_BUILD_TESTS)
    # enable_testing() must run at top level for `ctest --test-dir build` to
    # find anything from the build root.
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "YSQ ${PROJECT_VERSION}")
message(STATUS "  tests:        ${YSQ_BUILD_TESTS}")
message(STATUS "  compile-fail: ${YSQ_BUILD_COMPILE_FAIL_TESTS}")
message(STATUS "  graphics:     ${YSQ_BUILD_GRAPHICS}")
if(YSQ_BUILD_GRAPHICS)
    message(STATUS "  require GL:   ${YSQ_REQUIRE_HEADLESS_GL}")
endif()
