# Vendored dependencies. Everything here is third-party code; see README.md for
# pinned versions and the GLAD regeneration command.

# Third-party headers are marked SYSTEM so their warnings stay out of our build.
# CMake 3.25 added a SYSTEM argument to add_subdirectory; we target 3.20, so set
# the property directly instead.
function(ysq_includes_as_system target)
    get_target_property(_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_dirs)
        set_target_properties(${target} PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_dirs}")
    endif()
endfunction()

# Missing submodules are the most likely first failure for a new clone, and the
# raw CMake errors are a wall of "non-existent target" noise that never mentions
# submodules. Fail once, early, naming the command that fixes it. Only what the
# current configuration actually needs is checked.
function(ysq_require_submodule name marker)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}/${marker})
        message(FATAL_ERROR
            "third_party/${name} is empty: submodules have not been fetched.\n"
            "Run:  git submodule update --init --recursive")
    endif()
endfunction()

ysq_require_submodule(spdlog CMakeLists.txt)
if(YSQ_BUILD_TESTS)
    ysq_require_submodule(googletest CMakeLists.txt)
endif()
if(YSQ_BUILD_GRAPHICS)
    ysq_require_submodule(glfw CMakeLists.txt)
    ysq_require_submodule(imgui imgui.cpp)
endif()

# --- spdlog -------------------------------------------------------------------
# std::format instead of bundled fmt, so the project has exactly one formatting
# implementation and no extra dependency.
set(SPDLOG_USE_STD_FORMAT ON CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)
ysq_includes_as_system(spdlog)

# --- googletest ---------------------------------------------------------------
if(YSQ_BUILD_TESTS)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(googletest EXCLUDE_FROM_ALL)
    foreach(_t gtest gtest_main gmock gmock_main)
        ysq_includes_as_system(${_t})
    endforeach()
endif()

# --- graphics stack -----------------------------------------------------------
# Gated as a unit. With YSQ_BUILD_GRAPHICS=OFF none of this is configured, which
# is what keeps the headless core path honest rather than merely claimed.
#
# Scoped with if/endif rather than an early return() so that anything added
# below this block is not silently made graphics-only.
if(YSQ_BUILD_GRAPHICS)
    # GLFW: window, input, OpenGL context.
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(glfw EXCLUDE_FROM_ALL)
    ysq_includes_as_system(glfw)

    # GLAD: generated OpenGL loader, committed rather than submoduled.
    # Every GL entry point is a function pointer resolved at runtime, so nothing
    # here needs to link against a system OpenGL library.
    add_library(glad STATIC glad/src/gl.c)
    target_include_directories(glad SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/glad/include)

    # Dear ImGui: ships sources, no build system of its own.
    add_library(imgui STATIC
        imgui/imgui.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp
        imgui/imgui_demo.cpp
        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends)
    target_link_libraries(imgui PUBLIC glfw)

    # ImGui's GL3 backend keeps its own minimal loader. It is file-local to
    # imgui_impl_opengl3.cpp, so it cannot collide with GLAD, and it is the
    # configuration upstream actually tests. Overriding it with
    # IMGUI_IMPL_OPENGL_LOADER_CUSTOM is possible but buys nothing here.
endif()
