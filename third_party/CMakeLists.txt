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
    ysq_require_submodule(implot implot.cpp)
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
    target_link_libraries(imgui PUBLIC glfw glad)

    # ImGui's GL3 backend defaults to its own bundled loader
    # (imgui_impl_opengl3_loader.h), which dlopens the system's real GL
    # library independently of GLFW or GLAD. That works with a real display,
    # but not on headless Linux CI: the runner has OSMesa, which GLAD reaches
    # correctly through glfwGetProcAddress, and no system libGL/GLX for
    # ImGui's own loader to find, so ImGui_ImplOpenGL3_Init() fails there
    # ("Failed to initialize OpenGL loader!") even though every other GL path
    # in this repo works fine on the same runner.
    #
    # IMGUI_IMPL_OPENGL_LOADER_CUSTOM skips that bundled loader entirely and
    # makes imgui_impl_opengl3.cpp use whatever GL declarations are already
    # visible instead -- GLAD's, force-included here since the file is
    # vendored and not ours to add an #include to. GLAD is already proven to
    # load correctly under OSMesa by every other test in this suite.
    target_compile_definitions(imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_CUSTOM)
    # -include takes its path as a separate argv token, not attached to the
    # flag, which is why this is a two-element list rather than one string.
    if(MSVC)
        set(_imgui_opengl3_force_include "/FI${CMAKE_CURRENT_SOURCE_DIR}/glad/include/glad/gl.h")
    else()
        set(_imgui_opengl3_force_include
            "-include" "${CMAKE_CURRENT_SOURCE_DIR}/glad/include/glad/gl.h")
    endif()
    set_source_files_properties(imgui/backends/imgui_impl_opengl3.cpp PROPERTIES
        COMPILE_OPTIONS "${_imgui_opengl3_force_include}")

    # Dear ImPlot: charting on top of ImGui, same "ships sources, no build
    # system" shape. Pinned to v1.0, whose own IMGUI_VERSION_NUM guards are
    # satisfied by the imgui version pinned above.
    add_library(implot STATIC
        implot/implot.cpp
        implot/implot_items.cpp)
    target_include_directories(implot SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/implot)
    target_link_libraries(implot PUBLIC imgui)

    # stb_image: single public-domain header, vendored and committed rather
    # than submoduled, the same treatment as glad/. stb_image_impl.cpp is the
    # one translation unit that defines STB_IMAGE_IMPLEMENTATION and compiles
    # the implementation; every consumer links this target and gets
    # declarations plus symbols, rather than each risking a duplicate
    # definition by defining the macro itself.
    add_library(stb_image STATIC stb/stb_image_impl.cpp)
    target_include_directories(stb_image SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/stb)
endif()
