# Compute backends: a CPU reference (always built) plus GPU acceleration that
# is only as available as the hardware and SDKs on the machine building it.
# Physics dispatches through ComputeBackend and never talks to a GPU directly;
# see docs/architecture.md.

add_library(ysq_compute STATIC
    ComputeBackend.hpp
    ComputeBackend.cpp
    CPU/CpuBackend.hpp
    CPU/CpuBackend.cpp)
add_library(ysq::Compute ALIAS ysq_compute)

target_include_directories(ysq_compute PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})

# A real STATIC target, unlike Math and Units, so it carries the strict
# warning set directly instead of needing a smoke test to apply it. Compute is
# in the engine core per docs/architecture.md: a silent double -> float
# narrowing here is the same class of physics bug it would be in Math.
target_link_libraries(ysq_compute
    PRIVATE ysq::Core ysq::warnings_strict)

# --- OpenGL backend -----------------------------------------------------------
# Needs an offscreen context from Platform, which only exists in a graphics
# build. With YSQ_BUILD_GRAPHICS=OFF this backend is not compiled at all, and
# computeBackendAvailable(OpenGL) always answers false.
if(YSQ_BUILD_GRAPHICS)
    target_sources(ysq_compute PRIVATE
        OpenGL/ComputeShader.hpp
        OpenGL/ComputeShader.cpp
        OpenGL/OpenGLBackend.hpp
        OpenGL/OpenGLBackend.cpp)
    target_link_libraries(ysq_compute PRIVATE ysq::Platform glad)
    target_compile_definitions(ysq_compute PRIVATE YSQ_COMPUTE_HAS_OPENGL)
endif()

# --- CUDA backend --------------------------------------------------------------
# Detected, not required: an absent toolkit is a STATUS line, never a build
# failure, unlike the vendored submodule check in third_party/CMakeLists.txt.
#
# No .cu sources yet and no enable_language(CUDA): CudaBackend::create() does a
# real cudaGetDeviceCount() probe through the CUDA Runtime API, which is
# callable from plain C++ linked against CUDA::cudart, but its kernels are not
# implemented. See src/Compute/README.md.
if(YSQ_BUILD_COMPUTE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()
if(CUDAToolkit_FOUND)
    target_sources(ysq_compute PRIVATE
        CUDA/CudaBackend.hpp
        CUDA/CudaBackend.cpp)
    target_link_libraries(ysq_compute PRIVATE CUDA::cudart)
    target_compile_definitions(ysq_compute PRIVATE YSQ_COMPUTE_HAS_CUDA)
endif()

# --- Vulkan backend --------------------------------------------------------------
# Same shape as CUDA above: detected, a real instance/device probe, no kernels
# yet. The loader is a plain C API, so no shader toolchain (glslc) is needed
# until real kernels land either.
if(YSQ_BUILD_COMPUTE_VULKAN)
    find_package(Vulkan QUIET)
endif()
if(Vulkan_FOUND)
    target_sources(ysq_compute PRIVATE
        Vulkan/VulkanBackend.hpp
        Vulkan/VulkanBackend.cpp)
    target_link_libraries(ysq_compute PRIVATE Vulkan::Vulkan)
    target_compile_definitions(ysq_compute PRIVATE YSQ_COMPUTE_HAS_VULKAN)
endif()

# CUDAToolkit_FOUND / Vulkan_FOUND are only ever set by find_package above,
# which only runs when the matching YSQ_BUILD_COMPUTE_* option is on; with the
# option off they are unset, which would otherwise print as a blank line
# rather than a clear "disabled by option".
if(NOT YSQ_BUILD_COMPUTE_CUDA)
    set(_ysq_compute_cuda_status "disabled by option")
elseif(CUDAToolkit_FOUND)
    set(_ysq_compute_cuda_status "ON")
else()
    set(_ysq_compute_cuda_status "not found")
endif()

if(NOT YSQ_BUILD_COMPUTE_VULKAN)
    set(_ysq_compute_vulkan_status "disabled by option")
elseif(Vulkan_FOUND)
    set(_ysq_compute_vulkan_status "ON")
else()
    set(_ysq_compute_vulkan_status "not found")
endif()

message(STATUS "Compute backends:")
message(STATUS "  CPU:    always")
message(STATUS "  OpenGL: ${YSQ_BUILD_GRAPHICS} (compiled in; a 4.3+ context is a runtime, per-machine check)")
message(STATUS "  CUDA:   ${_ysq_compute_cuda_status}")
message(STATUS "  Vulkan: ${_ysq_compute_vulkan_status}")
