#include <Compute/CPU/CpuBackend.hpp>
#include <Compute/ComputeBackend.hpp>

#include <Core/Logger.hpp>

#if defined(YSQ_COMPUTE_HAS_OPENGL)
#include <Compute/OpenGL/OpenGLBackend.hpp>
#endif
#if defined(YSQ_COMPUTE_HAS_CUDA)
#include <Compute/CUDA/CudaBackend.hpp>
#endif
#if defined(YSQ_COMPUTE_HAS_VULKAN)
#include <Compute/Vulkan/VulkanBackend.hpp>
#endif

#include <array>

namespace ysq {

std::string_view toString(ComputeBackendKind kind) noexcept {
    switch (kind) {
        case ComputeBackendKind::OpenGL:
            return "OpenGL";
        case ComputeBackendKind::Cuda:
            return "CUDA";
        case ComputeBackendKind::Vulkan:
            return "Vulkan";
        case ComputeBackendKind::Cpu:
            break;
    }
    return "CPU";
}

namespace {

std::unique_ptr<ComputeBackend> createBackend(ComputeBackendKind kind) {
    switch (kind) {
        case ComputeBackendKind::Cpu:
            return CpuBackend::create();
        case ComputeBackendKind::OpenGL:
#if defined(YSQ_COMPUTE_HAS_OPENGL)
            return OpenGLBackend::create();
#else
            return nullptr;
#endif
        case ComputeBackendKind::Cuda:
#if defined(YSQ_COMPUTE_HAS_CUDA)
            return CudaBackend::create();
#else
            return nullptr;
#endif
        case ComputeBackendKind::Vulkan:
#if defined(YSQ_COMPUTE_HAS_VULKAN)
            return VulkanBackend::create();
#else
            return nullptr;
#endif
    }
    return nullptr;
}

}  // namespace

bool computeBackendAvailable(ComputeBackendKind kind) {
    return createBackend(kind) != nullptr;
}

std::unique_ptr<ComputeBackend>
selectComputeBackend(std::optional<ComputeBackendKind> forceBackend) {
    if (forceBackend) {
        std::unique_ptr<ComputeBackend> backend = createBackend(*forceBackend);
        if (!backend) {
            log::debug("Compute backend {} was forced but is not available",
                       toString(*forceBackend));
        }
        return backend;
    }

    constexpr std::array<ComputeBackendKind, 4> kPriority{
        ComputeBackendKind::Cuda, ComputeBackendKind::Vulkan, ComputeBackendKind::OpenGL,
        ComputeBackendKind::Cpu};
    for (const ComputeBackendKind kind : kPriority) {
        if (std::unique_ptr<ComputeBackend> backend = createBackend(kind)) {
            log::debug("Selected the {} compute backend", toString(kind));
            return backend;
        }
    }
    return nullptr;  // unreachable: CpuBackend::create() always succeeds
}

}  // namespace ysq
