#include <Compute/CUDA/CudaBackend.hpp>

#include <Core/Logger.hpp>

#include <cuda_runtime.h>

#include <cassert>
#include <memory>

namespace ysq {

std::unique_ptr<ComputeBackend> CudaBackend::create() {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
        logging::debug("CUDA compute backend unavailable: {}",
                       status == cudaSuccess ? "no device" : cudaGetErrorString(status));
        return nullptr;
    }
    return std::make_unique<CudaBackend>();
}

void CudaBackend::saxpy(std::span<const float>, std::span<float>, float) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

float CudaBackend::sum(std::span<const float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
    return 0.0f;
}

}  // namespace ysq
