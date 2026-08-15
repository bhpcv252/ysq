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

void CudaBackend::linearCombine(std::span<const std::span<const float>>,
                                std::span<const float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::gravitationalNBody(std::span<const float>, std::span<const float>,
                                     std::span<const float>, std::span<const float>,
                                     float, std::span<float>, std::span<float>,
                                     std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::electricFieldNBody(std::span<const float>, std::span<const float>,
                                     std::span<const float>, std::span<const float>,
                                     float, std::span<float>, std::span<float>,
                                     std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::magneticFieldNBody(std::span<const float>, std::span<const float>,
                                     std::span<const float>, std::span<const float>,
                                     std::span<const float>, std::span<const float>,
                                     std::span<const float>, float, std::span<float>,
                                     std::span<float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::sphDensityPressure(std::span<const float>, std::span<const float>,
                                     std::span<const float>, std::span<const float>,
                                     float, float, float, std::span<float>,
                                     std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::sphPressureAcceleration(std::span<const float>, std::span<const float>,
                                          std::span<const float>, std::span<const float>,
                                          std::span<const float>, std::span<const float>,
                                          float, std::span<float>, std::span<float>,
                                          std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::heatEquation3DStep(std::span<const float>, std::size_t, std::size_t,
                                     std::size_t, float, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::acoustic3DStep(std::span<const float>, std::span<const float>,
                                 std::span<const float>, std::span<const float>,
                                 std::size_t, std::size_t, std::size_t, float, float,
                                 std::span<float>, std::span<float>, std::span<float>,
                                 std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::maxwell3DStep(std::span<const float>, std::span<const float>,
                                std::span<const float>, std::span<const float>,
                                std::span<const float>, std::span<const float>,
                                std::size_t, std::size_t, std::size_t, float, float,
                                std::span<float>, std::span<float>, std::span<float>,
                                std::span<float>, std::span<float>,
                                std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::eulerianFluid3DSweep(std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::size_t, std::size_t,
                                       std::size_t, int, float, float, std::span<float>,
                                       std::span<float>, std::span<float>,
                                       std::span<float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::fftBatched(std::span<const float>, std::span<const float>, std::size_t,
                             std::size_t, bool, std::span<float>,
                             std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::matVec(std::span<const float>, std::size_t, std::size_t,
                         std::span<const float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::matMul(std::span<const float>, std::size_t, std::size_t,
                         std::span<const float>, std::size_t, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

bool CudaBackend::luDecomposeGpu(std::span<const float>, std::size_t, std::span<float>,
                                 std::span<std::uint32_t>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
    return false;
}

bool CudaBackend::choleskyDecomposeGpu(std::span<const float>, std::size_t,
                                       std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
    return false;
}

void CudaBackend::qrDecomposeGpu(std::span<const float>, std::size_t, std::size_t,
                                 std::span<float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::jacobiEigenSymmetricGpu(std::span<const float>, std::size_t, int, float,
                                          std::span<float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::jacobiSvdGpu(std::span<const float>, std::size_t, std::size_t, int,
                               float, std::span<float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchErf(std::span<const float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchErfc(std::span<const float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchGamma(std::span<const float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchLogGamma(std::span<const float>, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchLegendreP(unsigned, unsigned, std::span<const float>,
                                 std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchPolynomialEval(std::span<const float>, std::span<const float>,
                                      std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchCubicSplineEval(std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchUniformReal(std::uint64_t, std::uint64_t, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::batchNormal(std::uint64_t, std::uint64_t, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::sortAscending(std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::multigridRestrict3D(std::span<const float>, std::size_t, std::size_t,
                                      std::size_t, std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

void CudaBackend::multigridProlongateAndAdd3D(std::span<const float>,
                                              std::span<const float>, std::size_t,
                                              std::size_t, std::size_t,
                                              std::span<float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
}

std::size_t CudaBackend::minIndex(std::span<const float>) const {
    assert(false && "CudaBackend kernels are not implemented yet; see Compute/README.md");
    return 0;
}

}  // namespace ysq
