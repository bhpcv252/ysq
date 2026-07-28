#include <Compute/CPU/CpuBackend.hpp>
#include <Compute/ComputeBackend.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

// Every backend that reports itself available is run against the same input
// as the CPU reference and checked within a tolerance, never for exact
// equality: docs/architecture.md is explicit that a GPU generally runs
// float32 arithmetic in a different order than the CPU, so bit-identical
// output is not the bar. This is the test that makes "a kernel produces
// matching results on every available backend" a checked claim.
//
// On a machine with no GPU and no CUDA/Vulkan SDK, the loop below finds
// nothing to compare and the test reports itself skipped rather than passing
// on having checked nothing; see tests/README.md on a file of tests that
// always skip.

namespace {

using ysq::ComputeBackend;
using ysq::ComputeBackendKind;

constexpr std::array<ComputeBackendKind, 3> kGpuKinds{
    ComputeBackendKind::OpenGL, ComputeBackendKind::Cuda, ComputeBackendKind::Vulkan};

std::vector<float> randomVector(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> values(n);
    for (float& v : values) {
        v = dist(rng);
    }
    return values;
}

}  // namespace

TEST(ComputeBackendsAgree, SaxpyAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    const std::vector<float> x = randomVector(1000, 1);
    std::vector<float> yReference = randomVector(1000, 2);
    reference.saxpy(x, yReference, 1.5f);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> y = randomVector(1000, 2);
        backend->saxpy(x, y, 1.5f);
        for (std::size_t i = 0; i < y.size(); ++i) {
            EXPECT_NEAR(y[i], yReference[i], 1e-3f)
                << ysq::toString(kind) << " index " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, SumAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    const std::vector<float> x = randomVector(10000, 3);
    const float referenceSum = reference.sum(x);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        // Scaled by N and the input magnitude rather than a fixed epsilon,
        // since a fixed tolerance chosen for one input size says nothing
        // about another. Generous on purpose: it has to cover whatever
        // reduction shape a given GPU backend actually uses (a tree
        // reduction, which is what the OpenGL kernel does, accumulates far
        // less error than this bounds), and this has not been checked
        // against real GPU hardware yet; see src/Compute/README.md.
        const float tolerance = std::sqrt(10000.0f) * 100.0f * 1e-5f;
        EXPECT_NEAR(backend->sum(x), referenceSum, tolerance) << ysq::toString(kind);
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}
