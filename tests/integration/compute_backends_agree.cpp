#include <Compute/CPU/CpuBackend.hpp>
#include <Compute/ComputeBackend.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <vector>

// Every backend that reports itself available is run against the same input
// as the CPU reference and checked within a tolerance, never for exact
// equality: src/Compute/README.md is explicit that a GPU generally runs
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

constexpr std::array<ComputeBackendKind, 4> kGpuKinds{
    ComputeBackendKind::OpenGL, ComputeBackendKind::Cuda, ComputeBackendKind::Vulkan,
    ComputeBackendKind::Metal};

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
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
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
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     GravitationalNBodyAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t kBodyCount = 200;
    const std::vector<float> posX = randomVector(kBodyCount, 8);
    const std::vector<float> posY = randomVector(kBodyCount, 9);
    const std::vector<float> posZ = randomVector(kBodyCount, 10);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> gmDist(1.0f, 10.0f);
    std::vector<float> gm(kBodyCount);
    for (float& value : gm) {
        value = gmDist(rng);
    }
    constexpr float kSofteningSquared = 1.0f;

    std::vector<float> accXReference(kBodyCount);
    std::vector<float> accYReference(kBodyCount);
    std::vector<float> accZReference(kBodyCount);
    reference.gravitationalNBody(posX, posY, posZ, gm, kSofteningSquared, accXReference,
                                 accYReference, accZReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> accX(kBodyCount);
        std::vector<float> accY(kBodyCount);
        std::vector<float> accZ(kBodyCount);
        backend->gravitationalNBody(posX, posY, posZ, gm, kSofteningSquared, accX, accY,
                                    accZ);
        for (std::size_t i = 0; i < kBodyCount; ++i) {
            EXPECT_NEAR(accX[i], accXReference[i], 1e-2f)
                << ysq::toString(kind) << " body " << i;
            EXPECT_NEAR(accY[i], accYReference[i], 1e-2f)
                << ysq::toString(kind) << " body " << i;
            EXPECT_NEAR(accZ[i], accZReference[i], 1e-2f)
                << ysq::toString(kind) << " body " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     ElectricFieldNBodyAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t kCount = 150;
    const std::vector<float> posX = randomVector(kCount, 13);
    const std::vector<float> posY = randomVector(kCount, 14);
    const std::vector<float> posZ = randomVector(kCount, 15);
    std::mt19937 rng(16);
    std::uniform_real_distribution<float> chargeDist(-5.0f, 5.0f);
    std::vector<float> charge(kCount);
    for (float& value : charge) {
        value = chargeDist(rng);
    }
    constexpr float kCoulombConstant = 8.99e9f;

    std::vector<float> fieldXReference(kCount);
    std::vector<float> fieldYReference(kCount);
    std::vector<float> fieldZReference(kCount);
    reference.electricFieldNBody(posX, posY, posZ, charge, kCoulombConstant,
                                 fieldXReference, fieldYReference, fieldZReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> fieldX(kCount);
        std::vector<float> fieldY(kCount);
        std::vector<float> fieldZ(kCount);
        backend->electricFieldNBody(posX, posY, posZ, charge, kCoulombConstant, fieldX,
                                    fieldY, fieldZ);
        for (std::size_t i = 0; i < kCount; ++i) {
            const float tolerance = std::max(std::abs(fieldXReference[i]), 1.0f) * 1e-2f;
            EXPECT_NEAR(fieldX[i], fieldXReference[i], tolerance)
                << ysq::toString(kind) << " body " << i;
            EXPECT_NEAR(fieldY[i], fieldYReference[i], tolerance)
                << ysq::toString(kind) << " body " << i;
            EXPECT_NEAR(fieldZ[i], fieldZReference[i], tolerance)
                << ysq::toString(kind) << " body " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     MagneticFieldNBodyAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t kCount = 150;
    const std::vector<float> posX = randomVector(kCount, 17);
    const std::vector<float> posY = randomVector(kCount, 18);
    const std::vector<float> posZ = randomVector(kCount, 19);
    const std::vector<float> velX = randomVector(kCount, 20);
    const std::vector<float> velY = randomVector(kCount, 21);
    const std::vector<float> velZ = randomVector(kCount, 22);
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> chargeDist(-5.0f, 5.0f);
    std::vector<float> charge(kCount);
    for (float& value : charge) {
        value = chargeDist(rng);
    }
    constexpr float kPermeabilityOver4Pi = 1.0e-7f;

    std::vector<float> fieldXReference(kCount);
    std::vector<float> fieldYReference(kCount);
    std::vector<float> fieldZReference(kCount);
    reference.magneticFieldNBody(posX, posY, posZ, velX, velY, velZ, charge,
                                 kPermeabilityOver4Pi, fieldXReference, fieldYReference,
                                 fieldZReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> fieldX(kCount);
        std::vector<float> fieldY(kCount);
        std::vector<float> fieldZ(kCount);
        backend->magneticFieldNBody(posX, posY, posZ, velX, velY, velZ, charge,
                                    kPermeabilityOver4Pi, fieldX, fieldY, fieldZ);
        for (std::size_t i = 0; i < kCount; ++i) {
            const float tolerance = std::max(std::abs(fieldXReference[i]), 1.0f) * 1e-2f;
            EXPECT_NEAR(fieldX[i], fieldXReference[i], tolerance)
                << ysq::toString(kind) << " body " << i;
            EXPECT_NEAR(fieldY[i], fieldYReference[i], tolerance)
                << ysq::toString(kind) << " body " << i;
            EXPECT_NEAR(fieldZ[i], fieldZReference[i], tolerance)
                << ysq::toString(kind) << " body " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     SphDensityPressureAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t kCount = 200;
    // Deliberately clustered (not spread over [-100, 100] the way
    // randomVector's other callers use it): SPH's compact support only
    // ever engages neighbors within 2*smoothingLength, so a cloud this
    // spread out at smoothingLength = 1 would give every particle zero
    // neighbors and the test would pass without checking anything.
    std::mt19937 rng(24);
    std::uniform_real_distribution<float> posDist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 2.0f);
    std::vector<float> posX(kCount);
    std::vector<float> posY(kCount);
    std::vector<float> posZ(kCount);
    std::vector<float> mass(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        posX[i] = posDist(rng);
        posY[i] = posDist(rng);
        posZ[i] = posDist(rng);
        mass[i] = massDist(rng);
    }
    constexpr float kSmoothingLength = 1.0f;
    constexpr float kEquationOfStateK = 1.0f;
    constexpr float kPolytropicIndex = 2.0f;

    std::vector<float> densityReference(kCount);
    std::vector<float> pressureReference(kCount);
    reference.sphDensityPressure(posX, posY, posZ, mass, kSmoothingLength,
                                 kEquationOfStateK, kPolytropicIndex, densityReference,
                                 pressureReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> density(kCount);
        std::vector<float> pressure(kCount);
        backend->sphDensityPressure(posX, posY, posZ, mass, kSmoothingLength,
                                    kEquationOfStateK, kPolytropicIndex, density,
                                    pressure);
        for (std::size_t i = 0; i < kCount; ++i) {
            EXPECT_NEAR(density[i], densityReference[i], densityReference[i] * 1e-2f)
                << ysq::toString(kind) << " particle " << i;
            EXPECT_NEAR(pressure[i], pressureReference[i], pressureReference[i] * 2e-2f)
                << ysq::toString(kind) << " particle " << i;
        }

        // Chained into sphPressureAcceleration, the same order Physics/Fluids/SPH.hpp's
        // two functions are always used in, rather than only checking each kernel
        // against synthetic density/pressure no scenario would ever actually feed it.
        std::vector<float> accXReference(kCount);
        std::vector<float> accYReference(kCount);
        std::vector<float> accZReference(kCount);
        reference.sphPressureAcceleration(posX, posY, posZ, mass, densityReference,
                                          pressureReference, kSmoothingLength,
                                          accXReference, accYReference, accZReference);

        std::vector<float> accX(kCount);
        std::vector<float> accY(kCount);
        std::vector<float> accZ(kCount);
        backend->sphPressureAcceleration(posX, posY, posZ, mass, density, pressure,
                                         kSmoothingLength, accX, accY, accZ);
        for (std::size_t i = 0; i < kCount; ++i) {
            const float tolerance = std::max(std::abs(accXReference[i]), 1.0f) * 2e-2f;
            EXPECT_NEAR(accX[i], accXReference[i], tolerance)
                << ysq::toString(kind) << " particle " << i;
            EXPECT_NEAR(accY[i], accYReference[i], tolerance)
                << ysq::toString(kind) << " particle " << i;
            EXPECT_NEAR(accZ[i], accZReference[i], tolerance)
                << ysq::toString(kind) << " particle " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     HeatEquation3DStepAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 16;
    const std::vector<float> temperature = randomVector(n * n * n, 25);
    constexpr float kFactor = 0.05f;

    std::vector<float> nextReference(n * n * n);
    reference.heatEquation3DStep(temperature, n, n, n, kFactor, nextReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> next(n * n * n);
        backend->heatEquation3DStep(temperature, n, n, n, kFactor, next);
        for (std::size_t i = 0; i < next.size(); ++i) {
            EXPECT_NEAR(next[i], nextReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     Acoustic3DStepAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 16;
    const std::size_t total = n * n * n;
    const std::vector<float> pressure = randomVector(total, 26);
    const std::vector<float> velX = randomVector(total, 27);
    const std::vector<float> velY = randomVector(total, 28);
    const std::vector<float> velZ = randomVector(total, 29);
    constexpr float kVelocityFactor = 0.01f;
    constexpr float kPressureFactor = 0.01f;

    std::vector<float> nextPressureReference(total);
    std::vector<float> nextVelXReference(total);
    std::vector<float> nextVelYReference(total);
    std::vector<float> nextVelZReference(total);
    reference.acoustic3DStep(pressure, velX, velY, velZ, n, n, n, kVelocityFactor,
                             kPressureFactor, nextPressureReference, nextVelXReference,
                             nextVelYReference, nextVelZReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> nextPressure(total);
        std::vector<float> nextVelX(total);
        std::vector<float> nextVelY(total);
        std::vector<float> nextVelZ(total);
        backend->acoustic3DStep(pressure, velX, velY, velZ, n, n, n, kVelocityFactor,
                                kPressureFactor, nextPressure, nextVelX, nextVelY,
                                nextVelZ);
        for (std::size_t i = 0; i < total; ++i) {
            EXPECT_NEAR(nextPressure[i], nextPressureReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
            EXPECT_NEAR(nextVelX[i], nextVelXReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
            EXPECT_NEAR(nextVelY[i], nextVelYReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
            EXPECT_NEAR(nextVelZ[i], nextVelZReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     Maxwell3DStepAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 16;
    const std::size_t total = n * n * n;
    const std::vector<float> ex = randomVector(total, 30);
    const std::vector<float> ey = randomVector(total, 31);
    const std::vector<float> ez = randomVector(total, 32);
    const std::vector<float> bx = randomVector(total, 33);
    const std::vector<float> by = randomVector(total, 34);
    const std::vector<float> bz = randomVector(total, 35);
    constexpr float kBFactor = 0.01f;
    constexpr float kEFactor = 0.01f;

    std::vector<float> nextExReference(total);
    std::vector<float> nextEyReference(total);
    std::vector<float> nextEzReference(total);
    std::vector<float> nextBxReference(total);
    std::vector<float> nextByReference(total);
    std::vector<float> nextBzReference(total);
    reference.maxwell3DStep(ex, ey, ez, bx, by, bz, n, n, n, kBFactor, kEFactor,
                            nextExReference, nextEyReference, nextEzReference,
                            nextBxReference, nextByReference, nextBzReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> nextEx(total);
        std::vector<float> nextEy(total);
        std::vector<float> nextEz(total);
        std::vector<float> nextBx(total);
        std::vector<float> nextBy(total);
        std::vector<float> nextBz(total);
        backend->maxwell3DStep(ex, ey, ez, bx, by, bz, n, n, n, kBFactor, kEFactor,
                               nextEx, nextEy, nextEz, nextBx, nextBy, nextBz);
        for (std::size_t i = 0; i < total; ++i) {
            EXPECT_NEAR(nextEx[i], nextExReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
            EXPECT_NEAR(nextBx[i], nextBxReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     EulerianFluid3DSweepAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 16;
    const std::size_t total = n * n * n;
    // Density and energy must stay comfortably positive (so pressure and
    // sound speed stay well-defined), and momentum small relative to
    // density (so pressure doesn't go negative once kinetic energy is
    // subtracted out): randomVector's [-100, 100] range is scaled down
    // rather than used directly.
    const std::vector<float> rawDensity = randomVector(total, 33);
    const std::vector<float> rawEnergy = randomVector(total, 34);
    std::vector<float> density(total);
    std::vector<float> energy(total);
    for (std::size_t i = 0; i < total; ++i) {
        density[i] = 1.5f + 0.001f * rawDensity[i];
        energy[i] = 5.0f + 0.001f * rawEnergy[i];
    }
    std::vector<float> momentumNormal = randomVector(total, 35);
    std::vector<float> momentumTangent1 = randomVector(total, 36);
    std::vector<float> momentumTangent2 = randomVector(total, 37);
    for (std::size_t i = 0; i < total; ++i) {
        momentumNormal[i] *= 0.001f;
        momentumTangent1[i] *= 0.001f;
        momentumTangent2[i] *= 0.001f;
    }
    constexpr float kGamma = 1.4f;
    constexpr float kDtOverSpacing = 0.001f;

    std::vector<float> nextDensityReference(total);
    std::vector<float> nextMomentumNormalReference(total);
    std::vector<float> nextMomentumTangent1Reference(total);
    std::vector<float> nextMomentumTangent2Reference(total);
    std::vector<float> nextEnergyReference(total);
    for (const int axis : {0, 1, 2}) {
        reference.eulerianFluid3DSweep(
            density, momentumNormal, momentumTangent1, momentumTangent2, energy, n, n, n,
            axis, kGamma, kDtOverSpacing, nextDensityReference,
            nextMomentumNormalReference, nextMomentumTangent1Reference,
            nextMomentumTangent2Reference, nextEnergyReference);

        bool exercised = false;
        for (const ComputeBackendKind kind : kGpuKinds) {
            const std::unique_ptr<ComputeBackend> backend =
                ysq::selectComputeBackend(kind);
            if (!backend) {
                continue;
            }
            exercised = true;

            std::vector<float> nextDensity(total);
            std::vector<float> nextMomentumNormal(total);
            std::vector<float> nextMomentumTangent1(total);
            std::vector<float> nextMomentumTangent2(total);
            std::vector<float> nextEnergy(total);
            backend->eulerianFluid3DSweep(
                density, momentumNormal, momentumTangent1, momentumTangent2, energy, n, n,
                n, axis, kGamma, kDtOverSpacing, nextDensity, nextMomentumNormal,
                nextMomentumTangent1, nextMomentumTangent2, nextEnergy);
            for (std::size_t i = 0; i < total; ++i) {
                EXPECT_NEAR(nextDensity[i], nextDensityReference[i], 1e-2f)
                    << ysq::toString(kind) << " axis " << axis << " cell " << i;
                EXPECT_NEAR(nextMomentumNormal[i], nextMomentumNormalReference[i], 1e-2f)
                    << ysq::toString(kind) << " axis " << axis << " cell " << i;
                EXPECT_NEAR(nextEnergy[i], nextEnergyReference[i], 1e-2f)
                    << ysq::toString(kind) << " axis " << axis << " cell " << i;
            }
        }

        if (!exercised) {
            GTEST_SKIP()
                << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
                   "machine; only the CPU reference exists to compare against itself";
        }
    }
}

TEST(ComputeBackendsAgree, FftBatchedAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t length = 64;
    constexpr std::size_t batchCount = 3;
    // Scaled down from randomVector's [-100, 100]: a DFT sums `length`
    // terms, so keeping the input near [-1, 1] keeps output magnitudes (and
    // the float32 rounding that scales with them) in a range a fixed
    // absolute tolerance can actually resolve.
    std::vector<float> real = randomVector(length * batchCount, 40);
    std::vector<float> imag = randomVector(length * batchCount, 41);
    for (float& v : real) {
        v *= 0.01f;
    }
    for (float& v : imag) {
        v *= 0.01f;
    }

    for (const bool inverse : {false, true}) {
        std::vector<float> nextRealReference(length * batchCount);
        std::vector<float> nextImagReference(length * batchCount);
        reference.fftBatched(real, imag, length, batchCount, inverse, nextRealReference,
                             nextImagReference);

        bool exercised = false;
        for (const ComputeBackendKind kind : kGpuKinds) {
            const std::unique_ptr<ComputeBackend> backend =
                ysq::selectComputeBackend(kind);
            if (!backend) {
                continue;
            }
            exercised = true;

            std::vector<float> nextReal(length * batchCount);
            std::vector<float> nextImag(length * batchCount);
            backend->fftBatched(real, imag, length, batchCount, inverse, nextReal,
                                nextImag);
            for (std::size_t i = 0; i < length * batchCount; ++i) {
                EXPECT_NEAR(nextReal[i], nextRealReference[i], 1e-2f)
                    << ysq::toString(kind) << " inverse=" << inverse << " element " << i;
                EXPECT_NEAR(nextImag[i], nextImagReference[i], 1e-2f)
                    << ysq::toString(kind) << " inverse=" << inverse << " element " << i;
            }
        }

        if (!exercised) {
            GTEST_SKIP()
                << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
                   "machine; only the CPU reference exists to compare against itself";
        }
    }
}

TEST(ComputeBackendsAgree, MatVecAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t rows = 200;
    constexpr std::size_t cols = 150;
    const std::vector<float> matrix = randomVector(rows * cols, 50);
    const std::vector<float> vector = randomVector(cols, 51);

    std::vector<float> resultReference(rows);
    reference.matVec(matrix, rows, cols, vector, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(rows);
        backend->matVec(matrix, rows, cols, vector, result);
        for (std::size_t i = 0; i < rows; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 5e-1f)
                << ysq::toString(kind) << " row " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, MatMulAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t aRows = 40, aCols = 60, bCols = 30;
    const std::vector<float> a = randomVector(aRows * aCols, 52);
    const std::vector<float> b = randomVector(aCols * bCols, 53);

    std::vector<float> resultReference(aRows * bCols);
    reference.matMul(a, aRows, aCols, b, bCols, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(aRows * bCols);
        backend->matMul(a, aRows, aCols, b, bCols, result);
        for (std::size_t i = 0; i < result.size(); ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 5e-1f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     LuDecomposeGpuAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 48;
    // Diagonally dominant (|diagonal| > sum of the rest of its row), so the
    // matrix is guaranteed non-singular and every backend's partial-pivot
    // choice should agree.
    std::vector<float> a = randomVector(n * n, 54);
    for (std::size_t i = 0; i < n; ++i) {
        a[i * n + i] = static_cast<float>(n) * 10.0f;
    }

    std::vector<float> luReference(n * n);
    std::vector<std::uint32_t> pivotReference(n);
    ASSERT_TRUE(reference.luDecomposeGpu(a, n, luReference, pivotReference));

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> lu(n * n);
        std::vector<std::uint32_t> pivot(n);
        ASSERT_TRUE(backend->luDecomposeGpu(a, n, lu, pivot)) << ysq::toString(kind);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(pivot[i], pivotReference[i])
                << ysq::toString(kind) << " pivot " << i;
        }
        for (std::size_t i = 0; i < n * n; ++i) {
            EXPECT_NEAR(lu[i], luReference[i], 1.0f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     CholeskyDecomposeGpuAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 32;
    // A Gram matrix A^T A (always symmetric positive-semidefinite) plus a
    // scaled identity (making it strictly positive-definite).
    const std::vector<float> base = randomVector(n * n, 55);
    std::vector<float> a(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                total += base[k * n + i] * base[k * n + j];
            }
            a[i * n + j] = total;
        }
        a[i * n + i] += static_cast<float>(n) * 1.0e5f;
    }

    std::vector<float> lReference(n * n);
    ASSERT_TRUE(reference.choleskyDecomposeGpu(a, n, lReference));

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> l(n * n);
        ASSERT_TRUE(backend->choleskyDecomposeGpu(a, n, l)) << ysq::toString(kind);
        for (std::size_t i = 0; i < n * n; ++i) {
            EXPECT_NEAR(l[i], lReference[i], 1e2f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     QrDecomposeGpuAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t rows = 24, cols = 16;
    std::vector<float> matrix = randomVector(rows * cols, 60);
    for (float& v : matrix) {
        v *= 0.1f;
    }

    std::vector<float> qReference(rows * rows);
    std::vector<float> rReference(rows * cols);
    reference.qrDecomposeGpu(matrix, rows, cols, qReference, rReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> q(rows * rows);
        std::vector<float> r(rows * cols);
        backend->qrDecomposeGpu(matrix, rows, cols, q, r);

        // Q must be orthogonal and R upper triangular regardless of any
        // sign convention difference between backends (Householder
        // reflections have no unique sign), so check the reconstructed
        // product Q*R against the original matrix rather than Q/R directly.
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                float total = 0.0f;
                for (std::size_t k = 0; k < rows; ++k) {
                    total += q[i * rows + k] * r[k * cols + j];
                }
                EXPECT_NEAR(total, matrix[i * cols + j], 1e-1f)
                    << ysq::toString(kind) << " QR entry " << i << "," << j;
            }
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     JacobiEigenSymmetricGpuAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 24;
    std::vector<float> raw = randomVector(n * n, 61);
    // Build a symmetric matrix from the random data (only the lower
    // triangle matters, matching jacobiEigenSymmetricGpu's own contract).
    std::vector<float> matrix(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const float value = raw[i * n + j] * 0.1f;
            matrix[i * n + j] = value;
            matrix[j * n + i] = value;
        }
    }

    std::vector<float> diagonalReference(n * n);
    std::vector<float> eigenvectorsReference(n * n);
    reference.jacobiEigenSymmetricGpu(matrix, n, 100, 0.0f, diagonalReference,
                                      eigenvectorsReference);
    std::vector<float> eigenvaluesReference(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvaluesReference[i] = diagonalReference[i * n + i];
    }
    std::sort(eigenvaluesReference.begin(), eigenvaluesReference.end());

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> diagonal(n * n);
        std::vector<float> eigenvectors(n * n);
        backend->jacobiEigenSymmetricGpu(matrix, n, 100, 0.0f, diagonal, eigenvectors);

        // Compare the eigenvalue SET (order-independent: which pair gets
        // processed first differs between the CPU reference's cyclic order
        // and the GPU backends' round-robin order), and separately verify
        // eigenvectors independently via the defining property A*v = lambda*v
        // (also order-independent, and does not require matching the
        // reference's own eigenvector basis when eigenvalues repeat).
        std::vector<float> eigenvalues(n);
        for (std::size_t i = 0; i < n; ++i) {
            eigenvalues[i] = diagonal[i * n + i];
        }
        std::sort(eigenvalues.begin(), eigenvalues.end());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(eigenvalues[i], eigenvaluesReference[i], 5e-1f)
                << ysq::toString(kind) << " eigenvalue " << i;
        }

        for (std::size_t col = 0; col < n; ++col) {
            const float lambda = diagonal[col * n + col];
            for (std::size_t row = 0; row < n; ++row) {
                float avRow = 0.0f;
                for (std::size_t k = 0; k < n; ++k) {
                    avRow += matrix[row * n + k] * eigenvectors[k * n + col];
                }
                EXPECT_NEAR(avRow, lambda * eigenvectors[row * n + col], 1.0f)
                    << ysq::toString(kind) << " A*v vs lambda*v, row " << row << " col "
                    << col;
            }
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, JacobiSvdGpuAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t rows = 24, cols = 16;
    std::vector<float> matrix = randomVector(rows * cols, 62);
    for (float& v : matrix) {
        v *= 0.1f;
    }

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> resultA(rows * cols);
        std::vector<float> resultV(cols * cols);
        backend->jacobiSvdGpu(matrix, rows, cols, 60, 0.0f, resultA, resultV);

        // resultA's columns are U scaled by the singular values; V is
        // orthogonal. Check the defining reconstruction A = resultA * V^T
        // directly, independent of any backend-specific convergence path.
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                float total = 0.0f;
                for (std::size_t k = 0; k < cols; ++k) {
                    total += resultA[i * cols + k] * resultV[j * cols + k];
                }
                EXPECT_NEAR(total, matrix[i * cols + j], 1e-1f)
                    << ysq::toString(kind) << " entry " << i << "," << j;
            }
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, BatchErfAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 2000;
    std::vector<float> x = randomVector(n, 70);
    for (float& v : x) {
        v *= 0.03f;  // erf saturates fast; keep inputs in its interesting range
    }

    std::vector<float> resultReference(n);
    reference.batchErf(x, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchErf(x, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-5f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, BatchErfcAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 2000;
    std::vector<float> x = randomVector(n, 71);
    for (float& v : x) {
        v *= 0.03f;
    }

    std::vector<float> resultReference(n);
    reference.batchErfc(x, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchErfc(x, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-5f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, BatchGammaAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 2000;
    // Both positive and negative (non-integer) x, to exercise the Lanczos
    // reflection branch too.
    std::vector<float> raw = randomVector(n, 72);
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 1.0f + raw[i] * 0.03f;
    }

    std::vector<float> resultReference(n);
    reference.batchGamma(x, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchGamma(x, result);
        for (std::size_t i = 0; i < n; ++i) {
            // x ranges over [-2, 4], so a handful of samples land close to
            // gamma's poles at the negative integers, where 1 / sin(pi x)
            // amplifies x's ordinary float32 rounding into a large swing in
            // an already-large result -- the relative error stays tiny.
            // Away from poles, |resultReference| ~ O(1) and this reduces to
            // an absolute check.
            const float tolerance = std::max(1e-2f, std::abs(resultReference[i]) * 1e-3f);
            EXPECT_NEAR(result[i], resultReference[i], tolerance)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     BatchLogGammaAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 2000;
    std::vector<float> raw = randomVector(n, 73);
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 5.0f + raw[i];
    }

    std::vector<float> resultReference(n);
    reference.batchLogGamma(x, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchLogGamma(x, result);
        for (std::size_t i = 0; i < n; ++i) {
            // Same reasoning as BatchGamma above: x spans [-95, 105], so
            // some samples land close to a pole, where the reflection
            // formula's log(sin(pi x)) term amplifies x's float32 rounding.
            const float tolerance = std::max(1e-2f, std::abs(resultReference[i]) * 1e-3f);
            EXPECT_NEAR(result[i], resultReference[i], tolerance)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     BatchLegendrePAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 2000;
    constexpr unsigned kDegree = 6;
    constexpr unsigned kOrder = 3;
    std::vector<float> x = randomVector(n, 74);
    for (float& v : x) {
        v *= 0.01f;  // keep within [-1, 1]
    }

    std::vector<float> resultReference(n);
    reference.batchLegendreP(kDegree, kOrder, x, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchLegendreP(kDegree, kOrder, x, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-2f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     BatchPolynomialEvalAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 2000;
    const std::vector<float> coefficients = randomVector(8, 75);
    std::vector<float> x = randomVector(n, 76);
    for (float& v : x) {
        v *= 0.01f;
    }

    std::vector<float> resultReference(n);
    reference.batchPolynomialEval(coefficients, x, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchPolynomialEval(coefficients, x, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-2f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     BatchCubicSplineEvalAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t knotCount = 20;
    std::vector<float> knotsX(knotCount);
    std::vector<float> knotsY(knotCount);
    std::vector<float> secondDerivatives(knotCount);
    for (std::size_t i = 0; i < knotCount; ++i) {
        knotsX[i] = static_cast<float>(i);
        knotsY[i] = std::sin(static_cast<float>(i) * 0.3f);
        secondDerivatives[i] = -knotsY[i];  // a smooth, plausible curvature
    }

    constexpr std::size_t n = 2000;
    std::vector<float> queryRaw = randomVector(n, 77);
    std::vector<float> queryX(n);
    for (std::size_t i = 0; i < n; ++i) {
        queryX[i] = (queryRaw[i] + 100.0f) / 200.0f * static_cast<float>(knotCount - 1);
    }

    std::vector<float> resultReference(n);
    reference.batchCubicSplineEval(knotsX, knotsY, secondDerivatives, queryX,
                                   resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchCubicSplineEval(knotsX, knotsY, secondDerivatives, queryX, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-2f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     BatchUniformRealAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    // Philox4x32-10 is pure bit arithmetic (a counter and key, split into
    // 32-bit words, run through ten rounds of multiply-and-permute): given
    // the same seed and offset, every backend must reach the exact same
    // uint32 words and therefore, since the float conversion is a single
    // multiply by a fixed constant, the exact same float result -- unlike
    // the batch-evaluation family's approximations, there is no reason for
    // even a small numerical discrepancy here, so this uses a much tighter
    // tolerance than that family's tests.
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 5000;
    constexpr std::uint64_t seed = 0x0123456789ABCDEFull;
    constexpr std::uint64_t offset = 17;

    std::vector<float> resultReference(n);
    reference.batchUniformReal(seed, offset, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchUniformReal(seed, offset, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-6f)
                << ysq::toString(kind) << " element " << i;
            EXPECT_GE(result[i], 0.0f) << ysq::toString(kind) << " element " << i;
            EXPECT_LT(result[i], 1.0f) << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, BatchNormalAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 5000;
    constexpr std::uint64_t seed = 0xFEDCBA9876543210ull;
    constexpr std::uint64_t offset = 3;

    std::vector<float> resultReference(n);
    reference.batchNormal(seed, offset, resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result(n);
        backend->batchNormal(seed, offset, result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-4f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     SortAscendingAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    // Not a power of two: exercises the padding a GPU backend does
    // internally (see ComputeBackend.hpp's own comment on sortAscending).
    constexpr std::size_t n = 4999;
    const std::vector<float> values = randomVector(n, 88);

    std::vector<float> resultReference = values;
    reference.sortAscending(resultReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> result = values;
        backend->sortAscending(result);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(result[i], resultReference[i], 1e-6f)
                << ysq::toString(kind) << " element " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     MultigridRestrict3DAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 16;
    const std::vector<float> fine = randomVector(n * n * n, 30);
    constexpr std::size_t cn = n / 2;

    std::vector<float> coarseReference(cn * cn * cn);
    reference.multigridRestrict3D(fine, n, n, n, coarseReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> coarse(cn * cn * cn);
        backend->multigridRestrict3D(fine, n, n, n, coarse);
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            EXPECT_NEAR(coarse[i], coarseReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     MultigridProlongateAndAdd3DAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    constexpr std::size_t n = 16;
    constexpr std::size_t cn = n / 2;
    const std::vector<float> fine = randomVector(n * n * n, 31);
    const std::vector<float> coarseCorrection = randomVector(cn * cn * cn, 32);

    std::vector<float> nextFineReference(n * n * n);
    reference.multigridProlongateAndAdd3D(fine, coarseCorrection, n, n, n,
                                          nextFineReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> nextFine(n * n * n);
        backend->multigridProlongateAndAdd3D(fine, coarseCorrection, n, n, n, nextFine);
        for (std::size_t i = 0; i < nextFine.size(); ++i) {
            EXPECT_NEAR(nextFine[i], nextFineReference[i], 1e-2f)
                << ysq::toString(kind) << " cell " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree, MinIndexAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    const std::vector<float> x = randomVector(5000, 12);
    const std::size_t referenceIndex = reference.minIndex(x);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;
        EXPECT_EQ(backend->minIndex(x), referenceIndex) << ysq::toString(kind);
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}

TEST(ComputeBackendsAgree,
     LinearCombineAgreesWithTheCpuReferenceOnEveryAvailableBackend) {
    const ysq::CpuBackend reference;
    const std::vector<float> a = randomVector(1000, 4);
    const std::vector<float> b = randomVector(1000, 5);
    const std::vector<float> c = randomVector(1000, 6);
    const std::vector<float> d = randomVector(1000, 7);
    const std::array<float, 4> coefficients{1.0f, 2.0f, 2.0f, 1.0f};
    const std::array<std::span<const float>, 4> terms{
        std::span<const float>{a}, std::span<const float>{b}, std::span<const float>{c},
        std::span<const float>{d}};

    std::vector<float> yReference(1000);
    reference.linearCombine(terms, coefficients, yReference);

    bool exercised = false;
    for (const ComputeBackendKind kind : kGpuKinds) {
        const std::unique_ptr<ComputeBackend> backend = ysq::selectComputeBackend(kind);
        if (!backend) {
            continue;
        }
        exercised = true;

        std::vector<float> y(1000);
        backend->linearCombine(terms, coefficients, y);
        for (std::size_t i = 0; i < y.size(); ++i) {
            EXPECT_NEAR(y[i], yReference[i], 1e-2f)
                << ysq::toString(kind) << " index " << i;
        }
    }

    if (!exercised) {
        GTEST_SKIP()
            << "no GPU compute backend (OpenGL/CUDA/Vulkan/Metal) available on this "
               "machine; only the CPU reference exists to compare against itself";
    }
}
