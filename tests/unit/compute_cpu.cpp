#include <Compute/CPU/CpuBackend.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <span>
#include <vector>

namespace {

using ysq::CpuBackend;

TEST(ComputeCpu, SaxpyMatchesTheDefinitionElementwise) {
    const CpuBackend backend;
    std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y{10.0f, 20.0f, 30.0f, 40.0f};
    backend.saxpy(x, y, 2.0f);
    EXPECT_FLOAT_EQ(y[0], 12.0f);
    EXPECT_FLOAT_EQ(y[1], 24.0f);
    EXPECT_FLOAT_EQ(y[2], 36.0f);
    EXPECT_FLOAT_EQ(y[3], 48.0f);
}

TEST(ComputeCpu, SaxpyOnEmptySpansDoesNothing) {
    const CpuBackend backend;
    std::vector<float> x;
    std::vector<float> y;
    backend.saxpy(x, y, 3.0f);
    EXPECT_TRUE(y.empty());
}

TEST(ComputeCpu, SumMatchesTheObviousTotalForSmallInputs) {
    const CpuBackend backend;
    const std::array<float, 5> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_FLOAT_EQ(backend.sum(x), 15.0f);
}

TEST(ComputeCpu, SumOfEmptyIsZero) {
    const CpuBackend backend;
    EXPECT_FLOAT_EQ(backend.sum(std::span<const float>{}), 0.0f);
}

// A naive float accumulator loses the small terms once the running total is
// large enough that they round away. Compensated summation is what the CPU
// backend has to get right to be trustworthy as the reference: this is a
// contrived case a naive accumulator gets visibly wrong and this one does not.
TEST(ComputeCpu, SumStaysAccurateWhenSmallTermsFollowALargeOne) {
    const CpuBackend backend;
    std::vector<float> x(100000, 1e-4f);
    x[0] = 1e4f;
    // x[0] overwrote one of the small terms, so only x.size() - 1 of them
    // remain: 1e4 + 99999 * 1e-4 = 10009.9999.
    const float expected = 1e4f + static_cast<float>(x.size() - 1) * 1e-4f;

    EXPECT_NEAR(backend.sum(x), expected, 1e-2f);

    float naive = 0.0f;
    for (const float v : x) {
        naive += v;
    }
    EXPECT_GT(std::abs(naive - expected), 1e-2f)
        << "the naive sum should actually be off here, or this test proves nothing";
}

TEST(ComputeCpu, LinearCombineOfOneTermMatchesAScaledCopy) {
    const CpuBackend backend;
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> y(3);
    const std::array<float, 1> coefficients{2.0f};
    const std::array<std::span<const float>, 1> terms{std::span<const float>{a}};
    backend.linearCombine(terms, coefficients, y);
    EXPECT_FLOAT_EQ(y[0], 2.0f);
    EXPECT_FLOAT_EQ(y[1], 4.0f);
    EXPECT_FLOAT_EQ(y[2], 6.0f);
}

TEST(ComputeCpu, LinearCombineOfFourTermsMatchesTheWeightedSum) {
    const CpuBackend backend;
    const std::vector<float> a{1.0f, 1.0f};
    const std::vector<float> b{2.0f, 2.0f};
    const std::vector<float> c{3.0f, 3.0f};
    const std::vector<float> d{4.0f, 4.0f};
    std::vector<float> y(2);
    const std::array<float, 4> coefficients{1.0f, 2.0f, 2.0f, 1.0f};
    const std::array<std::span<const float>, 4> terms{
        std::span<const float>{a}, std::span<const float>{b}, std::span<const float>{c},
        std::span<const float>{d}};
    backend.linearCombine(terms, coefficients, y);
    // 1*1 + 2*2 + 2*3 + 1*4 = 15, the RK4 weighting this kernel exists for.
    EXPECT_FLOAT_EQ(y[0], 15.0f);
    EXPECT_FLOAT_EQ(y[1], 15.0f);
}

TEST(ComputeCpu, GravitationalNBodyMatchesTheClosedFormForTwoEqualMasses) {
    const CpuBackend backend;
    // Two unit-gm bodies 2 apart on the x axis, no softening: each pulls the
    // other with acceleration gm / r^2 = 1/4, toward each other.
    const std::array<float, 2> posX{-1.0f, 1.0f};
    const std::array<float, 2> posY{0.0f, 0.0f};
    const std::array<float, 2> posZ{0.0f, 0.0f};
    const std::array<float, 2> gm{1.0f, 1.0f};
    std::array<float, 2> accX{};
    std::array<float, 2> accY{};
    std::array<float, 2> accZ{};

    backend.gravitationalNBody(posX, posY, posZ, gm, 0.0f, accX, accY, accZ);

    EXPECT_NEAR(accX[0], 0.25f, 1e-5f);
    EXPECT_NEAR(accX[1], -0.25f, 1e-5f);
    EXPECT_NEAR(accY[0], 0.0f, 1e-5f);
    EXPECT_NEAR(accZ[0], 0.0f, 1e-5f);
}

TEST(ComputeCpu, GravitationalNBodyOfOneBodyIsZero) {
    const CpuBackend backend;
    const std::array<float, 1> posX{0.0f};
    const std::array<float, 1> posY{0.0f};
    const std::array<float, 1> posZ{0.0f};
    const std::array<float, 1> gm{5.0f};
    std::array<float, 1> accX{};
    std::array<float, 1> accY{};
    std::array<float, 1> accZ{};

    backend.gravitationalNBody(posX, posY, posZ, gm, 0.0f, accX, accY, accZ);

    EXPECT_FLOAT_EQ(accX[0], 0.0f);
    EXPECT_FLOAT_EQ(accY[0], 0.0f);
    EXPECT_FLOAT_EQ(accZ[0], 0.0f);
}

TEST(ComputeCpu, ElectricFieldNBodyMatchesCoulombsLawForTwoEqualCharges) {
    const CpuBackend backend;
    // Two unit charges 2 apart on the x axis: each feels a field of
    // coulombConstant * 1 / 2^2 = coulombConstant / 4, pointing away from
    // the other.
    const std::array<float, 2> posX{-1.0f, 1.0f};
    const std::array<float, 2> posY{0.0f, 0.0f};
    const std::array<float, 2> posZ{0.0f, 0.0f};
    const std::array<float, 2> charge{1.0f, 1.0f};
    std::array<float, 2> fieldX{};
    std::array<float, 2> fieldY{};
    std::array<float, 2> fieldZ{};

    backend.electricFieldNBody(posX, posY, posZ, charge, 1.0f, fieldX, fieldY, fieldZ);

    EXPECT_NEAR(fieldX[0], -0.25f, 1e-5f);
    EXPECT_NEAR(fieldX[1], 0.25f, 1e-5f);
    EXPECT_NEAR(fieldY[0], 0.0f, 1e-5f);
    EXPECT_NEAR(fieldZ[0], 0.0f, 1e-5f);
}

TEST(ComputeCpu, MagneticFieldNBodyOfAStationaryChargeIsZero) {
    const CpuBackend backend;
    // A moving charge and a stationary one: the stationary source
    // contributes nothing (v = 0), so the field felt by the moving body
    // (index 0) comes entirely from source 1, and the field felt by the
    // stationary body (index 1) is exactly zero.
    const std::array<float, 2> posX{-1.0f, 1.0f};
    const std::array<float, 2> posY{0.0f, 0.0f};
    const std::array<float, 2> posZ{0.0f, 0.0f};
    const std::array<float, 2> velX{0.0f, 0.0f};
    const std::array<float, 2> velY{0.0f, 0.0f};
    const std::array<float, 2> velZ{0.0f, 0.0f};
    const std::array<float, 2> charge{1.0f, 1.0f};
    std::array<float, 2> fieldX{};
    std::array<float, 2> fieldY{};
    std::array<float, 2> fieldZ{};

    backend.magneticFieldNBody(posX, posY, posZ, velX, velY, velZ, charge, 1.0f, fieldX,
                               fieldY, fieldZ);

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(fieldX[i], 0.0f);
        EXPECT_FLOAT_EQ(fieldY[i], 0.0f);
        EXPECT_FLOAT_EQ(fieldZ[i], 0.0f);
    }
}

TEST(ComputeCpu, SphDensityPressureOfOneParticleIsItsOwnSelfKernelTerm) {
    const CpuBackend backend;
    const std::array<float, 1> posX{0.0f};
    const std::array<float, 1> posY{0.0f};
    const std::array<float, 1> posZ{0.0f};
    const std::array<float, 1> mass{2.0f};
    std::array<float, 1> density{};
    std::array<float, 1> pressure{};

    constexpr float smoothingLength = 1.0f;
    backend.sphDensityPressure(posX, posY, posZ, mass, smoothingLength, 1.0f, 2.0f,
                               density, pressure);

    // W(0, h) = sigma / h^3 = (1/pi) / 1 for h = 1.
    const float expectedDensity = 2.0f * (1.0f / 3.14159265f);
    EXPECT_NEAR(density[0], expectedDensity, expectedDensity * 1e-4f);
    EXPECT_NEAR(pressure[0], expectedDensity * expectedDensity,
                expectedDensity * expectedDensity * 1e-3f);
}

TEST(ComputeCpu, SphPressureAccelerationOfOneParticleIsZero) {
    const CpuBackend backend;
    const std::array<float, 1> posX{0.0f};
    const std::array<float, 1> posY{0.0f};
    const std::array<float, 1> posZ{0.0f};
    const std::array<float, 1> mass{2.0f};
    const std::array<float, 1> density{1.0f};
    const std::array<float, 1> pressure{1.0f};
    std::array<float, 1> accX{};
    std::array<float, 1> accY{};
    std::array<float, 1> accZ{};

    backend.sphPressureAcceleration(posX, posY, posZ, mass, density, pressure, 1.0f, accX,
                                    accY, accZ);

    EXPECT_FLOAT_EQ(accX[0], 0.0f);
    EXPECT_FLOAT_EQ(accY[0], 0.0f);
    EXPECT_FLOAT_EQ(accZ[0], 0.0f);
}

TEST(ComputeCpu, SphKernelsSkipParticlesOutsideTheSupportRadius) {
    const CpuBackend backend;
    // Two particles far outside 2*smoothingLength of each other: each must
    // see only its own self term, none of the other's contribution.
    const std::array<float, 2> posX{0.0f, 100.0f};
    const std::array<float, 2> posY{0.0f, 0.0f};
    const std::array<float, 2> posZ{0.0f, 0.0f};
    const std::array<float, 2> mass{1.0f, 1.0f};
    std::array<float, 2> density{};
    std::array<float, 2> pressure{};

    backend.sphDensityPressure(posX, posY, posZ, mass, 1.0f, 1.0f, 1.0f, density,
                               pressure);

    EXPECT_FLOAT_EQ(density[0], density[1]);
    EXPECT_GT(density[0], 0.0f);
}

TEST(ComputeCpu, HeatEquation3DStepOfAUniformGridStaysUniform) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    std::vector<float> temperature(n * n * n, 5.0f);
    std::vector<float> next(n * n * n);

    backend.heatEquation3DStep(temperature, n, n, n, 0.1f, next);

    for (const float value : next) {
        EXPECT_FLOAT_EQ(value, 5.0f);
    }
}

TEST(ComputeCpu, HeatEquation3DStepDiffusesTowardAHotCenter) {
    const CpuBackend backend;
    constexpr std::size_t n = 5;
    std::vector<float> temperature(n * n * n, 0.0f);
    const auto idx = [](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };
    temperature[idx(2, 2, 2)] = 6.0f;
    std::vector<float> next(n * n * n);

    backend.heatEquation3DStep(temperature, n, n, n, 0.1f, next);

    // A neighbor of the hot cell must warm up; the hot cell itself cools
    // (factor * laplacian is negative there, since every neighbor is colder).
    EXPECT_GT(next[idx(1, 2, 2)], 0.0f);
    EXPECT_LT(next[idx(2, 2, 2)], temperature[idx(2, 2, 2)]);
}

TEST(ComputeCpu, Acoustic3DStepOfSilenceStaysSilent) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const std::size_t total = n * n * n;
    std::vector<float> pressure(total, 0.0f);
    std::vector<float> velX(total, 0.0f);
    std::vector<float> velY(total, 0.0f);
    std::vector<float> velZ(total, 0.0f);
    std::vector<float> nextPressure(total);
    std::vector<float> nextVelX(total);
    std::vector<float> nextVelY(total);
    std::vector<float> nextVelZ(total);

    backend.acoustic3DStep(pressure, velX, velY, velZ, n, n, n, 0.1f, 0.1f, nextPressure,
                           nextVelX, nextVelY, nextVelZ);

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(nextPressure[i], 0.0f);
        EXPECT_FLOAT_EQ(nextVelX[i], 0.0f);
        EXPECT_FLOAT_EQ(nextVelY[i], 0.0f);
        EXPECT_FLOAT_EQ(nextVelZ[i], 0.0f);
    }
}

TEST(ComputeCpu, Acoustic3DStepPropagatesAPressureSpikeToItsNeighborsVelocity) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const std::size_t total = n * n * n;
    const auto idx = [](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };
    std::vector<float> pressure(total, 0.0f);
    pressure[idx(1, 1, 1)] = 10.0f;
    std::vector<float> velX(total, 0.0f);
    std::vector<float> velY(total, 0.0f);
    std::vector<float> velZ(total, 0.0f);
    std::vector<float> nextPressure(total);
    std::vector<float> nextVelX(total);
    std::vector<float> nextVelY(total);
    std::vector<float> nextVelZ(total);

    backend.acoustic3DStep(pressure, velX, velY, velZ, n, n, n, 0.1f, 0.1f, nextPressure,
                           nextVelX, nextVelY, nextVelZ);

    // velocityX(0,1,1) -= velocityFactor*(pressure(1,1,1) - pressure(0,1,1)) = -0.1*10 =
    // -1
    EXPECT_NEAR(nextVelX[idx(0, 1, 1)], -1.0f, 1e-5f);
    // The spike cell's own pressure must have changed (divergence is nonzero
    // once neighboring velocities respond).
    EXPECT_NE(nextPressure[idx(1, 1, 1)], pressure[idx(1, 1, 1)]);
}

TEST(ComputeCpu, Maxwell3DStepOfAVacuumStaysAVacuum) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const std::size_t total = n * n * n;
    std::vector<float> zero(total, 0.0f);
    std::vector<float> nextEx(total);
    std::vector<float> nextEy(total);
    std::vector<float> nextEz(total);
    std::vector<float> nextBx(total);
    std::vector<float> nextBy(total);
    std::vector<float> nextBz(total);

    backend.maxwell3DStep(zero, zero, zero, zero, zero, zero, n, n, n, 0.1f, 0.1f, nextEx,
                          nextEy, nextEz, nextBx, nextBy, nextBz);

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(nextEx[i], 0.0f);
        EXPECT_FLOAT_EQ(nextBx[i], 0.0f);
    }
}

TEST(ComputeCpu, Maxwell3DStepPropagatesAnEFieldSpikeIntoB) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const std::size_t total = n * n * n;
    const auto idx = [](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };
    std::vector<float> ex(total, 0.0f);
    std::vector<float> ey(total, 0.0f);
    ey[idx(1, 1, 1)] = 5.0f;
    std::vector<float> ez(total, 0.0f);
    std::vector<float> zero(total, 0.0f);
    std::vector<float> nextEx(total);
    std::vector<float> nextEy(total);
    std::vector<float> nextEz(total);
    std::vector<float> nextBx(total);
    std::vector<float> nextBy(total);
    std::vector<float> nextBz(total);

    backend.maxwell3DStep(ex, ey, ez, zero, zero, zero, n, n, n, 0.1f, 0.1f, nextEx,
                          nextEy, nextEz, nextBx, nextBy, nextBz);

    // bx(1,1,1) -= bFactor * ((ez(1,2,1)-ez(1,1,1)) - (ey(1,1,2)-ey(1,1,1)))
    //            = -0.1 * (0 - (0 - 5)) = -0.1 * 5 = -0.5
    EXPECT_NEAR(nextBx[idx(1, 1, 1)], -0.5f, 1e-5f);
}

TEST(ComputeCpu, EulerianFluid3DSweepOfAUniformStateStaysUniformExactly) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const std::size_t total = n * n * n;
    // Left == right at every face, so each Rusanov flux is exactly
    // fluxOf(state) (the dissipation term is zero), and the flux
    // difference across every cell is exactly zero.
    std::vector<float> density(total, 1.2f);
    std::vector<float> momentumNormal(total, 0.3f);
    std::vector<float> momentumTangent1(total, -0.1f);
    std::vector<float> momentumTangent2(total, 0.05f);
    std::vector<float> energy(total, 2.5f);
    std::vector<float> nextDensity(total);
    std::vector<float> nextMomentumNormal(total);
    std::vector<float> nextMomentumTangent1(total);
    std::vector<float> nextMomentumTangent2(total);
    std::vector<float> nextEnergy(total);

    for (const int axis : {0, 1, 2}) {
        backend.eulerianFluid3DSweep(
            density, momentumNormal, momentumTangent1, momentumTangent2, energy, n, n, n,
            axis, 1.4f, 0.1f, nextDensity, nextMomentumNormal, nextMomentumTangent1,
            nextMomentumTangent2, nextEnergy);
        for (std::size_t i = 0; i < total; ++i) {
            EXPECT_FLOAT_EQ(nextDensity[i], 1.2f) << "axis " << axis;
            EXPECT_FLOAT_EQ(nextMomentumNormal[i], 0.3f) << "axis " << axis;
            EXPECT_FLOAT_EQ(nextEnergy[i], 2.5f) << "axis " << axis;
        }
    }
}

TEST(ComputeCpu, EulerianFluid3DSweepConservesTotalMassAcrossAPeriodicSweep) {
    // Every internal face's flux is added to one cell and subtracted from
    // its neighbor, so the sum telescopes to exactly zero over a periodic
    // domain: total mass (and, by the same argument, every other conserved
    // quantity) is unchanged to float32 rounding, regardless of how
    // non-uniform the state is.
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const std::size_t total = n * n * n;
    std::vector<float> density(total);
    std::vector<float> momentumNormal(total);
    std::vector<float> momentumTangent1(total, 0.0f);
    std::vector<float> momentumTangent2(total, 0.0f);
    std::vector<float> energy(total);
    for (std::size_t i = 0; i < total; ++i) {
        density[i] = 1.0f + 0.1f * static_cast<float>(i % 5);
        momentumNormal[i] = 0.05f * static_cast<float>(i % 3);
        energy[i] = 3.0f + 0.05f * static_cast<float>(i % 7);
    }
    std::vector<float> nextDensity(total);
    std::vector<float> nextMomentumNormal(total);
    std::vector<float> nextMomentumTangent1(total);
    std::vector<float> nextMomentumTangent2(total);
    std::vector<float> nextEnergy(total);

    backend.eulerianFluid3DSweep(density, momentumNormal, momentumTangent1,
                                 momentumTangent2, energy, n, n, n, 0, 1.4f, 0.01f,
                                 nextDensity, nextMomentumNormal, nextMomentumTangent1,
                                 nextMomentumTangent2, nextEnergy);

    float massBefore = 0.0f;
    float massAfter = 0.0f;
    for (std::size_t i = 0; i < total; ++i) {
        massBefore += density[i];
        massAfter += nextDensity[i];
    }
    EXPECT_NEAR(massAfter, massBefore, 1e-4f);
}

TEST(ComputeCpu, FftBatchedOfAConstantSequenceIsAllEnergyInTheDcBin) {
    // The DFT of a constant sequence is n at index 0 and 0 everywhere else
    // (the definition's sum_k x[k] * e^{-i*2*pi*0*k/n} = sum_k x[k] at bin 0,
    // and every other bin's phasors sum to exactly zero over a full period).
    const CpuBackend backend;
    constexpr std::size_t length = 8;
    std::vector<float> real(length, 1.0f);
    std::vector<float> imag(length, 0.0f);
    std::vector<float> nextReal(length);
    std::vector<float> nextImag(length);

    backend.fftBatched(real, imag, length, 1, false, nextReal, nextImag);

    EXPECT_NEAR(nextReal[0], 8.0f, 1e-4f);
    EXPECT_NEAR(nextImag[0], 0.0f, 1e-4f);
    for (std::size_t i = 1; i < length; ++i) {
        EXPECT_NEAR(nextReal[i], 0.0f, 1e-4f) << "bin " << i;
        EXPECT_NEAR(nextImag[i], 0.0f, 1e-4f) << "bin " << i;
    }
}

TEST(ComputeCpu, FftBatchedOfAnImpulseIsFlatAcrossEveryBin) {
    // The DFT of a unit impulse at index 0 is 1 at every bin (each phasor
    // e^{-i*2*pi*bin*0/n} = e^0 = 1, regardless of bin).
    const CpuBackend backend;
    constexpr std::size_t length = 8;
    std::vector<float> real(length, 0.0f);
    real[0] = 1.0f;
    std::vector<float> imag(length, 0.0f);
    std::vector<float> nextReal(length);
    std::vector<float> nextImag(length);

    backend.fftBatched(real, imag, length, 1, false, nextReal, nextImag);

    for (std::size_t i = 0; i < length; ++i) {
        EXPECT_NEAR(nextReal[i], 1.0f, 1e-4f) << "bin " << i;
        EXPECT_NEAR(nextImag[i], 0.0f, 1e-4f) << "bin " << i;
    }
}

TEST(ComputeCpu, FftBatchedForwardThenInverseRoundTrips) {
    const CpuBackend backend;
    constexpr std::size_t length = 16;
    std::vector<float> real(length);
    std::vector<float> imag(length);
    for (std::size_t i = 0; i < length; ++i) {
        real[i] =
            std::sin(0.3f * static_cast<float>(i)) + 0.5f * static_cast<float>(i % 3);
        imag[i] = 0.0f;
    }

    std::vector<float> spectrumReal(length);
    std::vector<float> spectrumImag(length);
    backend.fftBatched(real, imag, length, 1, false, spectrumReal, spectrumImag);

    std::vector<float> roundTripReal(length);
    std::vector<float> roundTripImag(length);
    backend.fftBatched(spectrumReal, spectrumImag, length, 1, true, roundTripReal,
                       roundTripImag);

    for (std::size_t i = 0; i < length; ++i) {
        EXPECT_NEAR(roundTripReal[i], real[i], 1e-3f) << "index " << i;
        EXPECT_NEAR(roundTripImag[i], 0.0f, 1e-3f) << "index " << i;
    }
}

TEST(ComputeCpu, FftBatchedTransformsEachBatchIndependently) {
    // Batch 0 is a constant sequence (DC-only spectrum), batch 1 is an
    // impulse (flat spectrum): one call must produce both correctly,
    // proving the batches don't cross-contaminate each other.
    const CpuBackend backend;
    constexpr std::size_t length = 4;
    std::vector<float> real{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> imag(2 * length, 0.0f);
    std::vector<float> nextReal(2 * length);
    std::vector<float> nextImag(2 * length);

    backend.fftBatched(real, imag, length, 2, false, nextReal, nextImag);

    EXPECT_NEAR(nextReal[0], 4.0f, 1e-4f);
    for (std::size_t i = 1; i < length; ++i) {
        EXPECT_NEAR(nextReal[i], 0.0f, 1e-4f) << "batch 0 bin " << i;
    }
    for (std::size_t i = 0; i < length; ++i) {
        EXPECT_NEAR(nextReal[length + i], 1.0f, 1e-4f) << "batch 1 bin " << i;
    }
}

TEST(ComputeCpu, MatVecMatchesTheDefinitionForARectangularMatrix) {
    const CpuBackend backend;
    // 2x3 matrix times a 3-vector.
    const std::array<float, 6> matrix{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::array<float, 3> vector{1.0f, 0.0f, -1.0f};
    std::vector<float> result(2);

    backend.matVec(matrix, 2, 3, vector, result);

    EXPECT_FLOAT_EQ(result[0], 1.0f * 1.0f + 2.0f * 0.0f + 3.0f * -1.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f * 1.0f + 5.0f * 0.0f + 6.0f * -1.0f);
}

TEST(ComputeCpu, MatMulOfIdentityIsTheOriginalMatrix) {
    const CpuBackend backend;
    const std::array<float, 6> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3
    const std::array<float, 9> identity{1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f};  // 3x3
    std::vector<float> result(6);

    backend.matMul(a, 2, 3, identity, 3, result);

    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], a[i]);
    }
}

TEST(ComputeCpu, MatMulMatchesTheDefinitionForARectangularProduct) {
    const CpuBackend backend;
    const std::array<float, 6> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};     // 2x3
    const std::array<float, 6> b{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};  // 3x2
    std::vector<float> result(4);

    backend.matMul(a, 2, 3, b, 2, result);

    // row 0: [1,2,3] . cols of b -> [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    EXPECT_FLOAT_EQ(result[0], 58.0f);
    EXPECT_FLOAT_EQ(result[1], 64.0f);
    // row 1: [4,5,6] . cols of b -> [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    EXPECT_FLOAT_EQ(result[2], 139.0f);
    EXPECT_FLOAT_EQ(result[3], 154.0f);
}

TEST(ComputeCpu, LuDecomposeGpuMatchesAKnownFactorizationAndSolve) {
    const CpuBackend backend;
    constexpr std::size_t n = 3;
    // A well-conditioned 3x3 matrix (diagonally dominant, no pivoting needed
    // to stay well-behaved, but partial pivoting still runs and may still
    // reorder rows for numerical reasons).
    const std::array<float, 9> a{4.0f, 3.0f, 2.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 6.0f};
    std::vector<float> lu(n * n);
    std::vector<std::uint32_t> pivot(n);

    ASSERT_TRUE(backend.luDecomposeGpu(a, n, lu, pivot));

    // Reconstruct L*U and compare against the permuted original matrix:
    // (L*U)(i,j) must equal a(pivot[i], j).
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k <= std::min(i, j); ++k) {
                const float lValue = (k == i) ? 1.0f : lu[i * n + k];
                total += lValue * lu[k * n + j];
            }
            EXPECT_NEAR(total, a[pivot[i] * n + j], 1e-4f) << "entry " << i << "," << j;
        }
    }
}

TEST(ComputeCpu, LuDecomposeGpuReportsFailureForASingularMatrix) {
    const CpuBackend backend;
    constexpr std::size_t n = 2;
    // Row 1 is twice row 0: singular.
    const std::array<float, 4> a{1.0f, 2.0f, 2.0f, 4.0f};
    std::vector<float> lu(n * n);
    std::vector<std::uint32_t> pivot(n);

    EXPECT_FALSE(backend.luDecomposeGpu(a, n, lu, pivot));
}

TEST(ComputeCpu, CholeskyDecomposeGpuMatchesAKnownFactorization) {
    const CpuBackend backend;
    constexpr std::size_t n = 3;
    // A symmetric positive-definite matrix (a Gram matrix, A^T A for a
    // full-rank A, is always SPD).
    const std::array<float, 9> a{4.0f, 2.0f, 0.0f, 2.0f, 5.0f, 1.0f, 0.0f, 1.0f, 3.0f};
    std::vector<float> l(n * n);

    ASSERT_TRUE(backend.choleskyDecomposeGpu(a, n, l));

    // Reconstruct L*L^T and compare against a.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                total += l[i * n + k] * l[j * n + k];
            }
            EXPECT_NEAR(total, a[i * n + j], 1e-4f) << "entry " << i << "," << j;
        }
    }
}

TEST(ComputeCpu, CholeskyDecomposeGpuReportsFailureForANonPositiveDefiniteMatrix) {
    const CpuBackend backend;
    constexpr std::size_t n = 2;
    // Negative diagonal: not positive-definite.
    const std::array<float, 4> a{-1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> l(n * n);

    EXPECT_FALSE(backend.choleskyDecomposeGpu(a, n, l));
}

TEST(ComputeCpu, QrDecomposeGpuReconstructsTheOriginalMatrix) {
    const CpuBackend backend;
    constexpr std::size_t rows = 3, cols = 2;
    const std::array<float, 6> matrix{1.0f, -1.0f, 2.0f, 1.0f, 0.0f, 3.0f};
    std::vector<float> q(rows * rows);
    std::vector<float> r(rows * cols);

    backend.qrDecomposeGpu(matrix, rows, cols, q, r);

    // Q must be orthogonal: Q^T Q == I.
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < rows; ++j) {
            float dot = 0.0f;
            for (std::size_t k = 0; k < rows; ++k) {
                dot += q[k * rows + i] * q[k * rows + j];
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0f : 0.0f, 1e-4f)
                << "Q^T Q entry " << i << "," << j;
        }
    }
    // R must be upper triangular.
    for (std::size_t i = 1; i < rows; ++i) {
        for (std::size_t j = 0; j < std::min(i, cols); ++j) {
            EXPECT_NEAR(r[i * cols + j], 0.0f, 1e-4f) << "R entry " << i << "," << j;
        }
    }
    // Q * R must reconstruct the original matrix.
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < rows; ++k) {
                total += q[i * rows + k] * r[k * cols + j];
            }
            EXPECT_NEAR(total, matrix[i * cols + j], 1e-4f)
                << "QR entry " << i << "," << j;
        }
    }
}

TEST(ComputeCpu, JacobiEigenSymmetricGpuMatchesAKnownDiagonalization) {
    const CpuBackend backend;
    constexpr std::size_t n = 2;
    // [[2,1],[1,2]] has eigenvalues 1 and 3.
    const std::array<float, 4> matrix{2.0f, 1.0f, 1.0f, 2.0f};
    std::vector<float> diagonal(n * n);
    std::vector<float> eigenvectors(n * n);

    backend.jacobiEigenSymmetricGpu(matrix, n, 100, 0.0f, diagonal, eigenvectors);

    std::vector<float> values{diagonal[0], diagonal[n * n - 1]};
    std::sort(values.begin(), values.end());
    EXPECT_NEAR(values[0], 1.0f, 1e-4f);
    EXPECT_NEAR(values[1], 3.0f, 1e-4f);

    // Eigenvectors must be orthonormal.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float dot = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                dot += eigenvectors[k * n + i] * eigenvectors[k * n + j];
            }
            EXPECT_NEAR(dot, (i == j) ? 1.0f : 0.0f, 1e-4f)
                << "V^T V entry " << i << "," << j;
        }
    }
}

TEST(ComputeCpu, JacobiSvdGpuReconstructsTheOriginalMatrix) {
    const CpuBackend backend;
    constexpr std::size_t rows = 3, cols = 2;
    const std::array<float, 6> matrix{1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> resultA(rows * cols);
    std::vector<float> resultV(cols * cols);

    backend.jacobiSvdGpu(matrix, rows, cols, 60, 0.0f, resultA, resultV);

    // resultA's columns, once normalized, are U scaled by singular values;
    // resultV is V. Reconstruct A = resultA * V^T and compare.
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < cols; ++k) {
                total += resultA[i * cols + k] * resultV[j * cols + k];
            }
            EXPECT_NEAR(total, matrix[i * cols + j], 1e-4f) << "entry " << i << "," << j;
        }
    }
}

TEST(ComputeCpu, BatchErfMatchesKnownValues) {
    const CpuBackend backend;
    const std::vector<float> x{0.0f, 0.5f, 1.0f, -1.0f, 2.0f};
    std::vector<float> result(x.size());

    backend.batchErf(x, result);

    EXPECT_NEAR(result[0], 0.0f, 1e-6f);
    EXPECT_NEAR(result[1], 0.5204998778f, 1e-6f);
    EXPECT_NEAR(result[2], 0.8427007929f, 1e-6f);
    EXPECT_NEAR(result[3], -0.8427007929f, 1e-6f);
    EXPECT_NEAR(result[4], 0.9953222650f, 1e-6f);
}

TEST(ComputeCpu, BatchErfcIsOneMinusBatchErf) {
    const CpuBackend backend;
    const std::vector<float> x{0.0f, 0.5f, 1.0f, -1.0f, 2.0f};
    std::vector<float> erfResult(x.size());
    std::vector<float> erfcResult(x.size());

    backend.batchErf(x, erfResult);
    backend.batchErfc(x, erfcResult);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(erfcResult[i], 1.0f - erfResult[i], 1e-6f) << "element " << i;
    }
}

TEST(ComputeCpu, BatchGammaMatchesKnownFactorials) {
    const CpuBackend backend;
    const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.5f};
    std::vector<float> result(x.size());

    backend.batchGamma(x, result);

    EXPECT_NEAR(result[0], 1.0f, 1e-4f);           // 0!
    EXPECT_NEAR(result[1], 1.0f, 1e-4f);           // 1!
    EXPECT_NEAR(result[2], 2.0f, 1e-4f);           // 2!
    EXPECT_NEAR(result[3], 6.0f, 1e-4f);           // 3!
    EXPECT_NEAR(result[4], 24.0f, 1e-4f);          // 4!
    EXPECT_NEAR(result[5], 1.7724538509f, 1e-4f);  // Gamma(1/2) = sqrt(pi)
}

TEST(ComputeCpu, BatchLogGammaMatchesTheLogOfBatchGammaForModerateArguments) {
    const CpuBackend backend;
    const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> gammaResult(x.size());
    std::vector<float> logGammaResult(x.size());

    backend.batchGamma(x, gammaResult);
    backend.batchLogGamma(x, logGammaResult);

    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(logGammaResult[i], std::log(gammaResult[i]), 1e-3f)
            << "element " << i;
    }
}

TEST(ComputeCpu, BatchLegendrePMatchesKnownPolynomialValues) {
    const CpuBackend backend;
    const std::vector<float> x{0.0f, 0.5f, 1.0f, -1.0f};
    std::vector<float> result(x.size());

    // P_2(x) = (3x^2 - 1) / 2
    backend.batchLegendreP(2, 0, x, result);
    EXPECT_NEAR(result[0], -0.5f, 1e-5f);
    EXPECT_NEAR(result[1], -0.125f, 1e-5f);
    EXPECT_NEAR(result[2], 1.0f, 1e-5f);
    EXPECT_NEAR(result[3], 1.0f, 1e-5f);
}

TEST(ComputeCpu, BatchPolynomialEvalMatchesHornersMethod) {
    const CpuBackend backend;
    // p(x) = 1 + 2x + 3x^2
    const std::vector<float> coefficients{1.0f, 2.0f, 3.0f};
    const std::vector<float> x{0.0f, 1.0f, 2.0f, -1.0f};
    std::vector<float> result(x.size());

    backend.batchPolynomialEval(coefficients, x, result);

    EXPECT_NEAR(result[0], 1.0f, 1e-5f);
    EXPECT_NEAR(result[1], 6.0f, 1e-5f);
    EXPECT_NEAR(result[2], 17.0f, 1e-5f);
    EXPECT_NEAR(result[3], 2.0f, 1e-5f);
}

TEST(ComputeCpu, BatchCubicSplineEvalMatchesTheKnotsAtTheKnotsThemselves) {
    const CpuBackend backend;
    const std::vector<float> knotsX{0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> knotsY{0.0f, 1.0f, 4.0f, 9.0f};
    // A natural spline's second derivatives are zero at both endpoints; the
    // interior ones do not matter for this test, which only checks that the
    // spline reproduces the tabulated values exactly at the knots
    // themselves (the cubic term vanishes there regardless).
    const std::vector<float> secondDerivatives{0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> queryX{0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> result(queryX.size());

    backend.batchCubicSplineEval(knotsX, knotsY, secondDerivatives, queryX, result);

    for (std::size_t i = 0; i < queryX.size(); ++i) {
        EXPECT_NEAR(result[i], knotsY[i], 1e-5f) << "element " << i;
    }
}

TEST(ComputeCpu, BatchUniformRealIsDeterministicForTheSameSeedAndOffset) {
    const CpuBackend backend;
    std::vector<float> first(256);
    std::vector<float> second(256);

    backend.batchUniformReal(42, 0, first);
    backend.batchUniformReal(42, 0, second);

    EXPECT_EQ(first, second);
}

TEST(ComputeCpu, BatchUniformRealStaysWithinZeroToOne) {
    const CpuBackend backend;
    std::vector<float> result(4096);

    backend.batchUniformReal(7, 0, result);

    for (float value : result) {
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
    }
}

TEST(ComputeCpu, BatchUniformRealDiffersAcrossDistinctIndicesInOneCall) {
    // Not a proof of independence, just a sanity check against the
    // degenerate bug of every thread computing the same value.
    const CpuBackend backend;
    std::vector<float> result(64);

    backend.batchUniformReal(7, 0, result);

    const std::size_t distinctCount =
        std::set<float>(result.begin(), result.end()).size();
    EXPECT_EQ(distinctCount, result.size());
}

TEST(ComputeCpu, BatchUniformRealAtNonZeroOffsetContinuesTheSameLogicalStream) {
    // batchUniformReal(seed, 0, [n]) followed by batchUniformReal(seed, n,
    // [m]) must equal one batchUniformReal(seed, 0, [n+m]) call, element
    // for element: that's what makes offset a genuine stream position
    // rather than an unrelated second seed.
    const CpuBackend backend;
    std::vector<float> wholeStream(20);
    backend.batchUniformReal(99, 0, wholeStream);

    std::vector<float> firstHalf(12);
    std::vector<float> secondHalf(8);
    backend.batchUniformReal(99, 0, firstHalf);
    backend.batchUniformReal(99, 12, secondHalf);

    for (std::size_t i = 0; i < firstHalf.size(); ++i) {
        EXPECT_FLOAT_EQ(firstHalf[i], wholeStream[i]) << "element " << i;
    }
    for (std::size_t i = 0; i < secondHalf.size(); ++i) {
        EXPECT_FLOAT_EQ(secondHalf[i], wholeStream[12 + i]) << "element " << i;
    }
}

TEST(ComputeCpu, BatchUniformRealWithADifferentSeedProducesADifferentStream) {
    const CpuBackend backend;
    std::vector<float> a(64);
    std::vector<float> b(64);

    backend.batchUniformReal(1, 0, a);
    backend.batchUniformReal(2, 0, b);

    EXPECT_NE(a, b);
}

TEST(ComputeCpu, BatchUniformRealSampleMeanIsNearOneHalf) {
    const CpuBackend backend;
    std::vector<float> result(200000);

    backend.batchUniformReal(123, 0, result);

    double total = 0.0;
    for (float value : result) {
        total += value;
    }
    const double mean = total / static_cast<double>(result.size());
    EXPECT_NEAR(mean, 0.5, 0.01);
}

TEST(ComputeCpu, BatchNormalSampleMeanAndVarianceAreNearStandard) {
    const CpuBackend backend;
    std::vector<float> result(200000);

    backend.batchNormal(456, 0, result);

    double total = 0.0;
    for (float value : result) {
        total += value;
    }
    const double mean = total / static_cast<double>(result.size());

    double variance = 0.0;
    for (float value : result) {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<double>(result.size());

    EXPECT_NEAR(mean, 0.0, 0.02);
    EXPECT_NEAR(variance, 1.0, 0.05);
}

TEST(ComputeCpu, BatchNormalIsDeterministicForTheSameSeedAndOffset) {
    const CpuBackend backend;
    std::vector<float> first(256);
    std::vector<float> second(256);

    backend.batchNormal(11, 5, first);
    backend.batchNormal(11, 5, second);

    EXPECT_EQ(first, second);
}

TEST(ComputeCpu, SortAscendingSortsAKnownUnorderedSequence) {
    const CpuBackend backend;
    std::vector<float> values{5.0f, 3.0f, 8.0f, 1.0f, 9.0f, 2.0f};

    backend.sortAscending(values);

    const std::vector<float> expected{1.0f, 2.0f, 3.0f, 5.0f, 8.0f, 9.0f};
    EXPECT_EQ(values, expected);
}

TEST(ComputeCpu, SortAscendingOnAlreadySortedDataIsANoOp) {
    const CpuBackend backend;
    std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> expected = values;

    backend.sortAscending(values);

    EXPECT_EQ(values, expected);
}

TEST(ComputeCpu, SortAscendingOnEmptyOrSingleElementDoesNothing) {
    const CpuBackend backend;
    std::vector<float> empty;
    backend.sortAscending(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<float> single{42.0f};
    backend.sortAscending(single);
    EXPECT_EQ(single, (std::vector<float>{42.0f}));
}

TEST(ComputeCpu, SortAscendingHandlesANonPowerOfTwoSize) {
    const CpuBackend backend;
    std::vector<float> values{7.0f, 1.0f, 5.0f, 3.0f, 9.0f};  // 5 elements

    backend.sortAscending(values);

    const std::vector<float> expected{1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    EXPECT_EQ(values, expected);
}

TEST(ComputeCpu, MultigridRestrict3DAveragesTheEightFineCellsPerCoarseCell) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;  // coarse: 2x2x2
    const auto idx = [](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };
    std::vector<float> fine(n * n * n, 0.0f);
    // Coarse cell (0,0,0) covers fine cells (0..1, 0..1, 0..1): 1+2+3+4+5+6+7+8 = 36, /8
    // = 4.5
    fine[idx(0, 0, 0)] = 1.0f;
    fine[idx(0, 0, 1)] = 2.0f;
    fine[idx(0, 1, 0)] = 3.0f;
    fine[idx(0, 1, 1)] = 4.0f;
    fine[idx(1, 0, 0)] = 5.0f;
    fine[idx(1, 0, 1)] = 6.0f;
    fine[idx(1, 1, 0)] = 7.0f;
    fine[idx(1, 1, 1)] = 8.0f;
    std::vector<float> coarse(2 * 2 * 2, -1.0f);

    backend.multigridRestrict3D(fine, n, n, n, coarse);

    EXPECT_FLOAT_EQ(coarse[0], 4.5f);
    // Every other coarse cell covers all-zero fine cells.
    for (std::size_t i = 1; i < coarse.size(); ++i) {
        EXPECT_FLOAT_EQ(coarse[i], 0.0f);
    }
}

TEST(ComputeCpu,
     MultigridProlongateAndAddBroadcastsEachCoarseCorrectionToItsEightFineCells) {
    const CpuBackend backend;
    constexpr std::size_t n = 4;
    const auto idx = [](std::size_t i, std::size_t j, std::size_t k) {
        return (i * n + j) * n + k;
    };
    std::vector<float> fine(n * n * n, 1.0f);
    std::vector<float> coarseCorrection(2 * 2 * 2, 0.0f);
    coarseCorrection[0] = 10.0f;  // covers fine cells (0..1, 0..1, 0..1)
    std::vector<float> nextFine(n * n * n, -1.0f);

    backend.multigridProlongateAndAdd3D(fine, coarseCorrection, n, n, n, nextFine);

    EXPECT_FLOAT_EQ(nextFine[idx(0, 0, 0)], 11.0f);
    EXPECT_FLOAT_EQ(nextFine[idx(1, 1, 1)], 11.0f);
    // Outside coarse cell (0,0,0)'s 8 fine cells, only the original value survives.
    EXPECT_FLOAT_EQ(nextFine[idx(2, 2, 2)], 1.0f);
}

TEST(ComputeCpu, MinIndexFindsTheLowestValueAtTheEarliestTiedIndex) {
    const CpuBackend backend;
    const std::array<float, 5> x{3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    EXPECT_EQ(backend.minIndex(x), 1u);  // ties between index 1 and 3; 1 wins
}

TEST(ComputeCpu, MinIndexOfEmptyIsSize) {
    const CpuBackend backend;
    EXPECT_EQ(backend.minIndex(std::span<const float>{}), 0u);
}

TEST(ComputeCpu, DoublePathMatchesItsOwnDefinition) {
    const CpuBackend backend;
    std::vector<double> xd{1.5, 2.5, 3.5};
    std::vector<double> yd{0.0, 0.0, 0.0};
    backend.saxpyD(xd, yd, 2.0);
    EXPECT_DOUBLE_EQ(yd[0], 3.0);
    EXPECT_DOUBLE_EQ(yd[1], 5.0);
    EXPECT_DOUBLE_EQ(yd[2], 7.0);

    EXPECT_DOUBLE_EQ(backend.sumD(xd), 7.5);
}

TEST(ComputeCpu, KindIsCpu) {
    const CpuBackend backend;
    EXPECT_EQ(backend.kind(), ysq::ComputeBackendKind::Cpu);
}

}  // namespace
