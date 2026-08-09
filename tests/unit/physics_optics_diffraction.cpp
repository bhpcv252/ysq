#include <Math/Scalar.hpp>
#include <Physics/Optics/Diffraction.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

namespace {

// Independent reference implementation of the O(n^2) discrete Fourier sum,
// matching Math/FFT.hpp's own convention (X_k = sum_n x_n exp(-2 pi i k n
// / N)), used to check fraunhoferDiffraction's FFT-based result against
// something that does not go through Math/FFT.hpp at all.
std::complex<double> directDft(const std::vector<double>& transmission, long k) {
    const auto n = static_cast<double>(transmission.size());
    std::complex<double> total{0.0, 0.0};
    for (std::size_t j = 0; j < transmission.size(); ++j) {
        const double angle =
            -2.0 * ysq::kPi<double> * static_cast<double>(k) * static_cast<double>(j) / n;
        total += transmission[j] * std::complex<double>{std::cos(angle), std::sin(angle)};
    }
    return total;
}

}  // namespace

TEST(PhysicsOpticsDiffraction, FraunhoferDiffractionCentralMaximumIsAtZeroAngle) {
    const std::vector<double> transmission(64, 1.0);  // a fully open aperture
    const ysq::DiffractionPattern pattern =
        ysq::fraunhoferDiffraction(transmission, 1.0e-3, 500e-9);

    ASSERT_FALSE(pattern.sinTheta.empty());

    std::size_t centralIndex = 0;
    for (std::size_t i = 0; i < pattern.sinTheta.size(); ++i) {
        if (std::abs(pattern.sinTheta[i]) < std::abs(pattern.sinTheta[centralIndex])) {
            centralIndex = i;
        }
    }
    EXPECT_NEAR(pattern.sinTheta[centralIndex], 0.0, 1e-9);

    for (double intensity : pattern.intensity) {
        EXPECT_LE(intensity, pattern.intensity[centralIndex] + 1e-12);
    }
}

TEST(PhysicsOpticsDiffraction, FraunhoferDiffractionMatchesTheDirectDftDefinition) {
    // A non-uniform aperture (a graded transmission profile), so the
    // cross-check isn't trivially symmetric.
    std::vector<double> transmission(32, 0.0);
    for (std::size_t i = 0; i < transmission.size(); ++i) {
        transmission[i] = (i < 20) ? (1.0 - static_cast<double>(i) / 40.0) : 0.0;
    }

    const double apertureWidth = 2.0e-3;
    const double wavelength = 600e-9;
    const ysq::DiffractionPattern pattern =
        ysq::fraunhoferDiffraction(transmission, apertureWidth, wavelength);

    for (std::size_t i = 0; i < pattern.sinTheta.size(); ++i) {
        const double spatialFrequency = pattern.sinTheta[i] / wavelength;
        const double kReal = spatialFrequency * apertureWidth;
        const auto k = static_cast<long>(std::lround(kReal));

        const std::complex<double> expected = directDft(transmission, k);
        const double expectedIntensity = std::norm(expected);

        // Recover the same normalization fraunhoferDiffraction itself
        // applies (division by the maximum raw intensity) by comparing
        // ratios rather than raw magnitudes.
        const std::complex<double> peak = directDft(transmission, 0);
        const double expectedNormalized = expectedIntensity / std::norm(peak);

        EXPECT_NEAR(pattern.intensity[i], expectedNormalized, 1e-9);
    }
}

TEST(PhysicsOpticsDiffraction, FraunhoferDiffractionDropsNonphysicalSpatialFrequencies) {
    // A long wavelength relative to a fine sampling pushes some bins'
    // implied sinTheta past 1, which must be dropped rather than returned.
    const std::vector<double> transmission(16, 1.0);
    const ysq::DiffractionPattern pattern =
        ysq::fraunhoferDiffraction(transmission, 1.0e-6, 500e-9);

    for (double s : pattern.sinTheta) {
        EXPECT_LE(std::abs(s), 1.0);
    }
}

TEST(PhysicsOpticsDiffraction, TwoSlitIntensityIsMaximalAtZeroAngle) {
    EXPECT_NEAR(ysq::twoSlitIntensity(0.0, 1e-5, 5e-5, 600e-9), 1.0, 1e-9);
}

TEST(PhysicsOpticsDiffraction,
     TwoSlitIntensityHasAnInterferenceNullAtHalfAWavelengthPathDifference) {
    const double slitWidth = 1e-6;  // narrow enough that the envelope stays broad here
    const double slitSeparation = 5e-5;
    const double wavelength = 600e-9;

    // gamma = pi/2 exactly when sinTheta = wavelength / (2 * separation).
    const double sinTheta = wavelength / (2.0 * slitSeparation);
    const double intensity =
        ysq::twoSlitIntensity(sinTheta, slitWidth, slitSeparation, wavelength);

    EXPECT_NEAR(intensity, 0.0, 1e-9);
}

TEST(PhysicsOpticsDiffraction, TwoSlitIntensityHasADiffractionNullAtTheSingleSlitZero) {
    const double slitWidth = 1e-5;
    const double slitSeparation = 3e-5;
    const double wavelength = 600e-9;

    // beta = pi exactly when sinTheta = wavelength / slitWidth: the
    // single-slit envelope's own first zero, which forces the whole
    // pattern to zero there regardless of the interference factor.
    const double sinTheta = wavelength / slitWidth;
    const double intensity =
        ysq::twoSlitIntensity(sinTheta, slitWidth, slitSeparation, wavelength);

    EXPECT_NEAR(intensity, 0.0, 1e-9);
}

TEST(PhysicsOpticsDiffraction, TwoSlitIntensityIsSymmetricInAngle) {
    const double slitWidth = 1e-5;
    const double slitSeparation = 4e-5;
    const double wavelength = 600e-9;
    const double sinTheta = 0.008;

    EXPECT_NEAR(ysq::twoSlitIntensity(sinTheta, slitWidth, slitSeparation, wavelength),
                ysq::twoSlitIntensity(-sinTheta, slitWidth, slitSeparation, wavelength),
                1e-12);
}
