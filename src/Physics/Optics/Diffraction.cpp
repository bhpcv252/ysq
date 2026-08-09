#include <Physics/Optics/Diffraction.hpp>

#include <Math/Complex.hpp>
#include <Math/FFT.hpp>
#include <Math/Scalar.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ysq {

DiffractionPattern fraunhoferDiffraction(std::span<const double> transmission,
                                         double apertureWidth, double wavelength) {
    const std::size_t n = transmission.size();
    std::vector<Complex<double>> field = fftReal<double>(transmission);

    std::vector<double> rawIntensity(n);
    double maxIntensity = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        rawIntensity[i] = field[i].re * field[i].re + field[i].im * field[i].im;
        maxIntensity = std::max(maxIntensity, rawIntensity[i]);
    }

    DiffractionPattern pattern;
    for (std::size_t k = 0; k < n; ++k) {
        const long signedK = (k <= n / 2) ? static_cast<long>(k)
                                          : static_cast<long>(k) - static_cast<long>(n);
        const double spatialFrequency = static_cast<double>(signedK) / apertureWidth;
        const double sinTheta = wavelength * spatialFrequency;
        if (std::abs(sinTheta) > 1.0) {
            continue;
        }

        pattern.sinTheta.push_back(sinTheta);
        pattern.intensity.push_back(rawIntensity[k] / maxIntensity);
    }

    return pattern;
}

double twoSlitIntensity(double sinTheta, double slitWidth, double slitSeparation,
                        double wavelength) {
    const double beta = kPi<double> * slitWidth * sinTheta / wavelength;
    const double gamma = kPi<double> * slitSeparation * sinTheta / wavelength;

    const double singleSlitFactor =
        isNearZero(beta) ? 1.0 : std::pow(std::sin(beta) / beta, 2.0);
    const double interferenceFactor = std::cos(gamma) * std::cos(gamma);

    return singleSlitFactor * interferenceFactor;
}

}  // namespace ysq
