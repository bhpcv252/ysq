#pragma once

#include <Math/Scalar.hpp>

#include <cmath>

namespace ysq {

/// Rayleigh scattering: a gas's per-molecule cross-section, and the
/// exponential number-density profile any isothermal barometric atmosphere
/// has (the same shape Physics/Thermodynamics/Thermodynamics.hpp's
/// isothermalAtmosphereDensity and Optics/RefractiveMedium.hpp's
/// refractivity share, since all three trace back to the same hydrostatic
/// gas). General for any gas; a scenario's real number density, refractive
/// index and scale height are its own data, not part of this law.

/// The Rayleigh scattering cross-section per molecule, sigma(lambda) =
/// (24 pi^3 / (lambda^4 N^2)) * ((n^2 - 1) / (n^2 + 2))^2, the standard
/// result for scattering by particles much smaller than the wavelength
/// (Bohren & Huffman, "Absorption and Scattering of Light by Small
/// Particles"). `refractiveIndex` and `numberDensity` are the gas's values
/// at whatever single reference condition they were measured at (its
/// surface, say); the 1/lambda^4 dependence is what makes blue light
/// scatter far more than red over the same path, the reason both the sky
/// and a sunset are colored at all.
[[nodiscard]] inline double
rayleighCrossSection(double wavelength, double refractiveIndex, double numberDensity) {
    const double nSquared = refractiveIndex * refractiveIndex;
    const double ratio = (nSquared - 1.0) / (nSquared + 2.0);
    const double lambda2 = wavelength * wavelength;
    const double lambda4 = lambda2 * lambda2;
    const double pi3 = kPi<double> * kPi<double> * kPi<double>;
    return (24.0 * pi3 * ratio * ratio) / (lambda4 * numberDensity * numberDensity);
}

/// The same isothermal barometric exponential shape
/// Thermodynamics::isothermalAtmosphereDensity has, in number-density terms:
/// N(r) = surfaceNumberDensity * exp(-(r - radius) / scaleHeight).
[[nodiscard]] inline double exponentialNumberDensity(double r,
                                                     double surfaceNumberDensity,
                                                     double radius, double scaleHeight) {
    return surfaceNumberDensity * std::exp(-(r - radius) / scaleHeight);
}

/// Beer-Lambert: the fraction of light surviving an optical depth `tau`.
[[nodiscard]] inline double transmission(double opticalDepth) {
    return std::exp(-opticalDepth);
}

}  // namespace ysq
