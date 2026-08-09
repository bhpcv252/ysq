#pragma once

#include <span>
#include <vector>

namespace ysq {

/// Fraunhofer (far-field) diffraction and interference: wave-optics
/// phenomena `Optics`'s other headers, all built on ray propagation
/// through a metric, do not cover. General for whatever aperture a caller
/// describes, not tied to a single slit, a double slit, or any other one
/// scenario.

/// The far-field diffraction pattern of a 1D aperture, as a function of
/// diffraction angle: `sinTheta[i]`/`intensity[i]` pairs, intensity
/// normalized to 1 at its own maximum.
struct DiffractionPattern {
    std::vector<double> sinTheta;
    std::vector<double> intensity;
};

/// The defining fact of Fraunhofer diffraction: the far-field pattern of an
/// aperture is exactly the (squared magnitude of the) Fourier transform of
/// its own transmission function -- not an approximation specific to one
/// aperture shape, general for whatever aperture the sampled
/// `transmission` array describes (a single slit, a double slit, an
/// arbitrary diffraction grating or apodized aperture). Computed via
/// `Math/FFT.hpp`, so `transmission.size()` must be a power of two.
///
/// `transmission[i]` is the aperture's own transmission amplitude
/// (typically 0 outside the aperture and 1 inside, though a graded or
/// apodized aperture can use any value between) at a position spanning
/// `apertureWidth`. FFT bin `k` corresponds to spatial frequency `k /
/// apertureWidth` (negative for `k > n/2`, the standard FFT convention for
/// the upper half of the output); `sinTheta = wavelength * frequency`
/// converts that to a diffraction angle, and a bin whose implied
/// `|sinTheta| > 1` (no real angle reaches that spatial frequency) is
/// dropped rather than returned as a nonphysical value.
[[nodiscard]] DiffractionPattern
fraunhoferDiffraction(std::span<const double> transmission, double apertureWidth,
                      double wavelength);

/// The classical two-slit (Young's) interference pattern, as a closed
/// form rather than a numerical transform: two slits of width `slitWidth`
/// separated by `slitSeparation`, illuminated by a plane wave of
/// `wavelength`. Intensity (normalized to 1 at `sinTheta = 0`) is the
/// single-slit diffraction envelope `sinc(beta)^2` times the two-beam
/// interference factor `cos(gamma)^2`, `beta = pi slitWidth sinTheta /
/// wavelength`, `gamma = pi slitSeparation sinTheta / wavelength`
/// (Hecht, *Optics*): the envelope from each slit's own finite width,
/// modulated by the interference between the two.
[[nodiscard]] double twoSlitIntensity(double sinTheta, double slitWidth,
                                      double slitSeparation, double wavelength);

}  // namespace ysq
