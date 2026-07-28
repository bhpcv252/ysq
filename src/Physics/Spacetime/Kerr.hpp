#pragma once

#include <Math/Scalar.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Units/Constants.hpp>
#include <Units/Length.hpp>

#include <cmath>

namespace ysq {

/// A rotating mass: the Boyer-Lindquist form of the unique stationary,
/// axisymmetric vacuum solution.
///
/// Chart is (T, r, polar, azimuth), the same convention as Schwarzschild,
/// which this reduces to exactly at spin = 0 (spacetime_kerr.cpp checks that
/// reduction directly rather than trusting the algebra by inspection).
/// Writing sigma = r^2 + a^2 cos^2(polar) and delta = r^2 - r_s r + a^2:
///
///     ds^2 = -(1 - r_s r / sigma) dT^2
///            - (2 r_s r a sin^2(polar) / sigma) dT dazimuth
///            + (sigma / delta) dr^2 + sigma dpolar^2
///            + (r^2 + a^2 + r_s r a^2 sin^2(polar) / sigma) sin^2(polar) dazimuth^2
///
/// The g_T,azimuth cross term is what frame dragging is: an observer at
/// fixed r and polar cannot hold azimuth fixed and stay on a timelike
/// worldline close enough to the horizon, since the term forces dT and
/// dazimuth to mix.
class Kerr {
public:
    /// `mass` is GM as in Schwarzschild. `spin` is a = J / (M c), a length,
    /// not the dimensionless a / GM ratio some references use.
    explicit Kerr(GravitationalParameter mass, Length spin)
        : m_schwarzschildRadius(
              2.0 * mass.value() /
              (constants::speedOfLight.value() * constants::speedOfLight.value())),
          m_spin(spin.value()) {}

    [[nodiscard]] double schwarzschildRadius() const noexcept {
        return m_schwarzschildRadius;
    }
    [[nodiscard]] double spin() const noexcept { return m_spin; }

    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>& at) const {
        using std::cos;
        using std::sin;

        const T r = at.y;
        const T polar = at.z;
        const T rs = static_cast<T>(m_schwarzschildRadius);
        const T a = static_cast<T>(m_spin);

        const T cosPolar = cos(polar);
        const T sinPolar = sin(polar);
        const T sigma = r * r + a * a * cosPolar * cosPolar;
        const T delta = r * r - rs * r + a * a;

        MetricTensor<T> g{};
        g(0, 0) = -(T{1} - rs * r / sigma);
        g(0, 3) = -rs * r * a * sinPolar * sinPolar / sigma;
        g(3, 0) = g(0, 3);
        g(1, 1) = sigma / delta;
        g(2, 2) = sigma;
        g(3, 3) = (r * r + a * a + rs * r * a * a * sinPolar * sinPolar / sigma) *
                  sinPolar * sinPolar;
        return g;
    }

private:
    double m_schwarzschildRadius;
    double m_spin;
};

}  // namespace ysq
