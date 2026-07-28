#pragma once

#include <Math/Scalar.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Units/Constants.hpp>

#include <cmath>

namespace ysq {

/// A non-rotating mass: the unique spherically symmetric vacuum solution.
///
/// Chart is (T, r, polar, azimuth) := (c t, radius, polar angle down from
/// the axis, azimuth), the physics convention Math/CoordinateSystems.hpp
/// already uses:
///
///     ds^2 = -(1 - r_s/r) dT^2 + dr^2 / (1 - r_s/r) + r^2 dpolar^2
///            + r^2 sin^2(polar) dazimuth^2
///
/// Components diverge at r = r_s, the coordinate singularity of this chart
/// rather than a physical one, and at r = 0, the genuine curvature
/// singularity. Neither is guarded against here: a geodesic that reaches
/// either has left the regime this chart, or classical general relativity's
/// description of it, can say anything about.
class Schwarzschild {
public:
    /// `mass` is the source's gravitational parameter GM, not its mass in
    /// kilograms, the same preference Gravity has and for the same reason:
    /// GM is what is actually measured, and carries none of G's
    /// uncertainty. r_s = 2GM/c^2 is computed once, here, rather than on
    /// every metric evaluation.
    explicit Schwarzschild(GravitationalParameter mass)
        : m_schwarzschildRadius(
              2.0 * mass.value() /
              (constants::speedOfLight.value() * constants::speedOfLight.value())) {}

    [[nodiscard]] double schwarzschildRadius() const noexcept {
        return m_schwarzschildRadius;
    }

    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>& at) const {
        using std::sin;

        const T r = at.y;
        const T polar = at.z;
        const T factor = T{1} - static_cast<T>(m_schwarzschildRadius) / r;
        const T sinPolar = sin(polar);

        MetricTensor<T> g{};
        g(0, 0) = -factor;
        g(1, 1) = T{1} / factor;
        g(2, 2) = r * r;
        g(3, 3) = r * r * sinPolar * sinPolar;
        return g;
    }

private:
    double m_schwarzschildRadius;
};

}  // namespace ysq
