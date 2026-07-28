#pragma once

#include <Math/Scalar.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>

#include <cmath>

namespace ysq {

/// An expanding, homogeneous, isotropic universe: comoving coordinates
/// (T, r, polar, azimuth), T = c t, and r, polar, azimuth unchanging for a
/// fundamental observer as the universe expands.
///
///     ds^2 = -dT^2 + a(T)^2 [ dr^2 / (1 - k r^2) + r^2 dpolar^2
///                             + r^2 sin^2(polar) dazimuth^2 ]
///
/// a is dimensionless, usually normalized to 1 at the present; k is +1, 0 or
/// -1 for closed, flat and open.
///
/// **What determines a(T) is a separate problem from what a metric is.** In
/// general it comes from the Friedmann equations, sourced by however much
/// matter, radiation and dark energy the universe holds, and solving that
/// coupled system is not implemented here: what is here are the three
/// standard single-component analytic solutions, for the regimes where one
/// component's stress-energy dominates the rest. Each is its own class
/// rather than one FLRW parameterized by a callable scale factor, because a
/// callable would have to be differentiable through Dual to give correct
/// Christoffel symbols, and Dual (Math/Dual.hpp) supports log, exp and sqrt
/// but not a general pow; writing t^(2/3) as exp((2/3) log t) below is
/// exact for exactly that reason, once, here, rather than asking every
/// caller supplying a scale factor to know the same trick.

namespace detail {

/// The spatial part every variant below shares; only a(T) differs between
/// them.
template <Numeric T>
[[nodiscard]] MetricTensor<T> flrwComponents(const T& scaleFactor, double curvature,
                                             const Vector4<T>& at) {
    using std::sin;

    const T r = at.y;
    const T polar = at.z;
    const T aSquared = scaleFactor * scaleFactor;
    const T sinPolar = sin(polar);

    MetricTensor<T> g{};
    g(0, 0) = T{-1};
    g(1, 1) = aSquared / (T{1} - static_cast<T>(curvature) * r * r);
    g(2, 2) = aSquared * r * r;
    g(3, 3) = aSquared * r * r * sinPolar * sinPolar;
    return g;
}

}  // namespace detail

/// Matter domination (dust, equation of state w = 0): a(T) = (T / T0)^(2/3).
/// T0 sets the normalization so a(T0) = 1.
class MatterDominatedFLRW {
public:
    MatterDominatedFLRW(double referenceTime, double curvature)
        : m_referenceTime(referenceTime), m_curvature(curvature) {}

    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>& at) const {
        using std::exp;
        using std::log;

        const T ratio = at.x / static_cast<T>(m_referenceTime);
        const T scaleFactor = exp((T{2} / T{3}) * log(ratio));
        return detail::flrwComponents(scaleFactor, m_curvature, at);
    }

private:
    double m_referenceTime;
    double m_curvature;
};

/// Radiation domination (w = 1/3): a(T) = (T / T0)^(1/2).
class RadiationDominatedFLRW {
public:
    RadiationDominatedFLRW(double referenceTime, double curvature)
        : m_referenceTime(referenceTime), m_curvature(curvature) {}

    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>& at) const {
        using std::sqrt;

        const T scaleFactor = sqrt(at.x / static_cast<T>(m_referenceTime));
        return detail::flrwComponents(scaleFactor, m_curvature, at);
    }

private:
    double m_referenceTime;
    double m_curvature;
};

/// A cosmological constant alone (de Sitter): a(T) = exp(H (T - T0)). H is
/// in the same inverse-length units as everything else here, since T = c t.
class LambdaDominatedFLRW {
public:
    LambdaDominatedFLRW(double hubbleRate, double referenceTime, double curvature)
        : m_hubbleRate(hubbleRate),
          m_referenceTime(referenceTime),
          m_curvature(curvature) {}

    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>& at) const {
        using std::exp;

        const T scaleFactor =
            exp(static_cast<T>(m_hubbleRate) * (at.x - static_cast<T>(m_referenceTime)));
        return detail::flrwComponents(scaleFactor, m_curvature, at);
    }

private:
    double m_hubbleRate;
    double m_referenceTime;
    double m_curvature;
};

}  // namespace ysq
