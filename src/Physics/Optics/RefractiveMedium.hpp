#pragma once

#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>

#include <cmath>

namespace ysq {

/// A static, spherically symmetric medium of refractive index n(r), exposed
/// as a SpacetimeMetric: not because it curves spacetime, it does not, but
/// because a static medium's ray paths (Fermat's principle: extremal
/// optical path length, integral n dl) are exactly the spatial projections
/// of null geodesics of the "ultra-static" metric
///
///     ds^2 = -dT^2 + n(r)^2 (dr^2 + r^2 dpolar^2 + r^2 sin^2(polar) dazimuth^2)
///
/// the standard optical-metric construction (Gordon 1923's moving-medium
/// metric, specialized to a medium at rest). g_TT = -1 exactly, so T does
/// not appear in any component: the metric has a Killing vector along T,
/// and the null condition (dT)^2 = n(r)^2 dl^2 makes T itself, along any
/// null geodesic, equal to the optical path length traveled so far.
///
/// This is not a second solver: Spacetime/Geodesic.hpp's unmodified
/// geodesic equation, Optics/Propagation.hpp's nullTangent() and
/// propagate(), and Optics/Lensing.hpp's shooting technique all apply to
/// this metric exactly as they do to Schwarzschild, because nothing in any
/// of them assumes which metric it is being handed. Atmospheric refraction
/// and gravitational lensing are, in this engine as in the physics, the
/// same computation: a null geodesic, evaluated against a different
/// metric. See src/Physics/README.md for the derivation and the
/// unit test that validates it against the real, measured horizontal
/// (grazing) refraction angle.
///
/// n(r) itself is a simple exponential profile, `1 + surfaceRefractivity *
/// exp(-(r - radius) / scaleHeight)`: the same shape Gladstone-Dale (n - 1
/// proportional to density) gives for any isothermal barometric atmosphere,
/// Physics/Thermodynamics/Thermodynamics.hpp's isothermalAtmosphereDensity.
/// This class does not depend on Thermodynamics itself, or reproduce its
/// derivation: `surfaceRefractivity` and `scaleHeight` are two plain
/// numbers a scenario supplies, however it derived them, general for any
/// planet's atmosphere, not Earth's specifically.
class RefractiveMedium {
public:
    /// Vacuum: surfaceRefractivity zero, so refractiveIndex() is 1
    /// everywhere regardless of scaleHeight. Lets a RefractiveMedium sit as
    /// a plain, harmlessly-inert member of a larger aggregate (see
    /// Optics/Illumination.hpp's RefractingOccluder) before it is given a
    /// body's real numbers.
    RefractiveMedium() = default;

    /// `radius` is the body's own radius (r = radius is the medium's
    /// "surface", where n = 1 + surfaceRefractivity exactly);
    /// `surfaceRefractivity` is n - 1 there; `scaleHeight` is the altitude
    /// over which surfaceRefractivity falls by a factor of e.
    RefractiveMedium(double radius, double surfaceRefractivity, double scaleHeight)
        : m_radius(radius),
          m_surfaceRefractivity(surfaceRefractivity),
          m_scaleHeight(scaleHeight) {}

    [[nodiscard]] double radius() const noexcept { return m_radius; }
    [[nodiscard]] double surfaceRefractivity() const noexcept {
        return m_surfaceRefractivity;
    }
    [[nodiscard]] double scaleHeight() const noexcept { return m_scaleHeight; }

    /// The refractive index at radial coordinate `r`, for any `r`, not only
    /// r >= radius(): the exponential is smooth and finite everywhere, and
    /// it is Optics/Illumination.hpp's job, not this metric's, to decide
    /// that a ray reaching r < radius() has hit the opaque body this
    /// atmosphere surrounds rather than to guard against it here.
    template <Numeric T>
    [[nodiscard]] T refractiveIndex(T r) const {
        using std::exp;
        return T{1} +
               static_cast<T>(m_surfaceRefractivity) *
                   exp(-(r - static_cast<T>(m_radius)) / static_cast<T>(m_scaleHeight));
    }

    template <Numeric T>
    [[nodiscard]] MetricTensor<T> components(const Vector4<T>& at) const {
        using std::sin;

        const T r = at.y;
        const T polar = at.z;
        const T n = refractiveIndex(r);
        const T sinPolar = sin(polar);

        MetricTensor<T> g{};
        g(0, 0) = T{-1};
        g(1, 1) = n * n;
        g(2, 2) = n * n * r * r;
        g(3, 3) = n * n * r * r * sinPolar * sinPolar;
        return g;
    }

private:
    double m_radius = 0.0;
    double m_surfaceRefractivity = 0.0;
    double m_scaleHeight = 1.0;
};

/// A null geodesic of `medium`, in the equatorial plane, launched from
/// `startRadius` with impact parameter `impactParameter`, moving inward.
/// The same E = 1, L = impactParameter normalization
/// Optics/Lensing.hpp's schwarzschildRayFromImpactParameter uses, adapted
/// to this metric's own g_TT = -1 (constant here, unlike Schwarzschild's
/// r-dependent factor, since this medium does not curve spacetime):
/// uT = 1 always; uPhi = impactParameter / (n(startRadius)^2
/// startRadius^2); ur from the null condition, negative root. Exact, not a
/// large-startRadius approximation.
[[nodiscard]] inline PhaseState<Vector4<double>>
refractiveMediumRayFromImpactParameter(const RefractiveMedium& medium,
                                       double impactParameter, double startRadius) {
    const double n = medium.refractiveIndex(startRadius);
    const double uPhi = impactParameter / (n * n * startRadius * startRadius);
    const double urSquared = 1.0 / (n * n) - startRadius * startRadius * uPhi * uPhi;
    const double ur = -std::sqrt(urSquared);

    return PhaseState<Vector4<double>>{
        Vector4<double>{0.0, startRadius, kPi<double> / 2.0, 0.0},
        Vector4<double>{1.0, ur, 0.0, uPhi}};
}

}  // namespace ysq
