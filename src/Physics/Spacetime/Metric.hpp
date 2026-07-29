#pragma once

#include <Math/Dual.hpp>
#include <Math/Matrix4.hpp>
#include <Math/Scalar.hpp>
#include <Math/Tensor.hpp>
#include <Math/Vector4.hpp>
#include <Units/Constants.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace ysq {

/// A metric assigns a symmetric rank-2 tensor g_mu_nu to every spacetime
/// point, in whatever coordinate chart the metric itself defines. That
/// tensor is all a metric is: everything else here, Christoffel symbols and
/// geodesics, is built from it alone.
///
/// **Convention, fixed across every metric in this module.** Signature
/// (-,+,+,+). Four-position components are (x0, x1, x2, x3) with x0 = c t,
/// so every component is in metres and no metric needs its own unit
/// conversion; src/Physics/README.md has the full statement. Which spatial
/// coordinates x1..x3 mean, Cartesian or spherical, is each metric's own
/// choice of chart, documented on the metric itself: the geodesic solver
/// only ever calls components(), so it does not need to know.
///
/// Components must be evaluable at any Numeric scalar, not just double.
/// Seeding Dual<double> through the same components() a plain evaluation
/// uses is how christoffelSymbols() below gets exact metric derivatives with
/// no finite-difference truncation, the same trick Math/Calculus.hpp's
/// gradient and jacobian use for Vector-shaped results; Tensor is not one of
/// those, which is why this module has its own differentiation rather than
/// reusing theirs.
template <class M>
concept SpacetimeMetric = requires(const M& metric, const Vector4<double>& at,
                                   const Vector4<Dual<double>>& dualAt) {
    { metric.components(at) } -> std::same_as<MetricTensor<double>>;
    { metric.components(dualAt) } -> std::same_as<MetricTensor<Dual<double>>>;
};

/// g_mu_nu(at) u^mu v^nu: the metric's inner product of two four-vectors at
/// a point. Negative for a timelike pair, positive for spacelike, zero for
/// null, in the (-,+,+,+) signature this module uses throughout.
template <SpacetimeMetric M>
[[nodiscard]] double metricProduct(const M& metric, const Vector4<double>& at,
                                   const Vector4<double>& u, const Vector4<double>& v) {
    const MetricTensor<double> g = metric.components(at);
    double total = 0.0;
    for (std::size_t mu = 0; mu < 4; ++mu) {
        for (std::size_t nu = 0; nu < 4; ++nu) {
            total += g(mu, nu) * u[mu] * v[nu];
        }
    }
    return total;
}

/// The causal character of a direction at a point, by the sign of its
/// self-product. isNull is a tolerance test rather than an exact-zero one,
/// since a numerically constructed null vector is null only up to rounding.
template <SpacetimeMetric M>
[[nodiscard]] bool isTimelike(const M& metric, const Vector4<double>& at,
                              const Vector4<double>& u) {
    return metricProduct(metric, at, u, u) < 0.0;
}

template <SpacetimeMetric M>
[[nodiscard]] bool isSpacelike(const M& metric, const Vector4<double>& at,
                               const Vector4<double>& u) {
    return metricProduct(metric, at, u, u) > 0.0;
}

template <SpacetimeMetric M>
[[nodiscard]] bool isNull(const M& metric, const Vector4<double>& at,
                          const Vector4<double>& u, double tolerance = 1e-9) {
    const double c = constants::speedOfLight.value();
    return std::abs(metricProduct(metric, at, u, u)) <= tolerance * c * c;
}

namespace detail {

/// d g_mu_nu / d x^alpha for every alpha, mu, nu: one Dual-seeded evaluation
/// of the whole metric per coordinate direction, four in total.
template <SpacetimeMetric M>
[[nodiscard]] std::array<MetricTensor<double>, 4>
metricPartials(const M& metric, const Vector4<double>& at) {
    std::array<MetricTensor<double>, 4> result{};

    for (std::size_t alpha = 0; alpha < 4; ++alpha) {
        Vector4<Dual<double>> seeded{Dual<double>{at.x}, Dual<double>{at.y},
                                     Dual<double>{at.z}, Dual<double>{at.w}};
        seeded[alpha].derivative = 1.0;

        const MetricTensor<Dual<double>> gDual = metric.components(seeded);
        for (std::size_t flat = 0; flat < MetricTensor<double>::size(); ++flat) {
            result[alpha][flat] = gDual[flat].derivative;
        }
    }
    return result;
}

}  // namespace detail

/// Gamma^lambda_mu_nu = (1/2) g^lambda_sigma (d_mu g_sigma_nu + d_nu
/// g_sigma_mu - d_sigma g_mu_nu), exactly, via metricPartials above rather
/// than a finite difference.
template <SpacetimeMetric M>
[[nodiscard]] ChristoffelSymbols<double> christoffelSymbols(const M& metric,
                                                            const Vector4<double>& at) {
    const MetricTensor<double> g = metric.components(at);
    const Matrix4<double> gInverseMatrix = inverse(toMatrix4(g));
    const MetricTensor<double> gInverse = toTensor(gInverseMatrix);
    const std::array<MetricTensor<double>, 4> dg = detail::metricPartials(metric, at);

    ChristoffelSymbols<double> gamma{};
    for (std::size_t lambda = 0; lambda < 4; ++lambda) {
        for (std::size_t mu = 0; mu < 4; ++mu) {
            for (std::size_t nu = 0; nu < 4; ++nu) {
                double sum = 0.0;
                for (std::size_t sigma = 0; sigma < 4; ++sigma) {
                    sum += gInverse(lambda, sigma) *
                           (dg[mu](sigma, nu) + dg[nu](sigma, mu) - dg[sigma](mu, nu));
                }
                gamma(lambda, mu, nu) = 0.5 * sum;
            }
        }
    }
    return gamma;
}

}  // namespace ysq
