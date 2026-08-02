#pragma once

#include <Math/Grid3D.hpp>
#include <Math/Scalar.hpp>

#include <array>
#include <cassert>
#include <cstddef>

namespace ysq {

/// Finite-difference stencils on a `Grid3D`, and the numerical dissipation a
/// hyperbolic evolution on one needs to stay stable.
///
/// **Why fourth order.** `Math/Calculus.hpp`'s finite differences
/// differentiate a black-box function of a point; these differentiate an
/// already-discretized field stored on a grid, the shape `Physics/Spacetime`'s
/// BSSN evolution (and any future 3D PDE rung built on `Grid3D`) actually
/// needs. Fourth order is the accuracy essentially every BSSN code in the
/// literature uses: second order is not accurate enough to resolve curvature
/// near a black hole at a tractable resolution, and going higher costs more
/// per point than it buys for a first, foundational implementation.
///
/// **Why dissipation is here, not optional.** A centered, non-dissipative
/// stencil lets the grid's own Nyquist-frequency mode grow unchecked, the
/// standard failure mode of a naive finite-difference evolution; Kreiss and
/// Oliger's fix (Kreiss & Oliger, "Methods for the approximate solution of
/// time dependent problems", 1973) damps exactly that mode without touching
/// the physical (well-resolved) part of the solution, and is what every
/// stable BSSN evolution in the literature actually adds on top of its
/// centered stencils.
///
/// Every function here is general numerical method: no spacetime, no
/// gravity, nothing BSSN-specific. `Physics/Spacetime` is the first
/// consumer, not a reason this lives there instead of here.
enum class Axis { X = 0, Y = 1, Z = 2 };

namespace detail {

/// f'(x) = sum_m coefficient[m] * f(x + (m-2) h), the standard centered
/// fourth-order first-derivative stencil (five points, error O(h^4)).
inline constexpr std::array<double, 5> kFirstDerivativeCoefficients{
    1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0};

/// f''(x) = sum_m coefficient[m] * f(x + (m-2) h), the standard centered
/// fourth-order second-derivative stencil (five points, error O(h^4)).
inline constexpr std::array<double, 5> kSecondDerivativeCoefficients{
    -1.0 / 12.0, 16.0 / 12.0, -30.0 / 12.0, 16.0 / 12.0, -1.0 / 12.0};

/// The sixth-order (2p, p=3) Kreiss-Oliger undivided difference, matching
/// what a fourth-order-accurate scheme needs to stay one order past its own
/// truncation error: epsilon/(64 h) * sum_m coefficient[m] * f(x + (m-3) h).
inline constexpr std::array<double, 7> kKreissOligerCoefficients{1.0,  -6.0, 15.0, -20.0,
                                                                  15.0, -6.0, 1.0};

/// Displaces (i, j, k) by `amount` cells along `axis` alone.
constexpr void axisOffset(Axis axis, std::ptrdiff_t amount, std::ptrdiff_t& di,
                          std::ptrdiff_t& dj, std::ptrdiff_t& dk) noexcept {
    di = 0;
    dj = 0;
    dk = 0;
    switch (axis) {
        case Axis::X:
            di = amount;
            break;
        case Axis::Y:
            dj = amount;
            break;
        case Axis::Z:
            dk = amount;
            break;
    }
}

}  // namespace detail

/// The fourth-order first partial derivative of `grid` along `axis`, at cell
/// (i, j, k). Reads two cells past (i, j, k) along `axis` in each direction,
/// so `grid`'s ghost-cell count on that axis must be at least 2.
template <Numeric T>
[[nodiscard]] T firstDerivative(const Grid3D<T>& grid, std::ptrdiff_t i, std::ptrdiff_t j,
                                std::ptrdiff_t k, Axis axis, double spacing) {
    T sum{};
    for (std::ptrdiff_t m = -2; m <= 2; ++m) {
        std::ptrdiff_t di = 0;
        std::ptrdiff_t dj = 0;
        std::ptrdiff_t dk = 0;
        detail::axisOffset(axis, m, di, dj, dk);
        const double coefficient =
            detail::kFirstDerivativeCoefficients[static_cast<std::size_t>(m + 2)];
        sum += static_cast<T>(coefficient) * grid(i + di, j + dj, k + dk);
    }
    return sum / static_cast<T>(spacing);
}

/// The fourth-order second partial derivative of `grid` along `axis`, at
/// cell (i, j, k). Same ghost-cell requirement as `firstDerivative`.
template <Numeric T>
[[nodiscard]] T secondDerivative(const Grid3D<T>& grid, std::ptrdiff_t i, std::ptrdiff_t j,
                                 std::ptrdiff_t k, Axis axis, double spacing) {
    T sum{};
    for (std::ptrdiff_t m = -2; m <= 2; ++m) {
        std::ptrdiff_t di = 0;
        std::ptrdiff_t dj = 0;
        std::ptrdiff_t dk = 0;
        detail::axisOffset(axis, m, di, dj, dk);
        const double coefficient =
            detail::kSecondDerivativeCoefficients[static_cast<std::size_t>(m + 2)];
        sum += static_cast<T>(coefficient) * grid(i + di, j + dj, k + dk);
    }
    return sum / static_cast<T>(spacing * spacing);
}

/// d^2 f / (d axisA d axisB) at (i, j, k), `axisA != axisB` (asserted): the
/// tensor product of the same fourth-order first-derivative stencil applied
/// along each axis in turn, the standard way a centered mixed partial is
/// built from two pure ones. Use `secondDerivative` instead when both axes
/// are the same.
template <Numeric T>
[[nodiscard]] T mixedSecondDerivative(const Grid3D<T>& grid, std::ptrdiff_t i,
                                      std::ptrdiff_t j, std::ptrdiff_t k, Axis axisA,
                                      Axis axisB, double spacingA, double spacingB) {
    assert(axisA != axisB);
    T sum{};
    for (std::ptrdiff_t p = -2; p <= 2; ++p) {
        std::ptrdiff_t diA = 0;
        std::ptrdiff_t djA = 0;
        std::ptrdiff_t dkA = 0;
        detail::axisOffset(axisA, p, diA, djA, dkA);
        const double coefficientA =
            detail::kFirstDerivativeCoefficients[static_cast<std::size_t>(p + 2)];

        for (std::ptrdiff_t q = -2; q <= 2; ++q) {
            std::ptrdiff_t diB = 0;
            std::ptrdiff_t djB = 0;
            std::ptrdiff_t dkB = 0;
            detail::axisOffset(axisB, q, diB, djB, dkB);
            const double coefficientB =
                detail::kFirstDerivativeCoefficients[static_cast<std::size_t>(q + 2)];

            sum += static_cast<T>(coefficientA * coefficientB) *
                   grid(i + diA + diB, j + djA + djB, k + dkA + dkB);
        }
    }
    return sum / static_cast<T>(spacingA * spacingB);
}

/// Kreiss-Oliger dissipation along one axis, at cell (i, j, k): damps the
/// grid's own Nyquist-frequency mode without corrupting a smooth solution
/// (it is itself O(h^5), one order past this module's O(h^4) stencils).
/// `sigma` is the dimensionless strength every BSSN code exposes as a
/// tunable, typically 0.1-0.5 (Baumgarte & Shapiro's convention) or as low
/// as O(0.01) (GRChombo's), tuned per resolution rather than fixed here.
/// Reads three cells past (i, j, k) along `axis`, so `grid`'s ghost-cell
/// count on that axis must be at least 3.
template <Numeric T>
[[nodiscard]] T kreissOligerDissipation(const Grid3D<T>& grid, std::ptrdiff_t i,
                                        std::ptrdiff_t j, std::ptrdiff_t k, Axis axis,
                                        double spacing, double sigma) {
    T sum{};
    for (std::ptrdiff_t m = -3; m <= 3; ++m) {
        std::ptrdiff_t di = 0;
        std::ptrdiff_t dj = 0;
        std::ptrdiff_t dk = 0;
        detail::axisOffset(axis, m, di, dj, dk);
        const double coefficient =
            detail::kKreissOligerCoefficients[static_cast<std::size_t>(m + 3)];
        sum += static_cast<T>(coefficient) * grid(i + di, j + dj, k + dk);
    }
    return static_cast<T>(sigma / (64.0 * spacing)) * sum;
}

/// The sum of `kreissOligerDissipation` over all three axes: the whole
/// dissipation term a right-hand side adds for one evolved field, one call
/// rather than three at every call site.
template <Numeric T>
[[nodiscard]] T kreissOligerDissipation3D(const Grid3D<T>& grid, std::ptrdiff_t i,
                                          std::ptrdiff_t j, std::ptrdiff_t k,
                                          double spacing, double sigma) {
    return kreissOligerDissipation(grid, i, j, k, Axis::X, spacing, sigma) +
           kreissOligerDissipation(grid, i, j, k, Axis::Y, spacing, sigma) +
           kreissOligerDissipation(grid, i, j, k, Axis::Z, spacing, sigma);
}

}  // namespace ysq
