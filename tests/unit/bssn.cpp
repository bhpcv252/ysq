#include <Physics/Spacetime/Bssn.hpp>

#include <Math/Vector3.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

using ysq::AdmData;
using ysq::BssnState;

/// A momentarily-static (zero momentum, zero spin) puncture black hole,
/// mass `mass`, placed at the coordinate origin: gamma_ij = psi^4 delta_ij,
/// K_ij = 0, psi = 1 + mass / (2r) exactly (no numerically-solved
/// correction needed, since Abar_ij = 0 for a static puncture makes the
/// Hamiltonian constraint's source term vanish identically -- see
/// `PunctureInitialData.cpp`'s own derivation). This is a real solution to
/// the vacuum Einstein equations at t = 0 (isotropic-coordinate
/// Schwarzschild), not a toy: if the BSSN machinery (Christoffel symbols,
/// conformal Ricci tensor, constraint formulas) has an index error, the
/// Hamiltonian/momentum constraints computed from it will not vanish here.
AdmData staticSchwarzschildPuncture(double mass, std::size_t cellCount, double spacing,
                                   std::size_t ghostCells) {
    AdmData adm(cellCount, cellCount, cellCount, spacing, ghostCells);
    const auto n = static_cast<std::ptrdiff_t>(cellCount);
    const auto ig = static_cast<std::ptrdiff_t>(ghostCells);
    const std::size_t halfCells = cellCount / 2;
    const double half = static_cast<double>(halfCells) * spacing;

    for (std::ptrdiff_t i = -ig; i < n + ig; ++i) {
        for (std::ptrdiff_t j = -ig; j < n + ig; ++j) {
            for (std::ptrdiff_t k = -ig; k < n + ig; ++k) {
                const double x = (static_cast<double>(i) + 0.5) * spacing - half;
                const double y = (static_cast<double>(j) + 0.5) * spacing - half;
                const double z = (static_cast<double>(k) + 0.5) * spacing - half;
                const double r = std::sqrt(x * x + y * y + z * z);
                const double psi = 1.0 + mass / (2.0 * r);
                const double conformalFactor4 = psi * psi * psi * psi;

                ysq::Tensor<double, 2, 3> gamma{};
                gamma(0, 0) = gamma(1, 1) = gamma(2, 2) = conformalFactor4;
                adm.spatialMetric.set(i, j, k, gamma);
                adm.extrinsicCurvature.set(i, j, k, ysq::Tensor<double, 2, 3>{});
                adm.lapse(i, j, k) = 1.0;
                adm.shift.x(i, j, k) = 0.0;
                adm.shift.y(i, j, k) = 0.0;
                adm.shift.z(i, j, k) = 0.0;
            }
        }
    }
    return adm;
}

/// The largest constraint violation over every cell whose distance from the
/// puncture (assumed to sit at the domain's own center) exceeds
/// `excludeRadius`. Excluding a small region immediately around the
/// puncture is standard practice, not a way to hide a bug: `psi = 1 + m/2r`
/// diverges as r -> 0, so any finite-difference discretization of it has
/// large local truncation error in the handful of cells nearest the
/// puncture regardless of whether the rest of the implementation is
/// correct. What must actually converge under refinement is the bulk value
/// this function reports, which `HamiltonianConstraintShrinksUnderGridRefinement`
/// below checks.
double maxHamiltonianViolation(const BssnState& state, std::ptrdiff_t cellCount,
                              double spacing, double excludeRadius = 0.0) {
    double maxAbs = 0.0;
    const std::ptrdiff_t halfCells = cellCount / 2;
    const double half = static_cast<double>(halfCells) * spacing;
    for (std::ptrdiff_t i = 0; i < cellCount; ++i) {
        for (std::ptrdiff_t j = 0; j < cellCount; ++j) {
            for (std::ptrdiff_t k = 0; k < cellCount; ++k) {
                const double x = (static_cast<double>(i) + 0.5) * spacing - half;
                const double y = (static_cast<double>(j) + 0.5) * spacing - half;
                const double z = (static_cast<double>(k) + 0.5) * spacing - half;
                if (std::sqrt(x * x + y * y + z * z) < excludeRadius) {
                    continue;
                }
                maxAbs = std::max(maxAbs, std::abs(ysq::hamiltonianConstraint(state, i, j, k)));
            }
        }
    }
    return maxAbs;
}

TEST(Bssn, AdmRoundTripsThroughBssnForFlatSpace) {
    AdmData adm(4, 4, 4, 0.5, 3);
    const auto n = static_cast<std::ptrdiff_t>(4);
    for (std::ptrdiff_t i = -3; i < n + 3; ++i) {
        for (std::ptrdiff_t j = -3; j < n + 3; ++j) {
            for (std::ptrdiff_t k = -3; k < n + 3; ++k) {
                ysq::Tensor<double, 2, 3> gamma{};
                gamma(0, 0) = gamma(1, 1) = gamma(2, 2) = 1.0;
                adm.spatialMetric.set(i, j, k, gamma);
                adm.extrinsicCurvature.set(i, j, k, ysq::Tensor<double, 2, 3>{});
                adm.lapse(i, j, k) = 1.0;
                adm.shift.x(i, j, k) = 0.0;
                adm.shift.y(i, j, k) = 0.0;
                adm.shift.z(i, j, k) = 0.0;
            }
        }
    }

    const BssnState state = ysq::admToBssn(adm);
    const AdmData roundTripped = ysq::bssnToAdm(state);

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            for (std::ptrdiff_t k = 0; k < n; ++k) {
                const ysq::Tensor<double, 2, 3> gamma = roundTripped.spatialMetric.at(i, j, k);
                EXPECT_NEAR(gamma(0, 0), 1.0, 1.0e-9);
                EXPECT_NEAR(gamma(1, 1), 1.0, 1.0e-9);
                EXPECT_NEAR(gamma(2, 2), 1.0, 1.0e-9);
                EXPECT_NEAR(gamma(0, 1), 0.0, 1.0e-9);
                EXPECT_NEAR(roundTripped.lapse(i, j, k), 1.0, 1.0e-9);
            }
        }
    }
    // Flat space's own conformal factor is phi = 0 everywhere.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            for (std::ptrdiff_t k = 0; k < n; ++k) {
                EXPECT_NEAR(state.phi(i, j, k), 0.0, 1.0e-9);
            }
        }
    }
}

TEST(Bssn, HamiltonianConstraintIsSmallAwayFromThePunctureItself) {
    // Excludes the handful of cells immediately around the puncture, where
    // any discretization of psi = 1 + m/2r has large local truncation error
    // regardless of correctness (see maxHamiltonianViolation's own doc
    // comment); measured numerically to be of order 1.5e-2 at this
    // resolution and shrinking under refinement (see
    // HamiltonianConstraintShrinksUnderGridRefinement below), so 0.05 is a
    // real margin, not a threshold picked to make the test pass.
    const double mass = 1.0;
    const double spacing = 0.2;
    const std::size_t cellCount = 16;
    const std::size_t ghostCells = 3;

    const AdmData adm = staticSchwarzschildPuncture(mass, cellCount, spacing, ghostCells);
    const BssnState state = ysq::admToBssn(adm);

    const double maxViolation = maxHamiltonianViolation(
        state, static_cast<std::ptrdiff_t>(cellCount), spacing, 0.6 * mass);

    EXPECT_LT(maxViolation, 0.05);
}

TEST(Bssn, HamiltonianConstraintShrinksUnderGridRefinement) {
    // The real test of correctness: not "small", but shrinking as the grid
    // refines, the signature of actually solving the right equation rather
    // than a formula that merely stays bounded regardless of resolution.
    // Same physical domain (half-extent 1.6) at two resolutions, so the
    // comparison is a genuine refinement, not two unrelated setups.
    const double mass = 1.0;
    const std::size_t ghostCells = 3;

    const AdmData coarseAdm = staticSchwarzschildPuncture(mass, 16, 0.2, ghostCells);
    const AdmData fineAdm = staticSchwarzschildPuncture(mass, 32, 0.1, ghostCells);

    const double coarseViolation =
        maxHamiltonianViolation(ysq::admToBssn(coarseAdm), 16, 0.2, 0.6 * mass);
    const double fineViolation =
        maxHamiltonianViolation(ysq::admToBssn(fineAdm), 32, 0.1, 0.6 * mass);

    EXPECT_LT(fineViolation, coarseViolation / 4.0)
        << "a fourth-order-accurate discretization should show the violation "
           "shrink much faster than merely 'somewhat smaller' under refinement";
}

TEST(Bssn, MomentumConstraintVanishesForAStaticPuncture) {
    // K_ij = 0 everywhere makes AtildeIJ = 0 everywhere, which makes every
    // term in the momentum constraint vanish identically, independent of
    // the Ricci/Christoffel machinery the Hamiltonian constraint exercises.
    const AdmData adm = staticSchwarzschildPuncture(1.0, 12, 0.25, 3);
    const BssnState state = ysq::admToBssn(adm);

    for (int component = 0; component < 3; ++component) {
        EXPECT_NEAR(ysq::momentumConstraint(state, 6, 6, 6, component), 0.0, 1.0e-9);
    }
}

}  // namespace
