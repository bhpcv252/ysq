#pragma once

#include <Math/Grid3D.hpp>
#include <Physics/Spacetime/ADM.hpp>

#include <cstddef>

namespace ysq {

/// The Baumgarte-Shapiro-Shibata-Nakamura reformulation of the ADM
/// equations (Baumgarte & Shapiro, Phys. Rev. D 59, 024007 (1998); Shibata &
/// Nakamura, Phys. Rev. D 52, 5428 (1995)): the strongly hyperbolic system
/// every working numerical-relativity code evolves, because plain ADM is
/// not. `src/Physics/README.md`'s 3+1/BSSN section has the full derivation
/// and every equation implemented here, cited individually; this header is
/// the summary.
///
/// **Conformal decomposition.** The physical spatial metric splits as
/// `gamma_ij = e^{4 phi} gammaTilde_ij`, with `phi` chosen so
/// `det(gammaTilde_ij) = 1`. The extrinsic curvature splits into its trace
/// `K` and a conformally rescaled trace-free part
/// `AtildeIJ = e^{-4 phi} (K_ij - (1/3) gamma_ij K)`. The Ricci tensor's
/// second-derivative part is computed from the conformal connection
/// functions `GammaTilde^i` (itself an independent evolved variable, the
/// specific promotion that makes this system strongly hyperbolic where raw
/// ADM is not) rather than directly from second derivatives of
/// `gammaTilde_ij`.
///
/// **Moving-puncture gauge**: 1+log slicing for the lapse, a Gamma-driver
/// for the shift, the combination that made dynamical black-hole evolution
/// numerically tractable in the first place (Campanelli et al. 2006; Baker
/// et al. 2006; van Meter et al. 2006). The Gamma-driver here omits the
/// advective (beta^j d_j B^i / beta^j d_j GammaTilde^i) terms some
/// implementations add: those mainly help track a puncture moving quickly
/// across the grid (an orbiting or merging binary), not the stability of
/// evolution itself, so they are not needed for this module's own scope
/// and can be added if a future consumer's scenario needs them.
struct BssnState {
    using value_type = double;

    Grid3D<double> phi;
    SymmetricSpatialTensorFields conformalMetric;   // gammaTilde_ij
    Grid3D<double> traceExtrinsicCurvature;         // K
    SymmetricSpatialTensorFields conformalTracelessExtrinsicCurvature;  // AtildeIJ
    SpatialVectorFields conformalConnection;        // GammaTilde^i
    Grid3D<double> lapse;                           // alpha
    SpatialVectorFields shift;                      // beta^i
    SpatialVectorFields shiftAuxiliary;              // B^i, the Gamma-driver's own variable

    /// See `Grid3D`'s own default constructor: `Rk4Stepper<BssnState>`
    /// default-constructs its scratch members before ever assigning a real
    /// state into them.
    BssnState() = default;

    BssnState(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
             double spacing, std::size_t ghostCells)
        : phi(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          conformalMetric(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          traceExtrinsicCurvature(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          conformalTracelessExtrinsicCurvature(cellCountX, cellCountY, cellCountZ, spacing,
                                               ghostCells),
          conformalConnection(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          lapse(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          shift(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          shiftAuxiliary(cellCountX, cellCountY, cellCountZ, spacing, ghostCells) {}

    /// Component-by-component, delegating to each field's own operator: what
    /// makes this an `OdeState` `Math/Integrators/RK4.hpp`'s `Rk4Stepper`
    /// can run directly, the same Method-of-Lines boundary
    /// `Mechanics/Dynamics.hpp`'s `NBodyState` already is for gravity.
    BssnState& operator+=(const BssnState& other) {
        phi += other.phi;
        conformalMetric += other.conformalMetric;
        traceExtrinsicCurvature += other.traceExtrinsicCurvature;
        conformalTracelessExtrinsicCurvature += other.conformalTracelessExtrinsicCurvature;
        conformalConnection += other.conformalConnection;
        lapse += other.lapse;
        shift += other.shift;
        shiftAuxiliary += other.shiftAuxiliary;
        return *this;
    }

    BssnState& operator-=(const BssnState& other) {
        phi -= other.phi;
        conformalMetric -= other.conformalMetric;
        traceExtrinsicCurvature -= other.traceExtrinsicCurvature;
        conformalTracelessExtrinsicCurvature -= other.conformalTracelessExtrinsicCurvature;
        conformalConnection -= other.conformalConnection;
        lapse -= other.lapse;
        shift -= other.shift;
        shiftAuxiliary -= other.shiftAuxiliary;
        return *this;
    }

    BssnState& operator*=(double scalar) {
        phi *= scalar;
        conformalMetric *= scalar;
        traceExtrinsicCurvature *= scalar;
        conformalTracelessExtrinsicCurvature *= scalar;
        conformalConnection *= scalar;
        lapse *= scalar;
        shift *= scalar;
        shiftAuxiliary *= scalar;
        return *this;
    }

    [[nodiscard]] friend BssnState operator+(BssnState lhs, const BssnState& rhs) {
        lhs += rhs;
        return lhs;
    }
    [[nodiscard]] friend BssnState operator-(BssnState lhs, const BssnState& rhs) {
        lhs -= rhs;
        return lhs;
    }
    [[nodiscard]] friend BssnState operator*(BssnState state, double scalar) {
        state *= scalar;
        return state;
    }
    [[nodiscard]] friend BssnState operator*(double scalar, BssnState state) {
        state *= scalar;
        return state;
    }
};

/// Tunables every BSSN evolution exposes; nothing here is scenario-specific.
struct BssnParameters {
    /// Kreiss-Oliger dissipation strength; see
    /// `Math/FiniteDifference.hpp::kreissOligerDissipation`.
    double kreissOligerSigma = 0.1;

    /// The Gamma-driver shift condition's damping coefficient (dimension
    /// 1/length in these geometric units), `eta` in
    /// `dtB^i = dtGammaTilde^i - eta B^i`. The standard "roughly 1-2 divided
    /// by the mass" heuristic; tuned per scenario.
    double gammaDriverEta = 1.0;
};

/// Converts primitive ADM data (a puncture solver's natural output) into
/// this module's evolved BSSN variables. `admToBssn` and `bssnToAdm` are
/// exact inverses of one another away from the conformal-factor coordinate
/// singularity a puncture itself sits on.
[[nodiscard]] BssnState admToBssn(const AdmData& adm);
[[nodiscard]] AdmData bssnToAdm(const BssnState& state);

/// The BSSN right-hand side: `d(state)/dt = bssnRhs(state, spacing, params)`,
/// a Method-of-Lines system (`Math/ODE.hpp`'s `OdeSystem` shape) evaluated
/// over every interior cell. `state`'s ghost-cell count must be at least 3
/// (the widest stencil `Math/FiniteDifference.hpp` uses, Kreiss-Oliger
/// dissipation); callers apply their own boundary condition to the ghost
/// cells before each evaluation (see `Physics/README.md`'s note on why a
/// generic periodic wrap is not appropriate for an asymptotically flat
/// spacetime, unlike `Grid3D::applyPeriodicBoundary`'s intended use).
[[nodiscard]] BssnState bssnRhs(const BssnState& state, BssnParameters params);

/// The Hamiltonian constraint's violation at one cell -- should be zero for
/// an exact solution, small and shrinking under grid refinement for a
/// numerically converged one. `Physics/README.md`'s BSSN section has the
/// derivation.
[[nodiscard]] double hamiltonianConstraint(const BssnState& state, std::ptrdiff_t i,
                                          std::ptrdiff_t j, std::ptrdiff_t k);

/// The momentum constraint's violation at one cell, one component per call
/// (`component` 0/1/2 for x/y/z); same role as `hamiltonianConstraint`.
[[nodiscard]] double momentumConstraint(const BssnState& state, std::ptrdiff_t i,
                                       std::ptrdiff_t j, std::ptrdiff_t k,
                                       int component);

}  // namespace ysq
