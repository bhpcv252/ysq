#pragma once

#include <Math/Multigrid.hpp>
#include <Math/Vector3.hpp>
#include <Physics/Spacetime/ADM.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

/// One black hole's puncture parameters: ADM mass parameter, coordinate
/// position, linear momentum, and spin, in the flat conformal background
/// the Brandt-Brügmann construction assumes. `mass` is a parameter of the
/// construction (Brandt & Brügmann's `m_i`), not necessarily the puncture's
/// own irreducible or ADM mass once momentum/spin are nonzero -- for a
/// single puncture at rest with zero spin (this stage's actual scenario) it
/// is exactly the Schwarzschild mass.
struct PunctureSpec {
    double mass = 0.0;
    Vec3 position = Vec3::zero();
    Vec3 momentum = Vec3::zero();
    Vec3 spin = Vec3::zero();
};

/// Builds valid initial data for one or more punctures via the standard
/// Brandt & Brügmann (1997, gr-qc/9711015) construction: a conformally flat
/// spatial metric, Bowen-York extrinsic curvature (Bowen & York 1980),
/// which analytically satisfies the momentum constraint for any
/// mass/momentum/spin, leaving the Hamiltonian constraint as the single
/// elliptic equation solved here -- by `Math/Multigrid.hpp`'s general FAS
/// solver, not a plain relaxation loop (a single static puncture's source
/// term is zero and converges either way; a puncture with nonzero momentum,
/// as an orbiting binary needs, does not, which is why Stage 2 moved this
/// off plain Gauss-Seidel). See `Physics/README.md`'s puncture-data section
/// for the derivation and why the Hamiltonian constraint is the only
/// equation left to solve numerically.
///
/// General for any number of punctures with arbitrary mass, momentum and
/// spin: this is not specific to a single static puncture. Lapse is
/// initialized to 1 everywhere (the standard moving-puncture choice: the
/// 1+log slicing condition, not this initial value, is what settles the
/// lapse into its physically correct profile over the evolution's own
/// transient); shift and its Gamma-driver auxiliary start at zero.
///
/// `cellCountX/Y/Z` must be even and divide down to at least
/// `settings.coarsestCellCount` for the multigrid solve to reach its
/// coarsest level as intended; see `Math/Multigrid.hpp`.
/// The built `AdmData`, and diagnostics a caller can use to confirm the
/// solver actually converged rather than merely stopping at
/// `settings.maxVCycles`. `iterationsUsed` is the multigrid V-cycle count
/// (not point-relaxation sweeps -- those happen many times per V-cycle).
struct PunctureInitialDataResult {
    AdmData adm;
    int iterationsUsed = 0;
    double finalResidual = 0.0;
    bool converged = false;
};

[[nodiscard]] PunctureInitialDataResult solvePunctureInitialData(
    const std::vector<PunctureSpec>& punctures, std::size_t cellCountX,
    std::size_t cellCountY, std::size_t cellCountZ, double spacing,
    std::size_t ghostCells, const MultigridSettings& settings = {});

/// The Newtonian circular-orbit estimate for the tangential ADM momentum
/// magnitude of each puncture in a two-body binary at coordinate separation
/// `separation`: `P = mu * sqrt(M / separation)`, `mu = mass1 mass2 / M` the
/// reduced mass, `M = mass1 + mass2` the total mass (geometric units).
///
/// The standard, simple starting point real numerical-relativity papers
/// themselves use before any iterative refinement (Cook's effective-
/// potential method, post-Newtonian momenta, eccentricity reduction);
/// genuinely accurate quasi-circular initial data construction is its own
/// research topic and not approximated further here. Each puncture gets
/// momentum of this magnitude, tangential to the separation vector,
/// opposite in sign to the other's -- how that's laid out in a specific
/// scenario (which axis is the separation, which is tangential) is scenario
/// construction, not this function's concern.
[[nodiscard]] double newtonianCircularMomentum(double mass1, double mass2, double separation);

}  // namespace ysq
