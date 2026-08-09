#include <Physics/Continuum/ElasticChain1D.hpp>

#include <Math/LinearSolve.hpp>

#include <cassert>

namespace ysq {

ElasticChain1D::ElasticChain1D(std::vector<SpringConstant> segmentStiffness) {
    m_stiffness.reserve(segmentStiffness.size());
    for (const SpringConstant& k : segmentStiffness) {
        m_stiffness.push_back(k.value());
    }
}

std::size_t ElasticChain1D::nodeCount() const noexcept {
    return m_stiffness.size() + 1;
}

std::optional<std::vector<Length>>
ElasticChain1D::solveDisplacements(std::span<const Force> load) const {
    const std::size_t n = nodeCount();
    assert(load.size() == n);

    if (n < 2) {
        return std::vector<Length>(n, Length{0.0});
    }

    // Node 0 is fixed, so only nodes 1..n-1 are free degrees of freedom;
    // free index i - 1 corresponds to global node i. Assembling directly
    // into this reduced system, rather than the full n x n stiffness
    // matrix followed by deleting row/column 0, is exact here because the
    // fixed displacement is exactly zero: a segment's coupling term to
    // node 0 would otherwise move `-k * u_0` to the load vector, which
    // vanishes since `u_0 = 0`.
    const std::size_t freeCount = n - 1;
    MatrixN<double> stiffnessMatrix(freeCount, freeCount, 0.0);
    VectorN<double> loadVector(freeCount);

    for (std::size_t segment = 0; segment < m_stiffness.size(); ++segment) {
        const double k = m_stiffness[segment];
        const std::size_t a = segment;
        const std::size_t b = segment + 1;

        if (a >= 1) {
            stiffnessMatrix(a - 1, a - 1) += k;
            stiffnessMatrix(a - 1, b - 1) -= k;
            stiffnessMatrix(b - 1, a - 1) -= k;
        }
        stiffnessMatrix(b - 1, b - 1) += k;
    }

    for (std::size_t i = 1; i < n; ++i) {
        loadVector[i - 1] = load[i].value();
    }

    const std::optional<VectorN<double>> solved = solve(stiffnessMatrix, loadVector);
    if (!solved) {
        return std::nullopt;
    }

    std::vector<Length> displacements(n);
    displacements[0] = Length{0.0};
    for (std::size_t i = 1; i < n; ++i) {
        displacements[i] = Length{(*solved)[i - 1]};
    }
    return displacements;
}

}  // namespace ysq
