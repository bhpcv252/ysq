#pragma once

#include <Math/Scalar.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace ysq {

/// A uniform one-dimensional grid of cell values, spacing `spacing`, with
/// ghost cells on each side for whatever boundary condition an owner
/// applies.
///
/// This is the piece Physics/Electromagnetism's FDTD rung,
/// Physics/Fluids' Eulerian rung, and Physics/Thermodynamics' heat-equation
/// rung all build on: one grid abstraction, three different update
/// equations evaluated on it, since the domain-neutral part, storage,
/// spacing and ghost cells, is identical between them. That is the same
/// reason all three already share Math/ODE.hpp's integrators rather than
/// each writing their own stepping loop.
///
/// **Scope: one dimension.** A full 3D solver is future work for each of
/// the PDE rungs built on this. This is the tractable first version of
/// each: real field, fluid or heat evolution over space and time, not a
/// toy, just restricted to one spatial axis.
template <Numeric T>
class Grid1D {
public:
    Grid1D(std::size_t cellCount, double spacing, std::size_t ghostCells = 1)
        : m_spacing(spacing),
          m_ghostCells(ghostCells),
          m_values(cellCount + 2 * ghostCells) {
        assert(cellCount > 0);
    }

    [[nodiscard]] std::size_t cellCount() const noexcept {
        return m_values.size() - 2 * m_ghostCells;
    }
    [[nodiscard]] std::size_t ghostCells() const noexcept { return m_ghostCells; }
    [[nodiscard]] double spacing() const noexcept { return m_spacing; }

    /// Index 0 is the first interior cell. Negative indices reach into the
    /// left ghost region and indices >= cellCount() into the right, both
    /// valid up to ghostCells() past the interior.
    [[nodiscard]] T& operator[](std::ptrdiff_t index) noexcept {
        return m_values[toStorageIndex(index)];
    }
    [[nodiscard]] const T& operator[](std::ptrdiff_t index) const noexcept {
        return m_values[toStorageIndex(index)];
    }

    /// Copies each side's interior edge into the opposite side's ghost
    /// cells, so a stencil reading past either edge sees the domain wrap
    /// around.
    void applyPeriodicBoundary() {
        const auto n = static_cast<std::ptrdiff_t>(cellCount());
        for (std::size_t g = 0; g < m_ghostCells; ++g) {
            const auto offset = static_cast<std::ptrdiff_t>(g) + 1;
            (*this)[-offset] = (*this)[n - offset];
            (*this)[n - 1 + offset] = (*this)[offset - 1];
        }
    }

private:
    double m_spacing;
    std::size_t m_ghostCells;
    std::vector<T> m_values;

    [[nodiscard]] std::size_t toStorageIndex(std::ptrdiff_t index) const noexcept {
        const auto shifted = index + static_cast<std::ptrdiff_t>(m_ghostCells);
        assert(shifted >= 0 && shifted < static_cast<std::ptrdiff_t>(m_values.size()));
        return static_cast<std::size_t>(shifted);
    }
};

}  // namespace ysq
