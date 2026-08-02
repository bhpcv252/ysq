#pragma once

#include <Math/Scalar.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace ysq {

/// A uniform three-dimensional grid of cell values, equal spacing on every
/// axis, with ghost cells on all six faces for whatever boundary condition
/// an owner applies.
///
/// The 3D sibling `Grid1D`'s own doc comment already names as expected
/// future work: the same role (storage, spacing, ghost cells), one more
/// dimension, for whichever PDE rung needs a genuine volume rather than a
/// line. `Physics/Spacetime`'s BSSN evolution is the first consumer; it is
/// not the only one this is built for; see `Grid1D`'s own doc comment for
/// why storage and boundary handling live here rather than in each rung
/// that uses them.
///
/// **Scope: uniform, equal spacing on every axis.** A non-uniform grid, or
/// one with a different spacing per axis, is not this; see `Grid1D`'s own
/// scope note for the same reasoning applied to one dimension.
template <Numeric T>
class Grid3D {
public:
    /// So `StateScalar` (`Math/ODE.hpp`) can recurse into this as an
    /// `OdeState`'s element type, the same role every other container state
    /// in this engine already gives it.
    using value_type = T;

    /// A trivial 1x1x1, no-ghost-cell placeholder: not meaningful on its
    /// own, but what lets a `Grid3D` be a scratch member of a generic
    /// stepper (`Math/Integrators/RK4.hpp`'s `Rk4Stepper`, which
    /// default-constructs its own `State` scratch variables before ever
    /// assigning a real one into them). Every real use replaces this via
    /// assignment before anything reads it.
    Grid3D() : Grid3D(1, 1, 1, 1.0, 0) {}

    Grid3D(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
           double spacing, std::size_t ghostCells = 1)
        : m_cellCount{cellCountX, cellCountY, cellCountZ},
          m_spacing(spacing),
          m_ghostCells(ghostCells),
          m_storageSize{cellCountX + 2 * ghostCells, cellCountY + 2 * ghostCells,
                        cellCountZ + 2 * ghostCells},
          m_values(m_storageSize[0] * m_storageSize[1] * m_storageSize[2]) {
        assert(cellCountX > 0 && cellCountY > 0 && cellCountZ > 0);
    }

    [[nodiscard]] std::size_t cellCountX() const noexcept { return m_cellCount[0]; }
    [[nodiscard]] std::size_t cellCountY() const noexcept { return m_cellCount[1]; }
    [[nodiscard]] std::size_t cellCountZ() const noexcept { return m_cellCount[2]; }
    [[nodiscard]] std::size_t ghostCells() const noexcept { return m_ghostCells; }
    [[nodiscard]] double spacing() const noexcept { return m_spacing; }

    /// (0,0,0) is the first interior cell on every axis. An index reaching
    /// up to ghostCells() past either end of an axis, negative or beyond
    /// that axis's cellCount, is valid and lands in that axis's ghost
    /// region, the same convention `Grid1D::operator[]` uses.
    [[nodiscard]] T& operator()(std::ptrdiff_t i, std::ptrdiff_t j,
                                std::ptrdiff_t k) noexcept {
        return m_values[toStorageIndex(i, j, k)];
    }
    [[nodiscard]] const T& operator()(std::ptrdiff_t i, std::ptrdiff_t j,
                                      std::ptrdiff_t k) const noexcept {
        return m_values[toStorageIndex(i, j, k)];
    }

    /// Copies each face's interior edge into the opposite face's ghost
    /// cells, one axis at a time, so a stencil reading past any of the six
    /// faces sees the domain wrap around. Matches
    /// `Grid1D::applyPeriodicBoundary`, extended to three axes; a
    /// physics-specific boundary condition (e.g. an outgoing-wave falloff
    /// for an asymptotically flat spacetime) is not this and belongs in
    /// whatever rung needs it, the same separation `Grid1D` already draws.
    void applyPeriodicBoundary() {
        applyPeriodicBoundaryAlongAxis(0);
        applyPeriodicBoundaryAlongAxis(1);
        applyPeriodicBoundaryAlongAxis(2);
    }

    /// Cell-by-cell (every stored cell, ghost regions included) vector-space
    /// operations: what a `Grid3D` needs to be an `OdeState` in its own
    /// right (`Math/ODE.hpp`), so a bundle of grid fields -- BSSN's evolved
    /// variables being the first example, not the only one this is for --
    /// can be handed straight to `Rk4Stepper` or any other general stepper
    /// in `Math/Integrators`. Two grids must share the same layout
    /// (cell counts, ghost cells); nothing here checks that beyond the
    /// assert, the same trust every other `Numeric`-constrained template in
    /// this engine already places in its caller.
    Grid3D& operator+=(const Grid3D& other) {
        assert(m_values.size() == other.m_values.size());
        for (std::size_t index = 0; index < m_values.size(); ++index) {
            m_values[index] += other.m_values[index];
        }
        return *this;
    }

    Grid3D& operator-=(const Grid3D& other) {
        assert(m_values.size() == other.m_values.size());
        for (std::size_t index = 0; index < m_values.size(); ++index) {
            m_values[index] -= other.m_values[index];
        }
        return *this;
    }

    Grid3D& operator*=(T scalar) {
        for (T& value : m_values) {
            value *= scalar;
        }
        return *this;
    }

    [[nodiscard]] friend Grid3D operator+(Grid3D lhs, const Grid3D& rhs) {
        lhs += rhs;
        return lhs;
    }

    [[nodiscard]] friend Grid3D operator-(Grid3D lhs, const Grid3D& rhs) {
        lhs -= rhs;
        return lhs;
    }

    [[nodiscard]] friend Grid3D operator*(Grid3D grid, T scalar) {
        grid *= scalar;
        return grid;
    }

    [[nodiscard]] friend Grid3D operator*(T scalar, Grid3D grid) {
        grid *= scalar;
        return grid;
    }

private:
    std::array<std::size_t, 3> m_cellCount;
    double m_spacing;
    std::size_t m_ghostCells;
    std::array<std::size_t, 3> m_storageSize;
    std::vector<T> m_values;

    [[nodiscard]] std::size_t toStorageIndex(std::ptrdiff_t i, std::ptrdiff_t j,
                                             std::ptrdiff_t k) const noexcept {
        const std::ptrdiff_t si = i + static_cast<std::ptrdiff_t>(m_ghostCells);
        const std::ptrdiff_t sj = j + static_cast<std::ptrdiff_t>(m_ghostCells);
        const std::ptrdiff_t sk = k + static_cast<std::ptrdiff_t>(m_ghostCells);
        assert(si >= 0 && si < static_cast<std::ptrdiff_t>(m_storageSize[0]));
        assert(sj >= 0 && sj < static_cast<std::ptrdiff_t>(m_storageSize[1]));
        assert(sk >= 0 && sk < static_cast<std::ptrdiff_t>(m_storageSize[2]));
        return (static_cast<std::size_t>(si) * m_storageSize[1] +
                static_cast<std::size_t>(sj)) *
                   m_storageSize[2] +
               static_cast<std::size_t>(sk);
    }

    /// One axis of applyPeriodicBoundary: walk every cell index on the
    /// other two axes (interior plus their own ghost regions, since corner
    /// and edge ghost cells need every axis's wrap applied once each) and
    /// copy that axis's interior edge into its ghost region.
    void applyPeriodicBoundaryAlongAxis(int axis) {
        const auto extent = [this](int a) {
            return static_cast<std::ptrdiff_t>(m_cellCount[static_cast<std::size_t>(a)]);
        };
        const auto ghost = static_cast<std::ptrdiff_t>(m_ghostCells);
        const int otherA = (axis == 0) ? 1 : 0;
        const int otherB = (axis == 2) ? 1 : 2;

        const std::ptrdiff_t n = extent(axis);
        const std::ptrdiff_t loA = -ghost;
        const std::ptrdiff_t hiA = extent(otherA) + ghost;
        const std::ptrdiff_t loB = -ghost;
        const std::ptrdiff_t hiB = extent(otherB) + ghost;

        for (std::ptrdiff_t a = loA; a < hiA; ++a) {
            for (std::ptrdiff_t b = loB; b < hiB; ++b) {
                for (std::ptrdiff_t g = 0; g < ghost; ++g) {
                    const std::ptrdiff_t offset = g + 1;
                    std::array<std::ptrdiff_t, 3> lowGhost{};
                    std::array<std::ptrdiff_t, 3> highSource{};
                    std::array<std::ptrdiff_t, 3> highGhost{};
                    std::array<std::ptrdiff_t, 3> lowSource{};

                    const auto setAxes = [&](std::array<std::ptrdiff_t, 3>& idx,
                                             std::ptrdiff_t alongAxis) {
                        idx[static_cast<std::size_t>(axis)] = alongAxis;
                        idx[static_cast<std::size_t>(otherA)] = a;
                        idx[static_cast<std::size_t>(otherB)] = b;
                    };
                    setAxes(lowGhost, -offset);
                    setAxes(highSource, n - offset);
                    setAxes(highGhost, n - 1 + offset);
                    setAxes(lowSource, offset - 1);

                    (*this)(lowGhost[0], lowGhost[1], lowGhost[2]) =
                        (*this)(highSource[0], highSource[1], highSource[2]);
                    (*this)(highGhost[0], highGhost[1], highGhost[2]) =
                        (*this)(lowSource[0], lowSource[1], lowSource[2]);
                }
            }
        }
    }
};

}  // namespace ysq
