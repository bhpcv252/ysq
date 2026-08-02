#pragma once

#include <Math/Grid3D.hpp>
#include <Math/Tensor.hpp>

#include <cstddef>

namespace ysq {

/// The 3+1 (ADM) split of spacetime: a spatial slice's own metric and its
/// extrinsic curvature (how that slice is embedded in spacetime), plus the
/// lapse and shift that say how to step from one slice to the next. This is
/// the general "spacetime that evolves" counterpart to the fixed, prescribed
/// metrics in `Minkowski.hpp`/`Schwarzschild.hpp`/`Kerr.hpp`/`FLRW.hpp`: a
/// real sibling in this module, not built for one scenario. See
/// `src/Physics/README.md`'s 3+1/BSSN section for the derivation
/// (Gourgoulhon, arXiv:gr-qc/0703035).
///
/// **Storage is structure-of-arrays, one `Grid3D<double>` per independent
/// tensor component**, not an array of `Tensor` per grid point: this is how
/// every real BSSN code is actually built (each evolution equation is a
/// scalar PDE for one component, referencing neighboring grid values of
/// several other component fields), and it is what
/// `Math/FiniteDifference.hpp`'s stencils operate on directly. Wherever
/// pointwise tensor algebra is needed (inverting the spatial metric, taking
/// a trace), `Math/Tensor.hpp`'s `Tensor<double, 2, 3>` is built from the six
/// components at that one point, used, and discarded -- the same pattern
/// `Physics/Spacetime/Metric.hpp` already uses for its 4x4 `MetricTensor`.
///
/// **Symmetric tensors store six components, not nine.** `xx, xy, xz, yy,
/// yz, zz`; the missing three (`yx, zx, zy`) equal their symmetric partner
/// and are never stored separately, so an evolution step never has two
/// copies of the same physical quantity to accidentally desynchronize.
struct SymmetricSpatialTensorFields {
    Grid3D<double> xx;
    Grid3D<double> xy;
    Grid3D<double> xz;
    Grid3D<double> yy;
    Grid3D<double> yz;
    Grid3D<double> zz;

    /// See `Grid3D`'s own default constructor: the same "scratch member of
    /// a generic stepper" need, one level up.
    SymmetricSpatialTensorFields() = default;

    SymmetricSpatialTensorFields(std::size_t cellCountX, std::size_t cellCountY,
                                 std::size_t cellCountZ, double spacing,
                                 std::size_t ghostCells)
        : xx(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          xy(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          xz(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          yy(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          yz(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          zz(cellCountX, cellCountY, cellCountZ, spacing, ghostCells) {}

    /// The full symmetric tensor at one grid point, for pointwise algebra
    /// (inversion, trace, contraction) via `Math/Tensor.hpp`.
    [[nodiscard]] Tensor<double, 2, 3> at(std::ptrdiff_t i, std::ptrdiff_t j,
                                          std::ptrdiff_t k) const {
        Tensor<double, 2, 3> t{};
        t(0, 0) = xx(i, j, k);
        t(0, 1) = t(1, 0) = xy(i, j, k);
        t(0, 2) = t(2, 0) = xz(i, j, k);
        t(1, 1) = yy(i, j, k);
        t(1, 2) = t(2, 1) = yz(i, j, k);
        t(2, 2) = zz(i, j, k);
        return t;
    }

    /// Writes the six independent components of `value` back at one grid
    /// point, the inverse of `at`.
    void set(std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t k,
            const Tensor<double, 2, 3>& value) {
        xx(i, j, k) = value(0, 0);
        xy(i, j, k) = value(0, 1);
        xz(i, j, k) = value(0, 2);
        yy(i, j, k) = value(1, 1);
        yz(i, j, k) = value(1, 2);
        zz(i, j, k) = value(2, 2);
    }

    /// Component-by-component, delegating to each `Grid3D`'s own operator:
    /// what makes a bundle of these (BSSN's evolved tensors) an `OdeState`
    /// in its own right.
    using value_type = double;

    SymmetricSpatialTensorFields& operator+=(const SymmetricSpatialTensorFields& other) {
        xx += other.xx;
        xy += other.xy;
        xz += other.xz;
        yy += other.yy;
        yz += other.yz;
        zz += other.zz;
        return *this;
    }

    SymmetricSpatialTensorFields& operator-=(const SymmetricSpatialTensorFields& other) {
        xx -= other.xx;
        xy -= other.xy;
        xz -= other.xz;
        yy -= other.yy;
        yz -= other.yz;
        zz -= other.zz;
        return *this;
    }

    SymmetricSpatialTensorFields& operator*=(double scalar) {
        xx *= scalar;
        xy *= scalar;
        xz *= scalar;
        yy *= scalar;
        yz *= scalar;
        zz *= scalar;
        return *this;
    }

    [[nodiscard]] friend SymmetricSpatialTensorFields operator+(
        SymmetricSpatialTensorFields lhs, const SymmetricSpatialTensorFields& rhs) {
        lhs += rhs;
        return lhs;
    }
    [[nodiscard]] friend SymmetricSpatialTensorFields operator-(
        SymmetricSpatialTensorFields lhs, const SymmetricSpatialTensorFields& rhs) {
        lhs -= rhs;
        return lhs;
    }
    [[nodiscard]] friend SymmetricSpatialTensorFields operator*(
        SymmetricSpatialTensorFields fields, double scalar) {
        fields *= scalar;
        return fields;
    }
    [[nodiscard]] friend SymmetricSpatialTensorFields operator*(
        double scalar, SymmetricSpatialTensorFields fields) {
        fields *= scalar;
        return fields;
    }
};

/// A vector field's three components, same storage convention as
/// `SymmetricSpatialTensorFields`.
struct SpatialVectorFields {
    Grid3D<double> x;
    Grid3D<double> y;
    Grid3D<double> z;

    SpatialVectorFields() = default;

    SpatialVectorFields(std::size_t cellCountX, std::size_t cellCountY,
                        std::size_t cellCountZ, double spacing, std::size_t ghostCells)
        : x(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          y(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          z(cellCountX, cellCountY, cellCountZ, spacing, ghostCells) {}

    using value_type = double;

    SpatialVectorFields& operator+=(const SpatialVectorFields& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    SpatialVectorFields& operator-=(const SpatialVectorFields& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    SpatialVectorFields& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    [[nodiscard]] friend SpatialVectorFields operator+(SpatialVectorFields lhs,
                                                       const SpatialVectorFields& rhs) {
        lhs += rhs;
        return lhs;
    }
    [[nodiscard]] friend SpatialVectorFields operator-(SpatialVectorFields lhs,
                                                       const SpatialVectorFields& rhs) {
        lhs -= rhs;
        return lhs;
    }
    [[nodiscard]] friend SpatialVectorFields operator*(SpatialVectorFields fields,
                                                       double scalar) {
        fields *= scalar;
        return fields;
    }
    [[nodiscard]] friend SpatialVectorFields operator*(double scalar,
                                                       SpatialVectorFields fields) {
        fields *= scalar;
        return fields;
    }
};

/// The primitive ADM variables: the spatial metric `gamma_ij`, extrinsic
/// curvature `K_ij`, lapse `alpha`, and shift `beta^i`, each a field over the
/// same `Grid3D` layout. Not what `Bssn.hpp` actually evolves (see there for
/// why plain ADM is not numerically stable); this is the physical
/// representation initial data is built in, and that diagnostics (ADM mass,
/// horizon finding) read observables out of.
struct AdmData {
    SymmetricSpatialTensorFields spatialMetric;
    SymmetricSpatialTensorFields extrinsicCurvature;
    Grid3D<double> lapse;
    SpatialVectorFields shift;

    AdmData(std::size_t cellCountX, std::size_t cellCountY, std::size_t cellCountZ,
           double spacing, std::size_t ghostCells)
        : spatialMetric(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          extrinsicCurvature(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          lapse(cellCountX, cellCountY, cellCountZ, spacing, ghostCells),
          shift(cellCountX, cellCountY, cellCountZ, spacing, ghostCells) {}
};

}  // namespace ysq
