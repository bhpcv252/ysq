#include <Physics/Spacetime/Bssn.hpp>

#include <Math/FiniteDifference.hpp>
#include <Math/Matrix3.hpp>
#include <Math/Tensor.hpp>

#include <array>
#include <cmath>
#include <cstddef>

namespace ysq {

namespace {

using Sym3 = Tensor<double, 2, 3>;  // A symmetric rank-2, dimension-3 tensor.
using Chr3 = Tensor<double, 3, 3>;  // Gamma^k_ij: rank 3, dimension 3, (k, i, j).

/// The stored component for (a, b), a <= b assumed by the caller (every
/// call site here only ever asks for the canonical ordering).
[[nodiscard]] const Grid3D<double>& componentField(const SymmetricSpatialTensorFields& f,
                                                   int a, int b) {
    if (a == 0 && b == 0) return f.xx;
    if ((a == 0 && b == 1) || (a == 1 && b == 0)) return f.xy;
    if ((a == 0 && b == 2) || (a == 2 && b == 0)) return f.xz;
    if (a == 1 && b == 1) return f.yy;
    if ((a == 1 && b == 2) || (a == 2 && b == 1)) return f.yz;
    return f.zz;
}

[[nodiscard]] const Grid3D<double>& vectorComponentField(const SpatialVectorFields& f,
                                                         int a) {
    if (a == 0) return f.x;
    if (a == 1) return f.y;
    return f.z;
}

[[nodiscard]] Sym3 conformalMetricAt(const BssnState& s, std::ptrdiff_t i, std::ptrdiff_t j,
                                     std::ptrdiff_t k) {
    return s.conformalMetric.at(i, j, k);
}

[[nodiscard]] Sym3 conformalMetricInverseAt(const BssnState& s, std::ptrdiff_t i,
                                            std::ptrdiff_t j, std::ptrdiff_t k) {
    const Sym3 g = conformalMetricAt(s, i, j, k);
    return toTensor(inverse(toMatrix3(g)));
}

/// Gradient (three first partials) of a scalar field at one point.
[[nodiscard]] std::array<double, 3> gradientAt(const Grid3D<double>& field,
                                               std::ptrdiff_t i, std::ptrdiff_t j,
                                               std::ptrdiff_t k, double spacing) {
    return {firstDerivative(field, i, j, k, Axis::X, spacing),
            firstDerivative(field, i, j, k, Axis::Y, spacing),
            firstDerivative(field, i, j, k, Axis::Z, spacing)};
}

/// d(gammaTilde_ab)/d(axis m), for every m, a, b: the raw ingredient the
/// conformal Christoffel symbols are built from.
[[nodiscard]] std::array<Sym3, 3> conformalMetricPartialsAt(const BssnState& s,
                                                            std::ptrdiff_t i,
                                                            std::ptrdiff_t j,
                                                            std::ptrdiff_t k,
                                                            double spacing) {
    std::array<Sym3, 3> dg{};
    for (int m = 0; m < 3; ++m) {
        for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
                const double value = firstDerivative(componentField(s.conformalMetric, a, b),
                                                     i, j, k, static_cast<Axis>(m), spacing);
                dg[static_cast<std::size_t>(m)](a, b) = value;
                dg[static_cast<std::size_t>(m)](b, a) = value;
            }
        }
    }
    return dg;
}

/// GammaTilde^k_ij = (1/2) gammaTilde^kl (d_i gammaTilde_lj + d_j gammaTilde_li
/// - d_l gammaTilde_ij): the ordinary Christoffel symbols of the conformal
/// metric, computed from finite differences (Baumgarte & Shapiro 1998).
[[nodiscard]] Chr3 conformalChristoffelAt(const BssnState& s, std::ptrdiff_t i,
                                          std::ptrdiff_t j, std::ptrdiff_t k,
                                          double spacing) {
    const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
    const std::array<Sym3, 3> dg = conformalMetricPartialsAt(s, i, j, k, spacing);

    Chr3 christoffel{};
    for (int upper = 0; upper < 3; ++upper) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double sum = 0.0;
                for (int l = 0; l < 3; ++l) {
                    sum += ginv(upper, l) * (dg[static_cast<std::size_t>(a)](l, b) +
                                            dg[static_cast<std::size_t>(b)](l, a) -
                                            dg[static_cast<std::size_t>(l)](a, b));
                }
                christoffel(upper, a, b) = 0.5 * sum;
            }
        }
    }
    return christoffel;
}

/// The physical-metric Christoffel symbols, from the conformal ones by the
/// standard conformal-rescaling identity for gamma_ij = e^{4 phi}
/// gammaTilde_ij:
///
///   Gamma^k_ij = GammaTilde^k_ij
///              + 2 (delta^k_i d_j phi + delta^k_j d_i phi
///                   - gammaTilde_ij gammaTilde^kl d_l phi)
[[nodiscard]] Chr3 physicalChristoffelAt(const BssnState& s, std::ptrdiff_t i,
                                         std::ptrdiff_t j, std::ptrdiff_t k,
                                         double spacing) {
    const Chr3 conformal = conformalChristoffelAt(s, i, j, k, spacing);
    const Sym3 g = conformalMetricAt(s, i, j, k);
    const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
    const std::array<double, 3> dphi = gradientAt(s.phi, i, j, k, spacing);

    Chr3 result{};
    for (int upper = 0; upper < 3; ++upper) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double raisedGradPhi = 0.0;
                for (int l = 0; l < 3; ++l) {
                    raisedGradPhi += ginv(upper, l) * dphi[static_cast<std::size_t>(l)];
                }
                double shift = (upper == a ? dphi[static_cast<std::size_t>(b)] : 0.0) +
                              (upper == b ? dphi[static_cast<std::size_t>(a)] : 0.0) -
                              g(a, b) * raisedGradPhi;
                result(upper, a, b) = conformal(upper, a, b) + 2.0 * shift;
            }
        }
    }
    return result;
}

/// The covariant Hessian of a scalar field, D_i D_j f = d_i d_j f -
/// Gamma^k_ij d_k f, given whichever Christoffel symbols the caller passes
/// (physical for alpha, conformal for phi -- the same formula either way,
/// since it only depends on the connection being torsion-free).
[[nodiscard]] Sym3 covariantHessianAt(const Grid3D<double>& field, const Chr3& christoffel,
                                     std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t k,
                                     double spacing) {
    const std::array<double, 3> d1 = gradientAt(field, i, j, k, spacing);
    Sym3 result{};
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            const double d2 = (a == b) ? secondDerivative(field, i, j, k,
                                                          static_cast<Axis>(a), spacing)
                                       : mixedSecondDerivative(field, i, j, k,
                                                              static_cast<Axis>(a),
                                                              static_cast<Axis>(b), spacing,
                                                              spacing);
            double christoffelTerm = 0.0;
            for (int upper = 0; upper < 3; ++upper) {
                christoffelTerm += christoffel(upper, a, b) * d1[static_cast<std::size_t>(upper)];
            }
            const double value = d2 - christoffelTerm;
            result(a, b) = value;
            result(b, a) = value;
        }
    }
    return result;
}

/// GammaTilde_kij := gammaTilde_kl GammaTilde^l_ij: the conformal
/// Christoffel with its upper index lowered by the conformal metric. Used
/// throughout the Ricci tensor formula below, which mixes raised and
/// lowered forms of the same connection.
[[nodiscard]] double loweredChristoffel(const Sym3& g, const Chr3& christoffel, int lowered,
                                       int a, int b) {
    double sum = 0.0;
    for (int p = 0; p < 3; ++p) {
        sum += g(lowered, p) * christoffel(p, a, b);
    }
    return sum;
}

/// The conformal metric's own Ricci tensor, built from the conformal
/// connection functions GammaTilde^i (the evolved variable) rather than
/// directly from second derivatives of gammaTilde_ij -- the specific
/// promotion that makes BSSN strongly hyperbolic where raw ADM is not
/// (Baumgarte & Shapiro 1998, Eq. 3.34-3.37; Shibata & Nakamura 1995):
///
///   RtildeIJ = -(1/2) gammaTilde^lm d_l d_m gammaTilde_ij
///            + gammaTilde_k(i d_j) GammaTilde^k
///            + GammaTilde^k GammaTilde_(ij)k
///            + gammaTilde^lm [ 2 GammaTilde^k_l(i GammaTilde_j)km
///                             + GammaTilde^k_im GammaTilde_klj ]
///
/// This is the single most index-heavy expression in this file. It is
/// transcribed here as literally as possible (direct nested summation, no
/// hand-simplified closed form) precisely so a reader can check it term by
/// term against the citation above; `tests/unit/bssn.cpp` and
/// `tests/integration/single_puncture_stability.cpp` are the real proof it
/// is right, not this comment.
[[nodiscard]] Sym3 conformalRicciAt(const BssnState& s, std::ptrdiff_t i, std::ptrdiff_t j,
                                    std::ptrdiff_t k, double spacing) {
    const Sym3 g = conformalMetricAt(s, i, j, k);
    const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
    const Chr3 christoffel = conformalChristoffelAt(s, i, j, k, spacing);

    std::array<double, 3> gammaTilde{};  // the evolved GammaTilde^k itself, at this point
    gammaTilde[0] = s.conformalConnection.x(i, j, k);
    gammaTilde[1] = s.conformalConnection.y(i, j, k);
    gammaTilde[2] = s.conformalConnection.z(i, j, k);

    std::array<std::array<double, 3>, 3> dGammaTilde{};  // d_axis GammaTilde^k
    for (int upper = 0; upper < 3; ++upper) {
        const std::array<double, 3> grad =
            gradientAt(vectorComponentField(s.conformalConnection, upper), i, j, k, spacing);
        for (int axis = 0; axis < 3; ++axis) {
            dGammaTilde[static_cast<std::size_t>(upper)][static_cast<std::size_t>(axis)] =
                grad[static_cast<std::size_t>(axis)];
        }
    }

    Sym3 result{};
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            // Term 1: -(1/2) gammaTilde^lm d_l d_m gammaTilde_ab
            double term1 = 0.0;
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    const double d2 =
                        (l == m) ? secondDerivative(componentField(s.conformalMetric, a, b),
                                                   i, j, k, static_cast<Axis>(l), spacing)
                                : mixedSecondDerivative(componentField(s.conformalMetric, a, b),
                                                        i, j, k, static_cast<Axis>(l),
                                                        static_cast<Axis>(m), spacing, spacing);
                    term1 += ginv(l, m) * d2;
                }
            }
            term1 *= -0.5;

            // Term 2: gammaTilde_k(a d_b) GammaTilde^k
            double term2 = 0.0;
            for (int kk = 0; kk < 3; ++kk) {
                term2 += g(kk, a) * dGammaTilde[static_cast<std::size_t>(kk)]
                                              [static_cast<std::size_t>(b)];
                term2 += g(kk, b) * dGammaTilde[static_cast<std::size_t>(kk)]
                                              [static_cast<std::size_t>(a)];
            }
            term2 *= 0.5;

            // Term 3: GammaTilde^k GammaTilde_(ab)k
            double term3 = 0.0;
            for (int kk = 0; kk < 3; ++kk) {
                const double loweredAB = loweredChristoffel(g, christoffel, a, b, kk);
                const double loweredBA = loweredChristoffel(g, christoffel, b, a, kk);
                term3 += gammaTilde[static_cast<std::size_t>(kk)] * 0.5 *
                        (loweredAB + loweredBA);
            }

            // Term 4: gammaTilde^lm [ 2 GammaTilde^k_l(a GammaTilde_b)km
            //                        + GammaTilde^k_am GammaTilde_klb ]
            double term4 = 0.0;
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    for (int kk = 0; kk < 3; ++kk) {
                        const double gammaTildeKLa = christoffel(kk, l, a);
                        const double gammaTildeKLb = christoffel(kk, l, b);
                        const double loweredBKM = loweredChristoffel(g, christoffel, b, kk, m);
                        const double loweredAKM = loweredChristoffel(g, christoffel, a, kk, m);
                        const double symmetrized =
                            gammaTildeKLa * loweredBKM + gammaTildeKLb * loweredAKM;

                        const double gammaTildeKAm = christoffel(kk, a, m);
                        const double loweredKLB = loweredChristoffel(g, christoffel, kk, l, b);

                        term4 += ginv(l, m) * (symmetrized + gammaTildeKAm * loweredKLB);
                    }
                }
            }

            const double value = term1 + term2 + term3 + term4;
            result(a, b) = value;
            result(b, a) = value;
        }
    }
    return result;
}

/// The conformal-factor contribution to the physical Ricci tensor, from the
/// standard 3D conformal transformation gamma_ij = e^{4 phi} gammaTilde_ij:
///
///   R^phi_ij = -2 DtildeI Dtilde J phi - 2 gammaTilde_ij DtildeL DtildeL phi
///            + 4 DtildeI phi DtildeJ phi - 4 gammaTilde_ij DtildeL phi DtildeL phi
[[nodiscard]] Sym3 phiRicciAt(const BssnState& s, std::ptrdiff_t i, std::ptrdiff_t j,
                             std::ptrdiff_t k, double spacing) {
    const Sym3 g = conformalMetricAt(s, i, j, k);
    const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
    const Chr3 conformalChristoffel = conformalChristoffelAt(s, i, j, k, spacing);
    const Sym3 hessianPhi = covariantHessianAt(s.phi, conformalChristoffel, i, j, k, spacing);
    const std::array<double, 3> dphi = gradientAt(s.phi, i, j, k, spacing);

    double traceHessian = 0.0;
    double gradientSquared = 0.0;
    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            traceHessian += ginv(l, m) * hessianPhi(l, m);
            gradientSquared +=
                ginv(l, m) * dphi[static_cast<std::size_t>(l)] * dphi[static_cast<std::size_t>(m)];
        }
    }

    Sym3 result{};
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            const double value = -2.0 * hessianPhi(a, b) - 2.0 * g(a, b) * traceHessian +
                                 4.0 * dphi[static_cast<std::size_t>(a)] *
                                     dphi[static_cast<std::size_t>(b)] -
                                 4.0 * g(a, b) * gradientSquared;
            result(a, b) = value;
            result(b, a) = value;
        }
    }
    return result;
}

/// [X_ij]^TF = X_ij - (1/3) gammaTilde_ij (gammaTilde^lm X_lm): trace-free
/// with respect to the physical metric and with respect to the conformal
/// one are the same projection (the common conformal factor cancels in this
/// exact contraction), so the conformal metric is what is actually used.
[[nodiscard]] Sym3 traceFreeAt(const Sym3& x, const Sym3& g, const Sym3& ginv) {
    double trace = 0.0;
    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            trace += ginv(l, m) * x(l, m);
        }
    }
    Sym3 result{};
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            const double value = x(a, b) - (trace / 3.0) * g(a, b);
            result(a, b) = value;
            result(b, a) = value;
        }
    }
    return result;
}

/// Sum_axis beta^axis d_axis(field): the advection term every BSSN
/// evolution equation carries, beta^i d_i(quantity), for a scalar field.
[[nodiscard]] double advectionAt(const Grid3D<double>& field, const BssnState& s,
                                std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t k,
                                double spacing) {
    return s.shift.x(i, j, k) * firstDerivative(field, i, j, k, Axis::X, spacing) +
           s.shift.y(i, j, k) * firstDerivative(field, i, j, k, Axis::Y, spacing) +
           s.shift.z(i, j, k) * firstDerivative(field, i, j, k, Axis::Z, spacing);
}

/// d_axis(beta^component), the shift's own Jacobian at a point: every
/// evolution equation for a tensor needs how the shift itself varies
/// (the terms that make the equation a genuine Lie derivative, not a plain
/// advection).
[[nodiscard]] std::array<std::array<double, 3>, 3> shiftJacobianAt(const BssnState& s,
                                                                   std::ptrdiff_t i,
                                                                   std::ptrdiff_t j,
                                                                   std::ptrdiff_t k,
                                                                   double spacing) {
    std::array<std::array<double, 3>, 3> result{};
    for (int component = 0; component < 3; ++component) {
        const std::array<double, 3> grad =
            gradientAt(vectorComponentField(s.shift, component), i, j, k, spacing);
        for (int axis = 0; axis < 3; ++axis) {
            result[static_cast<std::size_t>(component)][static_cast<std::size_t>(axis)] =
                grad[static_cast<std::size_t>(axis)];
        }
    }
    return result;
}

/// The genuine partial derivative of the raised tensor field
/// AtildeIJ = gammaTilde^al gammaTilde^bm Atilde_lm, along `derivativeAxis`,
/// at (i, j, k): the inverse conformal metric is re-evaluated at each of
/// the same five stencil points `Math/FiniteDifference.hpp`'s fourth-order
/// stencil reads, not pulled out of the derivative at the center point
/// alone. Holding the inverse metric fixed while differentiating only the
/// stored (lowered) components would drop d(gammaTilde^-1)/d(axis)
/// entirely -- an O(1) term, not a higher-order truncation error, whenever
/// the conformal metric is not locally uniform (i.e. whenever there is
/// curvature to actually check).
[[nodiscard]] double raisedTracelessExtrinsicCurvatureDerivative(const BssnState& s,
                                                                 std::ptrdiff_t i,
                                                                 std::ptrdiff_t j,
                                                                 std::ptrdiff_t k, int a,
                                                                 int b, Axis derivativeAxis,
                                                                 double spacing) {
    double sum = 0.0;
    for (std::ptrdiff_t m = -2; m <= 2; ++m) {
        std::ptrdiff_t di = 0;
        std::ptrdiff_t dj = 0;
        std::ptrdiff_t dk = 0;
        detail::axisOffset(derivativeAxis, m, di, dj, dk);

        const Sym3 ginvHere = conformalMetricInverseAt(s, i + di, j + dj, k + dk);
        const Sym3 aTildeHere =
            s.conformalTracelessExtrinsicCurvature.at(i + di, j + dj, k + dk);
        double raised = 0.0;
        for (int l = 0; l < 3; ++l) {
            for (int mm = 0; mm < 3; ++mm) {
                raised += ginvHere(a, l) * ginvHere(b, mm) * aTildeHere(l, mm);
            }
        }

        const double coefficient =
            detail::kFirstDerivativeCoefficients[static_cast<std::size_t>(m + 2)];
        sum += coefficient * raised;
    }
    return sum / spacing;
}

}  // namespace

BssnState admToBssn(const AdmData& adm) {
    const std::size_t nx = adm.spatialMetric.xx.cellCountX();
    const std::size_t ny = adm.spatialMetric.xx.cellCountY();
    const std::size_t nz = adm.spatialMetric.xx.cellCountZ();
    const double spacing = adm.spatialMetric.xx.spacing();
    const std::size_t ghostCells = adm.spatialMetric.xx.ghostCells();

    BssnState state(nx, ny, nz, spacing, ghostCells);

    const auto ix = static_cast<std::ptrdiff_t>(nx);
    const auto iy = static_cast<std::ptrdiff_t>(ny);
    const auto iz = static_cast<std::ptrdiff_t>(nz);
    const auto ig = static_cast<std::ptrdiff_t>(ghostCells);

    for (std::ptrdiff_t i = -ig; i < ix + ig; ++i) {
        for (std::ptrdiff_t j = -ig; j < iy + ig; ++j) {
            for (std::ptrdiff_t k = -ig; k < iz + ig; ++k) {
                const Sym3 gamma = adm.spatialMetric.at(i, j, k);
                const Sym3 kExtrinsic = adm.extrinsicCurvature.at(i, j, k);
                const double det = determinant(toMatrix3(gamma));

                // phi = (1/12) ln(det gamma), so gammaTilde = e^{-4 phi} gamma has
                // unit determinant by construction.
                const double phiValue = std::log(det) / 12.0;
                const double conformalFactor = std::exp(-4.0 * phiValue);

                Sym3 gammaTilde{};
                double traceK = 0.0;
                {
                    const Sym3 gammaInv = toTensor(inverse(toMatrix3(gamma)));
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            traceK += gammaInv(a, b) * kExtrinsic(a, b);
                        }
                    }
                }

                Sym3 aTilde{};
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        gammaTilde(a, b) = gammaTilde(b, a) = conformalFactor * gamma(a, b);
                        const double physicalTraceFree =
                            kExtrinsic(a, b) - (traceK / 3.0) * gamma(a, b);
                        aTilde(a, b) = aTilde(b, a) = conformalFactor * physicalTraceFree;
                    }
                }

                state.phi(i, j, k) = phiValue;
                state.conformalMetric.set(i, j, k, gammaTilde);
                state.traceExtrinsicCurvature(i, j, k) = traceK;
                state.conformalTracelessExtrinsicCurvature.set(i, j, k, aTilde);
                state.lapse(i, j, k) = adm.lapse(i, j, k);
                state.shift.x(i, j, k) = adm.shift.x(i, j, k);
                state.shift.y(i, j, k) = adm.shift.y(i, j, k);
                state.shift.z(i, j, k) = adm.shift.z(i, j, k);
                state.shiftAuxiliary.x(i, j, k) = 0.0;
                state.shiftAuxiliary.y(i, j, k) = 0.0;
                state.shiftAuxiliary.z(i, j, k) = 0.0;
            }
        }
    }

    // GammaTilde^i = gammaTilde^jk GammaTilde^i_jk (the trace of the
    // conformal Christoffel symbols, identically equal to -d_j gammaTilde^ij
    // for a unit-determinant conformal metric): needs gammaTilde's own
    // derivatives, so it is only well-defined once every cell above has a
    // gammaTilde. A second pass over the interior (a one-sided derivative
    // near the ghost boundary would be inconsistent with the rest of this
    // module's centered stencils) computes it from the fields just written.
    const auto ixInterior = static_cast<std::ptrdiff_t>(nx);
    const auto iyInterior = static_cast<std::ptrdiff_t>(ny);
    const auto izInterior = static_cast<std::ptrdiff_t>(nz);
    for (std::ptrdiff_t i = 0; i < ixInterior; ++i) {
        for (std::ptrdiff_t j = 0; j < iyInterior; ++j) {
            for (std::ptrdiff_t k = 0; k < izInterior; ++k) {
                const Sym3 ginv = conformalMetricInverseAt(state, i, j, k);
                const Chr3 christoffel = conformalChristoffelAt(state, i, j, k, spacing);
                std::array<double, 3> gammaTildeFromChristoffel{};
                for (int upper = 0; upper < 3; ++upper) {
                    double sum = 0.0;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            sum += ginv(a, b) * christoffel(upper, a, b);
                        }
                    }
                    gammaTildeFromChristoffel[static_cast<std::size_t>(upper)] = sum;
                }
                state.conformalConnection.x(i, j, k) = gammaTildeFromChristoffel[0];
                state.conformalConnection.y(i, j, k) = gammaTildeFromChristoffel[1];
                state.conformalConnection.z(i, j, k) = gammaTildeFromChristoffel[2];
            }
        }
    }

    return state;
}

AdmData bssnToAdm(const BssnState& state) {
    const std::size_t nx = state.phi.cellCountX();
    const std::size_t ny = state.phi.cellCountY();
    const std::size_t nz = state.phi.cellCountZ();
    const double spacing = state.phi.spacing();
    const std::size_t ghostCells = state.phi.ghostCells();

    AdmData adm(nx, ny, nz, spacing, ghostCells);

    const auto ix = static_cast<std::ptrdiff_t>(nx);
    const auto iy = static_cast<std::ptrdiff_t>(ny);
    const auto iz = static_cast<std::ptrdiff_t>(nz);
    const auto ig = static_cast<std::ptrdiff_t>(ghostCells);

    for (std::ptrdiff_t i = -ig; i < ix + ig; ++i) {
        for (std::ptrdiff_t j = -ig; j < iy + ig; ++j) {
            for (std::ptrdiff_t k = -ig; k < iz + ig; ++k) {
                const double conformalFactor = std::exp(4.0 * state.phi(i, j, k));
                const Sym3 gammaTilde = state.conformalMetric.at(i, j, k);
                const Sym3 aTilde = state.conformalTracelessExtrinsicCurvature.at(i, j, k);
                const double traceK = state.traceExtrinsicCurvature(i, j, k);

                Sym3 gamma{};
                Sym3 kExtrinsic{};
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        gamma(a, b) = gamma(b, a) = conformalFactor * gammaTilde(a, b);
                        const double physicalTraceFree = aTilde(a, b) / conformalFactor;
                        const double value =
                            physicalTraceFree + (traceK / 3.0) * gamma(a, b);
                        kExtrinsic(a, b) = kExtrinsic(b, a) = value;
                    }
                }

                adm.spatialMetric.set(i, j, k, gamma);
                adm.extrinsicCurvature.set(i, j, k, kExtrinsic);
                adm.lapse(i, j, k) = state.lapse(i, j, k);
                adm.shift.x(i, j, k) = state.shift.x(i, j, k);
                adm.shift.y(i, j, k) = state.shift.y(i, j, k);
                adm.shift.z(i, j, k) = state.shift.z(i, j, k);
            }
        }
    }
    return adm;
}

BssnState bssnRhs(const BssnState& s, BssnParameters params) {
    const std::size_t nx = s.phi.cellCountX();
    const std::size_t ny = s.phi.cellCountY();
    const std::size_t nz = s.phi.cellCountZ();
    const double spacing = s.phi.spacing();
    const std::size_t ghostCells = s.phi.ghostCells();

    BssnState rhs(nx, ny, nz, spacing, ghostCells);

    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nx); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(ny); ++j) {
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(nz); ++k) {
                const Sym3 g = conformalMetricAt(s, i, j, k);
                const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
                const Sym3 aTilde = s.conformalTracelessExtrinsicCurvature.at(i, j, k);
                const double alpha = s.lapse(i, j, k);
                const double traceK = s.traceExtrinsicCurvature(i, j, k);
                const double bx = s.shift.x(i, j, k);
                const double by = s.shift.y(i, j, k);
                const double bz = s.shift.z(i, j, k);
                const std::array<std::array<double, 3>, 3> dBeta =
                    shiftJacobianAt(s, i, j, k, spacing);
                double divBeta = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    divBeta += dBeta[static_cast<std::size_t>(axis)][static_cast<std::size_t>(axis)];
                }

                // Raised (both indices) AtildeIJ, used repeatedly below.
                Sym3 aTildeUpper{};
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        double sum = 0.0;
                        for (int l = 0; l < 3; ++l) {
                            for (int m = 0; m < 3; ++m) {
                                sum += ginv(a, l) * ginv(b, m) * aTilde(l, m);
                            }
                        }
                        aTildeUpper(a, b) = aTildeUpper(b, a) = sum;
                    }
                }
                double aTildeSquared = 0.0;  // AtildeIJ Atilde^ij
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        aTildeSquared += aTilde(a, b) * aTildeUpper(a, b);
                    }
                }

                // --- phi ---------------------------------------------------
                rhs.phi(i, j, k) = advectionAt(s.phi, s, i, j, k, spacing) -
                                  (alpha * traceK) / 6.0 +
                                  kreissOligerDissipation3D(s.phi, i, j, k, spacing,
                                                            params.kreissOligerSigma);

                // --- gammaTilde_ij ------------------------------------------
                {
                    Sym3 rhsGamma{};
                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            double lieShift = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                const double gac =
                                    (c == 0)   ? componentField(s.conformalMetric, a, 0)(i, j, k)
                                    : (c == 1) ? componentField(s.conformalMetric, a, 1)(i, j, k)
                                              : componentField(s.conformalMetric, a, 2)(i, j, k);
                                const double gbc =
                                    (c == 0)   ? componentField(s.conformalMetric, b, 0)(i, j, k)
                                    : (c == 1) ? componentField(s.conformalMetric, b, 1)(i, j, k)
                                              : componentField(s.conformalMetric, b, 2)(i, j, k);
                                lieShift += gac * dBeta[static_cast<std::size_t>(c)]
                                                       [static_cast<std::size_t>(b)] +
                                           gbc * dBeta[static_cast<std::size_t>(c)]
                                                      [static_cast<std::size_t>(a)];
                            }
                            const double value =
                                advectionAt(componentField(s.conformalMetric, a, b), s, i, j, k,
                                          spacing) -
                                2.0 * alpha * aTilde(a, b) + lieShift -
                                (2.0 / 3.0) * g(a, b) * divBeta +
                                kreissOligerDissipation3D(componentField(s.conformalMetric, a, b),
                                                         i, j, k, spacing,
                                                         params.kreissOligerSigma);
                            rhsGamma(a, b) = rhsGamma(b, a) = value;
                        }
                    }
                    rhs.conformalMetric.set(i, j, k, rhsGamma);
                }

                // --- Physical Ricci, needed for K and AtildeIJ --------------
                const Sym3 conformalRicci = conformalRicciAt(s, i, j, k, spacing);
                const Sym3 phiRicci = phiRicciAt(s, i, j, k, spacing);
                Sym3 ricci{};
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const double value = conformalRicci(a, b) + phiRicci(a, b);
                        ricci(a, b) = ricci(b, a) = value;
                    }
                }

                const Chr3 physicalChristoffel = physicalChristoffelAt(s, i, j, k, spacing);
                const Sym3 hessianAlpha =
                    covariantHessianAt(s.lapse, physicalChristoffel, i, j, k, spacing);

                // --- K -------------------------------------------------------
                {
                    const double conformalFactor = std::exp(-4.0 * s.phi(i, j, k));
                    double laplacianAlpha = 0.0;  // gamma^ij D_i D_j alpha
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            laplacianAlpha += conformalFactor * ginv(a, b) * hessianAlpha(a, b);
                        }
                    }
                    rhs.traceExtrinsicCurvature(i, j, k) =
                        advectionAt(s.traceExtrinsicCurvature, s, i, j, k, spacing) -
                        laplacianAlpha + alpha * (aTildeSquared + (traceK * traceK) / 3.0) +
                        kreissOligerDissipation3D(s.traceExtrinsicCurvature, i, j, k, spacing,
                                                 params.kreissOligerSigma);
                }

                // --- AtildeIJ -------------------------------------------------
                {
                    Sym3 sourceTerm{};  // [-D_iD_j alpha + alpha R_ij]^TF
                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            const double value =
                                -hessianAlpha(a, b) + alpha * ricci(a, b);
                            sourceTerm(a, b) = sourceTerm(b, a) = value;
                        }
                    }
                    const Sym3 tf = traceFreeAt(sourceTerm, g, ginv);
                    const double conformalFactor = std::exp(-4.0 * s.phi(i, j, k));

                    // AtildeIK Atilde^k_j = AtildeIK gammaTilde^kl Atilde_lj
                    Sym3 aSquared{};
                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            double sum = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                for (int l = 0; l < 3; ++l) {
                                    sum += aTilde(a, c) * ginv(c, l) * aTilde(l, b);
                                }
                            }
                            aSquared(a, b) = aSquared(b, a) = sum;
                        }
                    }

                    Sym3 rhsATilde{};
                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            double lieShift = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                lieShift += aTilde(a, c) * dBeta[static_cast<std::size_t>(c)]
                                                                [static_cast<std::size_t>(b)] +
                                           aTilde(b, c) * dBeta[static_cast<std::size_t>(c)]
                                                               [static_cast<std::size_t>(a)];
                            }
                            const double value =
                                conformalFactor * tf(a, b) +
                                alpha * (traceK * aTilde(a, b) - 2.0 * aSquared(a, b)) +
                                advectionAt(componentField(s.conformalTracelessExtrinsicCurvature,
                                                          a, b),
                                          s, i, j, k, spacing) +
                                lieShift - (2.0 / 3.0) * aTilde(a, b) * divBeta +
                                kreissOligerDissipation3D(
                                    componentField(s.conformalTracelessExtrinsicCurvature, a, b),
                                    i, j, k, spacing, params.kreissOligerSigma);
                            rhsATilde(a, b) = rhsATilde(b, a) = value;
                        }
                    }
                    rhs.conformalTracelessExtrinsicCurvature.set(i, j, k, rhsATilde);
                }

                // --- GammaTilde^i ---------------------------------------------
                // dt GammaTilde^i = gammaTilde^jk d_j d_k beta^i
                //                 + (1/3) gammaTilde^ij d_j d_k beta^k
                //                 + beta^j d_j GammaTilde^i - GammaTilde^j d_j beta^i
                //                 + (2/3) GammaTilde^i d_j beta^j
                //                 - 2 Atilde^ij d_j alpha
                //                 + 2 alpha (GammaTilde^i_jk Atilde^jk
                //                            - (2/3) gammaTilde^ij d_j K
                //                            - 6 Atilde^ij d_j phi)
                std::array<double, 3> rhsGammaTilde{};
                {
                    const std::array<double, 3> dAlpha = gradientAt(s.lapse, i, j, k, spacing);
                    const std::array<double, 3> dK =
                        gradientAt(s.traceExtrinsicCurvature, i, j, k, spacing);
                    const std::array<double, 3> dPhi = gradientAt(s.phi, i, j, k, spacing);
                    const Chr3 conformalChristoffel = conformalChristoffelAt(s, i, j, k, spacing);

                    std::array<double, 3> gammaTilde{};
                    gammaTilde[0] = s.conformalConnection.x(i, j, k);
                    gammaTilde[1] = s.conformalConnection.y(i, j, k);
                    gammaTilde[2] = s.conformalConnection.z(i, j, k);

                    // d_axis GammaTilde^upper, computed once per upper index
                    // rather than inside the advection loop below.
                    std::array<std::array<double, 3>, 3> dGammaTilde{};
                    for (int upper = 0; upper < 3; ++upper) {
                        dGammaTilde[static_cast<std::size_t>(upper)] = gradientAt(
                            vectorComponentField(s.conformalConnection, upper), i, j, k, spacing);
                    }

                    for (int upper = 0; upper < 3; ++upper) {
                        // gammaTilde^ab d_a d_b beta^upper
                        double laplacianBeta = 0.0;
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                const double d2 =
                                    (a == b)
                                        ? secondDerivative(vectorComponentField(s.shift, upper),
                                                         i, j, k, static_cast<Axis>(a), spacing)
                                        : mixedSecondDerivative(
                                              vectorComponentField(s.shift, upper), i, j, k,
                                              static_cast<Axis>(a), static_cast<Axis>(b), spacing,
                                              spacing);
                                laplacianBeta += ginv(a, b) * d2;
                            }
                        }

                        // (1/3) gammaTilde^{upper,a} d_a d_c beta^c: for each
                        // a, sum over c of d_a d_c(beta^c), then contract a
                        // with gammaTilde^{upper,a}.
                        double divBetaGradient = 0.0;
                        for (int a = 0; a < 3; ++a) {
                            double sumOverC = 0.0;
                            for (int c = 0; c < 3; ++c) {
                                sumOverC +=
                                    (a == c)
                                        ? secondDerivative(vectorComponentField(s.shift, c), i, j,
                                                         k, static_cast<Axis>(a), spacing)
                                        : mixedSecondDerivative(
                                              vectorComponentField(s.shift, c), i, j, k,
                                              static_cast<Axis>(a), static_cast<Axis>(c), spacing,
                                              spacing);
                            }
                            divBetaGradient += ginv(upper, a) * sumOverC;
                        }
                        divBetaGradient /= 3.0;

                        // beta^a d_a GammaTilde^upper
                        double advectionGamma = 0.0;
                        // GammaTilde^a d_a beta^upper (note: d_a beta^upper is
                        // dBeta[upper][a], not dBeta[a][upper] --
                        // shiftJacobianAt(...)[component][axis] = d_axis(beta^component)).
                        double shiftTerm = 0.0;
                        for (int axis = 0; axis < 3; ++axis) {
                            const double betaAxis = (axis == 0) ? bx : (axis == 1) ? by : bz;
                            advectionGamma +=
                                betaAxis * dGammaTilde[static_cast<std::size_t>(upper)]
                                                     [static_cast<std::size_t>(axis)];
                            shiftTerm += gammaTilde[static_cast<std::size_t>(axis)] *
                                        dBeta[static_cast<std::size_t>(upper)]
                                            [static_cast<std::size_t>(axis)];
                        }

                        double aTildeDotDAlpha = 0.0;
                        double christoffelATilde = 0.0;
                        double aTildeDGammaTerm = 0.0;
                        double aTildeDPhiTerm = 0.0;
                        for (int a = 0; a < 3; ++a) {
                            aTildeDotDAlpha +=
                                aTildeUpper(upper, a) * dAlpha[static_cast<std::size_t>(a)];
                            aTildeDGammaTerm += ginv(upper, a) * dK[static_cast<std::size_t>(a)];
                            aTildeDPhiTerm +=
                                aTildeUpper(upper, a) * dPhi[static_cast<std::size_t>(a)];
                            for (int b = 0; b < 3; ++b) {
                                christoffelATilde +=
                                    conformalChristoffel(upper, a, b) * aTildeUpper(a, b);
                            }
                        }

                        rhsGammaTilde[static_cast<std::size_t>(upper)] =
                            laplacianBeta + divBetaGradient + advectionGamma - shiftTerm +
                            (2.0 / 3.0) * gammaTilde[static_cast<std::size_t>(upper)] * divBeta -
                            2.0 * aTildeDotDAlpha +
                            2.0 * alpha *
                                (christoffelATilde - (2.0 / 3.0) * aTildeDGammaTerm -
                                6.0 * aTildeDPhiTerm) +
                            kreissOligerDissipation3D(
                                vectorComponentField(s.conformalConnection, upper), i, j, k,
                                spacing, params.kreissOligerSigma);
                    }
                }
                rhs.conformalConnection.x(i, j, k) = rhsGammaTilde[0];
                rhs.conformalConnection.y(i, j, k) = rhsGammaTilde[1];
                rhs.conformalConnection.z(i, j, k) = rhsGammaTilde[2];

                // --- Gauge: 1+log slicing, (non-advective) Gamma-driver -----
                rhs.lapse(i, j, k) = advectionAt(s.lapse, s, i, j, k, spacing) -
                                    2.0 * alpha * traceK +
                                    kreissOligerDissipation3D(s.lapse, i, j, k, spacing,
                                                             params.kreissOligerSigma);

                rhs.shift.x(i, j, k) = 0.75 * s.shiftAuxiliary.x(i, j, k);
                rhs.shift.y(i, j, k) = 0.75 * s.shiftAuxiliary.y(i, j, k);
                rhs.shift.z(i, j, k) = 0.75 * s.shiftAuxiliary.z(i, j, k);

                rhs.shiftAuxiliary.x(i, j, k) =
                    rhsGammaTilde[0] - params.gammaDriverEta * s.shiftAuxiliary.x(i, j, k);
                rhs.shiftAuxiliary.y(i, j, k) =
                    rhsGammaTilde[1] - params.gammaDriverEta * s.shiftAuxiliary.y(i, j, k);
                rhs.shiftAuxiliary.z(i, j, k) =
                    rhsGammaTilde[2] - params.gammaDriverEta * s.shiftAuxiliary.z(i, j, k);
            }
        }
    }

    return rhs;
}

double hamiltonianConstraint(const BssnState& s, std::ptrdiff_t i, std::ptrdiff_t j,
                            std::ptrdiff_t k) {
    const double spacing = s.phi.spacing();
    const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
    const Sym3 conformalRicci = conformalRicciAt(s, i, j, k, spacing);
    const Sym3 phiRicci = phiRicciAt(s, i, j, k, spacing);

    double physicalRicciTrace = 0.0;
    const double conformalFactor = std::exp(-4.0 * s.phi(i, j, k));
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            physicalRicciTrace +=
                conformalFactor * ginv(a, b) * (conformalRicci(a, b) + phiRicci(a, b));
        }
    }

    const Sym3 aTilde = s.conformalTracelessExtrinsicCurvature.at(i, j, k);
    double aTildeSquared = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    aTildeSquared += ginv(a, l) * ginv(b, m) * aTilde(a, b) * aTilde(l, m);
                }
            }
        }
    }

    const double traceK = s.traceExtrinsicCurvature(i, j, k);

    // H = R + (2/3) K^2 - AtildeIJ Atilde^ij (vacuum: no matter density term).
    return physicalRicciTrace + (2.0 / 3.0) * traceK * traceK - aTildeSquared;
}

double momentumConstraint(const BssnState& s, std::ptrdiff_t i, std::ptrdiff_t j,
                         std::ptrdiff_t k, int component) {
    const double spacing = s.phi.spacing();
    const Sym3 ginv = conformalMetricInverseAt(s, i, j, k);
    const Sym3 aTilde = s.conformalTracelessExtrinsicCurvature.at(i, j, k);
    const Chr3 christoffel = conformalChristoffelAt(s, i, j, k, spacing);
    const std::array<double, 3> dK = gradientAt(s.traceExtrinsicCurvature, i, j, k, spacing);
    const std::array<double, 3> dPhi = gradientAt(s.phi, i, j, k, spacing);

    Sym3 aTildeUpper{};
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            double sum = 0.0;
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    sum += ginv(a, l) * ginv(b, m) * aTilde(l, m);
                }
            }
            aTildeUpper(a, b) = aTildeUpper(b, a) = sum;
        }
    }

    // Ordinary partial divergence of Atilde^{component, b}, along b: the
    // raised field's own derivative (see raisedTracelessExtrinsicCurvatureDerivative),
    // not the inverse metric held fixed at this point while only the
    // lowered components are differentiated.
    double partialDivergence = 0.0;
    for (int b = 0; b < 3; ++b) {
        partialDivergence += raisedTracelessExtrinsicCurvatureDerivative(
            s, i, j, k, component, b, static_cast<Axis>(b), spacing);
    }

    double christoffelCorrection = 0.0;
    for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
            christoffelCorrection += christoffel(component, b, c) * aTildeUpper(b, c);
        }
    }

    double dKTerm = 0.0;
    double dPhiTerm = 0.0;
    for (int b = 0; b < 3; ++b) {
        dKTerm += ginv(component, b) * dK[static_cast<std::size_t>(b)];
        dPhiTerm += aTildeUpper(component, b) * dPhi[static_cast<std::size_t>(b)];
    }

    // M^i = D_j Atilde^ij - (2/3) gammaTilde^ij d_j K - 6 Atilde^ij d_j phi
    //     = (d_j Atilde^ij + Gamma^i_jk Atilde^jk) - (2/3) gammaTilde^ij d_j K
    //       - 6 Atilde^ij d_j phi   (vacuum: no matter momentum density).
    return partialDivergence + christoffelCorrection - (2.0 / 3.0) * dKTerm - 6.0 * dPhiTerm;
}

}  // namespace ysq
