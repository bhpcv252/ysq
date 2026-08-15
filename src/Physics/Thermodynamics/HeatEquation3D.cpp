#include <Physics/Thermodynamics/HeatEquation3D.hpp>

#include <Compute/ComputeBackend.hpp>

#include <cstddef>
#include <vector>

namespace ysq {

namespace {

// Measured on the development machine (Apple Silicon, Metal backend) by
// benchmarks/compute_thresholds.cpp: this kernel is O(n) per cell (six
// neighbor reads, no inner loop over every other cell), so the GPU path
// never won up to 128^3 (~2.1M cells), the largest grid tried -- this is a
// measured floor (double the largest size tried), not an observed
// crossover, and higher than the original provisional guess of 200000
// turned out to be. Also keeps every existing grid size in
// tests/unit/thermodynamics_heatequation3d.cpp (up to 48^3 = 110592 cells)
// on the exact double-precision CPU path, so a test like
// TotalHeatIsConservedExactly (which checks conservation to 1e-9, tighter
// than float32 can promise) keeps meaning what it already meant rather
// than silently gaining a GPU-dispatch precision caveat. Re-run the
// benchmark and update this if the reference machine or backend ever
// changes.
constexpr std::size_t kGpuDispatchThreshold = 4194304;

}  // namespace

HeatEquation3D::HeatEquation3D(std::size_t cellCountX, std::size_t cellCountY,
                               std::size_t cellCountZ, double spacing, double diffusivity)
    : m_diffusivity(diffusivity),
      m_temperature(cellCountX, cellCountY, cellCountZ, spacing, 1) {}

std::size_t HeatEquation3D::cellCountX() const noexcept {
    return m_temperature.cellCountX();
}

std::size_t HeatEquation3D::cellCountY() const noexcept {
    return m_temperature.cellCountY();
}

std::size_t HeatEquation3D::cellCountZ() const noexcept {
    return m_temperature.cellCountZ();
}

double HeatEquation3D::spacing() const noexcept {
    return m_temperature.spacing();
}

double HeatEquation3D::diffusivity() const noexcept {
    return m_diffusivity;
}

void HeatEquation3D::setTemperature(std::size_t i, std::size_t j, std::size_t k,
                                    double value) {
    m_temperature(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                  static_cast<std::ptrdiff_t>(k)) = value;
}

double HeatEquation3D::temperature(std::size_t i, std::size_t j, std::size_t k) const {
    return m_temperature(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                         static_cast<std::ptrdiff_t>(k));
}

double HeatEquation3D::stableTimeStep(double safetyFactor) const {
    const double h = m_temperature.spacing();
    return safetyFactor * (h * h) / (6.0 * m_diffusivity);
}

void HeatEquation3D::step(double dt) {
    const std::size_t cellCount = cellCountX() * cellCountY() * cellCountZ();
    const double h = m_temperature.spacing();
    const double factor = m_diffusivity * dt / (h * h);

    if (cellCount >= kGpuDispatchThreshold) {
        const std::size_t nx = cellCountX();
        const std::size_t ny = cellCountY();
        const std::size_t nz = cellCountZ();
        std::vector<float> temperature(cellCount);
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    temperature[(i * ny + j) * nz + k] = static_cast<float>(m_temperature(
                        static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j),
                        static_cast<std::ptrdiff_t>(k)));
                }
            }
        }

        std::vector<float> next(cellCount);
        defaultBackend().heatEquation3DStep(temperature, nx, ny, nz,
                                            static_cast<float>(factor), next);

        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    m_temperature(static_cast<std::ptrdiff_t>(i),
                                  static_cast<std::ptrdiff_t>(j),
                                  static_cast<std::ptrdiff_t>(k)) =
                        static_cast<double>(next[(i * ny + j) * nz + k]);
                }
            }
        }
        return;
    }

    m_temperature.applyPeriodicBoundary();

    const auto nx = static_cast<std::ptrdiff_t>(m_temperature.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_temperature.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_temperature.cellCountZ());

    Grid3D<double> next(static_cast<std::size_t>(nx), static_cast<std::size_t>(ny),
                        static_cast<std::size_t>(nz), h, 1);
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                const double laplacian =
                    m_temperature(i + 1, j, k) + m_temperature(i - 1, j, k) +
                    m_temperature(i, j + 1, k) + m_temperature(i, j - 1, k) +
                    m_temperature(i, j, k + 1) + m_temperature(i, j, k - 1) -
                    6.0 * m_temperature(i, j, k);
                next(i, j, k) = m_temperature(i, j, k) + factor * laplacian;
            }
        }
    }

    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                m_temperature(i, j, k) = next(i, j, k);
            }
        }
    }
}

double HeatEquation3D::totalHeat() const {
    const auto nx = static_cast<std::ptrdiff_t>(m_temperature.cellCountX());
    const auto ny = static_cast<std::ptrdiff_t>(m_temperature.cellCountY());
    const auto nz = static_cast<std::ptrdiff_t>(m_temperature.cellCountZ());
    const double h = m_temperature.spacing();

    double total = 0.0;
    for (std::ptrdiff_t i = 0; i < nx; ++i) {
        for (std::ptrdiff_t j = 0; j < ny; ++j) {
            for (std::ptrdiff_t k = 0; k < nz; ++k) {
                total += m_temperature(i, j, k);
            }
        }
    }
    return total * h * h * h;
}

}  // namespace ysq
