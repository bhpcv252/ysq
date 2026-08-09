# Physics/QuantumMechanics API reference

The time-independent Schrödinger equation as an eigenvalue problem, and
time-dependent evolution by split-step Fourier, in 1D and 3D. Start with
[docs/physics/quantummechanics.md](../../physics/quantummechanics.md) for
the two-flavor framing; [src/Physics/README.md](../../../src/Physics/README.md)
has both discretizations and the split-step scheme in full.

## `Physics/QuantumMechanics/Schrodinger.hpp`

The time-independent Schrödinger equation,
`-hbar^2/(2m) d^2(psi)/dx^2 + V(x) psi = E psi`, as a symmetric matrix
eigenvalue problem.

```cpp
struct QuantumEigenstates {
    std::vector<double> energies;                    // ascending
    std::vector<std::vector<double>> wavefunctions;   // wavefunctions[n][i], normalized
};

QuantumEigenstates solveTimeIndependentSchrodinger(std::span<const double> potential,
                                                   double spacing, double mass, double hbar);
```

| Member | Description |
| --- | --- |
| `wavefunctions[n][i]` | The n-th eigenstate's amplitude at grid point `i`, normalized so `sum_i wavefunctions[n][i]^2 * spacing == 1`. |
| `solveTimeIndependentSchrodinger` | Discretizes the Hamiltonian via the standard 3-point central-difference stencil (`H[i][i] = hbar^2/(m spacing^2) + V(x_i)`, `H[i][i+-1] = -hbar^2/(2 m spacing^2)`, Dirichlet boundary: `psi = 0` just outside the grid), then diagonalizes with `Math/Eigen.hpp`'s `jacobiEigenSymmetric`. |

```cpp
const ysq::QuantumEigenstates states =
    ysq::solveTimeIndependentSchrodinger(potential, spacing, mass, hbar);
const double groundStateEnergy = states.energies[0];
```

## `Physics/QuantumMechanics/Schrodinger3D.hpp`

The same eigenvalue problem in three spatial dimensions,
`-hbar^2/(2m) (d^2/dx^2 + d^2/dy^2 + d^2/dz^2) psi + V psi = E psi`, the
direct generalization of the 1D tridiagonal problem to the 7-point 3D
Laplacian stencil.

```cpp
struct QuantumEigenstates3D {
    std::vector<double> energies;
    std::vector<std::vector<double>> wavefunctions;  // wavefunctions[n][idx(i,j,k)]
};

QuantumEigenstates3D solveTimeIndependentSchrodinger3D(
    std::span<const double> potential, std::size_t nx, std::size_t ny, std::size_t nz,
    double spacing, double mass, double hbar);
```

| Member | Description |
| --- | --- |
| `wavefunctions[n][idx(i,j,k)]` | `idx(i,j,k) = (i*ny+j)*nz+k` (row-major, `x` slowest, `z` fastest, matching `Math/FFT.hpp`'s convention), normalized so `sum_idx wavefunctions[n][idx]^2 * spacing^3 == 1`. |
| `potential` | Flat, row-major, size `nx*ny*nz`. |
| `solveTimeIndependentSchrodinger3D` | The 7-point 3D Laplacian discretization, Dirichlet boundary (a neighbor past the domain edge is simply not a term), diagonalized with the exact same `jacobiEigenSymmetric` `Schrodinger.hpp` uses. |

**Scope: dense eigensolver, so only modest grids.** The Hamiltonian is
`N x N` for `N = nx*ny*nz`; `jacobiEigenSymmetric` is `O(N^3)`, so this is
correct and general but only practical for small grids (validated up to
`8x8x8 = 512`), not a production-scale simulation. No sparse/iterative
eigensolver (Lanczos, say) is implemented; that would be new, separate
scope.

```cpp
const ysq::QuantumEigenstates3D states =
    ysq::solveTimeIndependentSchrodinger3D(potential, nx, ny, nz, spacing, mass, hbar);
```

## `Physics/QuantumMechanics/WavePacket.hpp`

Time-dependent evolution under the Schrödinger equation by the
second-order Strang-split split-step Fourier method (Feit, Fleck & Steiger
1982): half a potential-phase step, a full kinetic-phase step in momentum
space via `Math/FFT.hpp`, then the other half potential-phase step. This is
the reason `Math/FFT.hpp` exists in this engine at all.

```cpp
class TimeDependentWavefunction1D {
public:
    TimeDependentWavefunction1D(std::vector<Complex<double>> initialWavefunction,
                                std::vector<double> potential, double spacing, double mass,
                                double hbar);

    std::size_t size() const noexcept;
    Complex<double> amplitude(std::size_t i) const;
    double probabilityDensity(std::size_t i) const;

    void step(double dt);

    double totalProbability() const;    // sum |psi_i|^2 spacing; stays 1, structurally
    double expectationPosition() const; // <x> = sum x_i |psi_i|^2 spacing
    double expectationEnergy() const;   // <H> = <T> + <V>, via the finite-difference Hamiltonian
};
```

| Member | Description |
| --- | --- |
| `step(dt)` | One split-step Fourier evolution step. `psi.size()` must be a power of two, `Math/FFT.hpp`'s own precondition. |
| `totalProbability` | Every operator this method applies is a pure phase, so unitarity is structural here, not merely approximate. |
| `expectationEnergy` | Computed by the same 3-point finite-difference Hamiltonian `Schrodinger.hpp`'s eigenvalue problem uses, **not** the FFT `step` uses internally: an independent check on the propagator, not a tautology. |

```cpp
ysq::TimeDependentWavefunction1D psi(initialWavefunction, potential, spacing, mass, hbar);
psi.step(dt);
const double probability = psi.totalProbability();
const double energy = psi.expectationEnergy();
```

## `Physics/QuantumMechanics/WavePacket3D.hpp`

The same split-step Fourier evolution in 3D, via `Math/FFT.hpp`'s
`fft3D`/`ifft3D`. The 3D kinetic operator is diagonal in momentum space
exactly like the 1D one, mode by mode, since `kx^2+ky^2+kz^2` separates
additively. Not built on `Grid3D`: `Complex` does not satisfy `Numeric`, so
storage is a flat `std::vector`, the same choice `WavePacket.hpp` makes.

```cpp
class TimeDependentWavefunction3D {
public:
    TimeDependentWavefunction3D(std::vector<Complex<double>> initialWavefunction,
                                std::vector<double> potential, std::size_t nx, std::size_t ny,
                                std::size_t nz, double spacing, double mass, double hbar);

    std::size_t size() const noexcept;
    Complex<double> amplitude(std::size_t i, std::size_t j, std::size_t k) const;
    double probabilityDensity(std::size_t i, std::size_t j, std::size_t k) const;

    void step(double dt);

    double totalProbability() const;    // sum |psi|^2 spacing^3
    double expectationEnergy() const;   // via the 7-point finite-difference Hamiltonian
};
```

| Member | Description |
| --- | --- |
| Constructor | `initialWavefunction`/`potential` flat, row-major, size `nx*ny*nz`, matching `Schrodinger3D.hpp`'s `idx` convention. `nx`, `ny`, `nz` must each independently be a power of two. |
| `expectationEnergy` | The same independent-check role `WavePacket.hpp`'s own version plays, via `Schrodinger3D.hpp`'s 7-point Hamiltonian rather than the FFT `step` uses. |

```cpp
ysq::TimeDependentWavefunction3D psi(initialWavefunction, potential, nx, ny, nz, spacing, mass, hbar);
psi.step(dt);
const double energy = psi.expectationEnergy();
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/physics/quantummechanics)
and let us know.
