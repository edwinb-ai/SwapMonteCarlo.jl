# Potentials

```@meta
DocTestSetup = :(using SwapMonteCarlo)
```

All interactions are pair potentials ``u(r; \sigma_i, \sigma_j)`` that depend on the
distance ``r`` and on the diameters of the two particles, usually through a pair
diameter ``\sigma_{ij}`` given by a [mixing rule](@ref "Mixing rules"). Every potential
vanishes exactly beyond a cutoff, so that energies are computed with cell lists.

Kernels receive the *squared* distance `r2` and avoid square roots where possible.

## Built-in potentials

### Polydisperse soft spheres (SS12)

[`PolySoftSphere`](@ref)`(; epsilon=1.0, xc=1.25, mixing=NonAdditive(0.2))` is the
potential of Ninarello, Berthier and Coslovich (PRX 7, 021039, 2017),

```math
u(r) = \varepsilon \left[ \left(\frac{\sigma_{ij}}{r}\right)^{12}
    + c_0 + c_2 \left(\frac{r}{\sigma_{ij}}\right)^2
    + c_4 \left(\frac{r}{\sigma_{ij}}\right)^4 \right]
\quad \text{for } r < x_c \sigma_{ij},
```

and zero beyond. With ``x = r/\sigma_{ij}``, the three conditions ``u(x_c) = u'(x_c) =
u''(x_c) = 0`` give

```math
c_0 = -\frac{28}{x_c^{12}}, \qquad c_2 = \frac{48}{x_c^{14}}, \qquad
c_4 = -\frac{21}{x_c^{16}},
```

so the energy, the force and its derivative are continuous at the cutoff. The constants
are precomputed, and the kernel uses neither `sqrt` nor `^`. With the defaults and the
diameters ``P(\sigma) \propto \sigma^{-3}`` on ``[0.73, 1.62]`` (see
[`InversePowerLaw`](@ref)) this is the model of the swap Monte Carlo literature.

```jldoctest
julia> u = SwapMonteCarlo.pair_energy(PolySoftSphere(), 1.0, 1.0, 1.0);  # contact, r = σ

julia> u ≈ 1 - 28 / 1.25^12 + 48 / 1.25^14 - 21 / 1.25^16
true

julia> SwapMonteCarlo.pair_energy(PolySoftSphere(), 1.25^2, 1.0, 1.0)  # at the cutoff
0.0
```

### Lennard-Jones

[`LennardJones`](@ref)`(; epsilon=1.0, xc=2.5, shift=EnergyShift(), mixing=Additive(),
tail_correction=false)` is

```math
u_\text{LJ}(r) = 4\varepsilon \left[ \left(\frac{\sigma_{ij}}{r}\right)^{12}
    - \left(\frac{\sigma_{ij}}{r}\right)^6 \right]
```

truncated at ``r_c = x_c \sigma_{ij}``, with one of three shift modes:

| Shift mode | ``u(r)`` for ``r < r_c`` | at ``r_c`` |
|---|---|---|
| [`NoShift`](@ref)`()` | ``u_\text{LJ}(r)`` | energy jumps |
| [`EnergyShift`](@ref)`()` | ``u_\text{LJ}(r) - u_\text{LJ}(r_c)`` | continuous energy |
| [`ForceShift`](@ref)`()` | ``u_\text{LJ}(r) - u_\text{LJ}(r_c) - (r - r_c)\, u_\text{LJ}'(r_c)`` | continuous energy and force |

Since the cutoff is relative to ``\sigma_{ij}``, the shift constants do not depend on
the diameters. The force-shifted potential needs a square root inside the cutoff.

With `tail_correction=true` (only for `Additive` mixing and `shift=NoShift()`) the
standard long range corrections of the full potential beyond the cutoff are added,
assuming ``g(r) = 1`` there:

```math
U_\text{lrc} = \frac{S_D}{2V} \sum_{i,j} \int_{r_c}^\infty u_\text{LJ}(r)\, r^{D-1}\,dr,
\qquad
P_\text{lrc} = \frac{S_D}{2DV^2} \sum_{i,j} \int_{r_c}^\infty w_\text{LJ}(r)\, r^{D-1}\,dr,
```

with ``S_2 = 2\pi``, ``S_3 = 4\pi`` and ``w = -r\,du/dr``. The integrals are
``\varepsilon\sigma_{ij}^D K`` with constants ``K`` that depend on ``x_c`` and ``D``, and
the double sum over all ordered pairs (including ``i = j``, the usual convention) is
computed in ``O(N)`` from the moments ``\sum_i \sigma_i^k`` by the binomial expansion of
``(\sigma_i + \sigma_j)^D``. See [`energy_lrc`](@ref SwapMonteCarlo.energy_lrc) and
[`pressure_lrc`](@ref SwapMonteCarlo.pressure_lrc). The pressure correction contains the
impulsive term of the energy jump at the cutoff, so it is only consistent with the
virial pressure of the plain truncation; a shifted potential with
`tail_correction=true` throws an `ArgumentError`. Truncated without shift and without
tail corrections, the virial pressure misses the impulsive term. The recommended choices
are `NoShift()` with tail corrections (the convention of the NIST reference data) or
`EnergyShift()`/`ForceShift()` without them; see [the pressure of truncated
potentials](@ref "Truncated potentials and the impulsive term").

### Hard spheres

[`HardSphere`](@ref)`(; mixing=Additive())` has ``u = \infty`` for ``r < \sigma_{ij}``
and ``u = 0`` otherwise (touching particles do not overlap). It is *hard*
([`is_hard`](@ref SwapMonteCarlo.is_hard)): in a valid configuration all energies are
zero, so moves only check for overlaps, which stop at the first one. The virial of hard
spheres is a contact term that the pair virial cannot express; measure the pressure with
[`VirtualPressure`](@ref).

### Pseudo hard spheres

[`PseudoHS`](@ref)`(; epsilon=1.0, mixing=Additive())` is the (50, 49) Mie potential of
Jover *et al.* (J. Chem. Phys. 137, 144505, 2012), cut and shifted at its minimum, as in
MolecularDynamics.jl:

```math
u(r) = A\varepsilon \left[ \left(\frac{\sigma_{ij}}{r}\right)^{50}
    - \left(\frac{\sigma_{ij}}{r}\right)^{49} \right] + \varepsilon
\quad \text{for } r < \tfrac{50}{49}\sigma_{ij},
\qquad A = 50 \left(\tfrac{50}{49}\right)^{49}.
```

At ``k_BT = 1.4737`` (with ``\varepsilon = 1``) its thermodynamics is close to that of
hard spheres. Unlike `HardSphere`, it has a virial pressure.

## Mixing rules

Each built-in potential takes a mixing rule (a type parameter, so there is no runtime
cost) that gives the pair diameter:

| Rule | ``\sigma_{ij}`` |
|---|---|
| [`Additive`](@ref)`()` | ``(\sigma_i + \sigma_j)/2`` |
| [`NonAdditive`](@ref)`(ε)` | ``\tfrac12 (\sigma_i + \sigma_j)(1 - \varepsilon\lvert\sigma_i - \sigma_j\rvert)``, ``0 \le \varepsilon < 1`` |

The non-additive rule of Ninarello *et al.* prevents fractionation and crystallization of
polydisperse mixtures. Restricting ``\varepsilon \ge 0`` guarantees ``\sigma_{ij} \le
\max(\sigma_i, \sigma_j)``, so the largest cutoff is always that of a pair of extreme
diameters. The pair diameters are positive only if
``\varepsilon(\sigma_\text{max} - \sigma_\text{min}) < 1``; a [`System`](@ref) whose
diameters give negative pair diameters throws an `ArgumentError`. Custom potentials can
use the rules through
[`mix`](@ref SwapMonteCarlo.mix):

```jldoctest
julia> SwapMonteCarlo.mix(NonAdditive(0.2), 1.0, 1.5)
1.125

julia> SwapMonteCarlo.mix(Additive(), 1.0, 1.5)
1.25
```

## Defining a potential

A potential is a subtype of [`SwapMonteCarlo.Potential`](@ref). The interface (all
functions are public but not exported, so `import` them to add methods):

| Method | Required | Purpose |
|---|---|---|
| `pair_energy(pot, r2, σi, σj)` | yes (or `evaluate`) | pair energy at squared distance `r2` |
| `cutoff(pot, σi, σj)` | yes | distance beyond which the pair energy is exactly zero |
| `pair_virial(pot, r2, σi, σj)` | for [`Pressure`](@ref) | ``w = -r\,du/dr`` |
| `is_repulsive(pot)` | no | `true` if all pair energies are ``\ge 0`` (early rejection) |
| `is_hard(pot)` | no | `true` if pair energies are either ``0`` or ``\infty`` and non-increasing in ``r`` |
| `energy_lrc(pot, sys)`, `pressure_lrc(pot, sys)` | no | long range corrections (default zero) |
| `max_cutoff(pot, diameters)` | no | largest pair cutoff (default: the extreme diameters) |

Requirements:

- `pair_energy` must return **exactly zero** at and beyond the cutoff, and it must be
  **symmetric** in `(σi, σj)`. Swap moves rely on the symmetry to leave out the energy
  of the swapped pair; [`System`](@ref) checks it on random pairs of the actual diameters
  and throws an `ArgumentError` if it fails.
- `is_repulsive(pot) = true` is a promise: the energy sums of a move stop as soon as they
  exceed the acceptance threshold, which is only exact if no later term can be negative.
- `is_hard(pot) = true` is a promise as well: pair energies are ``0`` or ``\infty`` and
  non-increasing in the distance, so that expanding the box never creates an overlap.
- The arguments are of the float type of the system (`Float64` by default, `Float32` if
  the box is `Float32`). Return `zero(r2)` rather than `0.0` to keep kernels type stable;
  results are converted to the float type of the system.

```@example custom
using SwapMonteCarlo
import SwapMonteCarlo: Potential, pair_energy, pair_virial, cutoff, is_repulsive
using SwapMonteCarlo: mix

# Harmonic spheres, u = ε/2 (1 - r/σij)² for r < σij.
struct Harmonic{M} <: Potential
    epsilon::Float64
    mixing::M
end
Harmonic(; epsilon=1.0, mixing=Additive()) = Harmonic(epsilon, mixing)

cutoff(p::Harmonic, σi, σj) = mix(p.mixing, σi, σj)

function pair_energy(p::Harmonic, r2, σi, σj)
    σij = mix(p.mixing, σi, σj)
    r2 < σij^2 || return zero(r2)
    return p.epsilon / 2 * (1 - sqrt(r2) / σij)^2
end

function pair_virial(p::Harmonic, r2, σi, σj)
    σij = mix(p.mixing, σi, σj)
    r2 < σij^2 || return zero(r2)
    r = sqrt(r2)
    return p.epsilon * (1 - r / σij) * r / σij
end

is_repulsive(::Harmonic) = true

box = box_for_density(Val(2), 256, 1.2)
sys = System(box, lattice_positions(box, 256), ones(256), Harmonic())
sim = Simulation(sys, NVT(1e-3), Displacement(0.05) => 1.0; seed=1)
run!(sim, 200)
(total_energy(sim.system) / 256, pressure(sim.system, 1e-3))
```

### MolecularDynamics.jl potentials

A potential written for MolecularDynamics.jl defines
[`evaluate`](@ref SwapMonteCarlo.evaluate)`(pot, r, σi, σj)` returning the energy and the
force ``f = -du/dr``. The default `pair_energy` and `pair_virial` call it (through
[`evaluate_r2`](@ref SwapMonteCarlo.evaluate_r2), which takes the square root), so only
`cutoff` has to be added:

```@example custom
import SwapMonteCarlo: evaluate

struct GaussianCore <: Potential end

function evaluate(::GaussianCore, r, σi, σj)
    σ = (σi + σj) / 2
    r < 3σ || return (zero(r), zero(r))
    u = exp(-(r / σ)^2)
    return (u, 2 * r / σ^2 * u)
end

cutoff(::GaussianCore, σi, σj) = 3 * (σi + σj) / 2

sys = System(box, lattice_positions(box, 256), ones(256), GaussianCore())
(total_energy(sys) / length(sys), pressure(sys, 1.0))
```

Defining `pair_energy` directly is faster, since it avoids the square root and the
force.

### Hard potentials

A hard potential returns `Inf` for overlapping pairs and zero otherwise, and declares
`is_hard(pot) = true` and `is_repulsive(pot) = true`. Moves then skip the energy of the
old configuration (it is zero), volume expansions skip the energy altogether, and
[`Pressure`](@ref) throws; use [`VirtualPressure`](@ref), which counts the pairs whose
energy becomes infinite after a small compression. Skipping expansions requires the pair
energy to be non-increasing in the distance: a pair that does not overlap must not
overlap at a larger distance. See [`HardSphere`](@ref) for a complete example.

### Long range corrections and custom cutoffs

[`energy_lrc`](@ref SwapMonteCarlo.energy_lrc) and
[`pressure_lrc`](@ref SwapMonteCarlo.pressure_lrc) receive the [`System`](@ref) and are
added to [`total_energy`](@ref) and [`pressure`](@ref). Since the energy correction
depends on the volume, volume moves include its change.

The cell list uses the largest pair cutoff,
[`max_cutoff`](@ref SwapMonteCarlo.max_cutoff)`(pot, diameters)`. The default checks the
pairs of the smallest and the largest diameters, which is exact when the cutoff grows
with both diameters; override it for other mixing rules.
