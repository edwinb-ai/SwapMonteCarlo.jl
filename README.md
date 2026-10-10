# SwapMonteCarlo.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

Metropolis Monte Carlo of (polydisperse) particles in two and three dimensions, in the
canonical ($`NVT`$) and isothermal-isobaric ($`NPT`$) ensembles. Besides single particle
displacements and volume moves it implements *swap* moves, which exchange the diameters of
two particles and equilibrate dense polydisperse liquids many orders of magnitude faster
than local moves (Ninarello, Berthier and Coslovich, PRX 7, 021039, 2017). Pair potentials
can be user-defined, and potentials written for
[MolecularDynamics.jl](https://github.com/edwinb-ai/MolecularDynamics.jl) work unchanged.

It is single threaded; displacement and swap moves cost $`O(1)`$ with cell lists and do
not allocate. It follows the conventions of MolecularDynamics.jl (reduced units, extended
XYZ files).

## Installation

The package needs Julia 1.13. It is not registered; add it from a local clone,

```julia
using Pkg: Pkg
Pkg.develop(; path="/path/to/SwapMonteCarlo.jl")
```

or from its repository (placeholder URL),

```julia
using Pkg: Pkg
Pkg.add(; url="https://github.com/edwinb-ai/SwapMonteCarlo.jl")
```

## Quick start: 2D swap Monte Carlo

The continuously polydisperse soft-sphere mixture of Ninarello *et al.*: diameters drawn
from $`P(\sigma) \propto \sigma^{-3}`$ on $`[0.73, 1.62]`$, a soft repulsion that vanishes
smoothly at $`1.25\,\sigma_{ij}`$ with a non-additive mixing rule, at number density 1.

```julia
using SwapMonteCarlo
using Random: Xoshiro

N = 1000
kT = 0.1
rng = Xoshiro(2026)

# Continuous polydisperse mixture of Ninarello, Berthier and Coslovich (PRX 2017):
# P(σ) ∝ σ⁻³ on [0.73, 1.62], soft spheres at number density ρ = 1.
σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N)
box = box_for_density(Val(2), N, 1.0)
x = lattice_positions(box, N)
sys = System(box, x, σ, PolySoftSphere())  # SS12, xc = 1.25, NonAdditive(0.2)

# 80 % displacements and 20 % swaps; one sweep is N attempts.
sim = Simulation(sys, NVT(kT), Displacement(0.1) => 0.8, Swap() => 0.2; seed=42)

# Equilibrate while tuning the maximum displacement toward 40 % acceptance.
equilibrate!(sim, 2_000)
reset_stats!(sim)

# Production run with a fixed step size, writing thermodynamics and configurations.
mkpath("output")
thermo = ThermoWriter(
    "output/thermo.txt",
    PotentialEnergy(),
    Pressure(),
    Acceptance(:swap);
    schedule=Every(100),
)
traj = TrajectoryWriter("output/trajectory.xyz"; schedule=Every(1_000))
run!(sim, 10_000; observers=(thermo, traj))
close(thermo)
close(traj)

acceptance_rate(sim)  # (displacement = 0.40…, swap = 0.13…)
```

This takes about ten seconds. The dimension is part of the types (`Val(2)` above), and
positions are plain vectors of `SVector`s. `examples/ninarello2d.jl` is a complete version
of this script that takes the temperature and the number of particles from the command
line.

## Hard spheres at constant pressure

Hard spheres need a non-overlapping start, `insert_hard_spheres` (random sequential
insertion, largest particles first), and the pressure is measured with virtual
compressions, since the virial of hard spheres is a contact term.

```julia
using SwapMonteCarlo
using Random: Xoshiro
using Statistics: mean

N = 500
rng = Xoshiro(7)
σ = ones(N)

# Random sequential insertion at packing fraction 0.35, then compress at βP = 5.
box = box_for_packing_fraction(Val(3), σ, 0.35)
x = insert_hard_spheres(rng, box, σ)
sys = System(box, x, σ, HardSphere())

# About five volume moves per sweep (1 % of the N = 500 attempts).
sim = Simulation(sys, NPT(1.0, 5.0), Displacement(0.1) => 1.0, VolumeMove(0.01) => 0.01)
equilibrate!(sim, 10_000)  # tunes displacement (target 0.4) and volume (target 0.3) steps
reset_stats!(sim)

ts = TimeSeries(Density(), VirtualPressure(); schedule=Every(10))
run!(sim, 10_000; observers=ts)
mean(ts["density"]), mean(ts["virtual_pressure"])  # ≈ (0.750, 5.0)
```

The Kolafa equation of state gives $`\rho = 0.7504`$ at $`\beta P = 5`$.
`examples/hs_npt3d.jl` compresses through several pressures with pressure ramps and
compares with it.

## Defining your own potential

A potential is a subtype of `SwapMonteCarlo.Potential` with two methods: `pair_energy(pot,
r2, σi, σj)`, the pair energy at *squared* distance `r2` between particles of diameters
`σi` and `σj`, and `cutoff(pot, σi, σj)`, the distance beyond which it is exactly zero. The
pair energy must be symmetric in `(σi, σj)`, which swap moves rely on (it is checked when
the `System` is built).

Optionally, `pair_virial(pot, r2, σi, σj)` $`= -r\,du/dr`$ enables the virial pressure, and
`is_repulsive(pot) = true` declares that all pair energies are non-negative, so energy sums
stop as soon as a move is certain to be rejected.

```julia
using SwapMonteCarlo
import SwapMonteCarlo: Potential, pair_energy, pair_virial, cutoff, is_repulsive
using SwapMonteCarlo: mix

# Harmonic spheres, u = ε/2 (1 - r/σij)² for r < σij, the classic jamming potential.
struct Harmonic <: Potential
    epsilon::Float64
end

cutoff(::Harmonic, σi, σj) = mix(Additive(), σi, σj)

function pair_energy(p::Harmonic, r2, σi, σj)
    σij = mix(Additive(), σi, σj)
    r2 < σij^2 || return zero(r2)
    return p.epsilon / 2 * (1 - sqrt(r2) / σij)^2
end

function pair_virial(p::Harmonic, r2, σi, σj)
    σij = mix(Additive(), σi, σj)
    r2 < σij^2 || return zero(r2)
    r = sqrt(r2)
    return p.epsilon * (1 - r / σij) * r / σij
end

is_repulsive(::Harmonic) = true

box = box_for_density(Val(2), 256, 1.0)
sys = System(box, lattice_positions(box, 256), ones(256), Harmonic(1.0))
sim = Simulation(sys, NVT(1e-3), Displacement(0.05) => 1.0; seed=1)
run!(sim, 100)
pressure(sim.system, 1e-3)
```

A potential written for MolecularDynamics.jl, which defines `evaluate(pot, r, σi, σj)`
returning the energy and the force $`f = -du/dr`$, works as is; only `cutoff` has to be
added:

```julia
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
total_energy(sys) / length(sys), pressure(sys, 1.0)
```

The full interface (long range corrections, hard potentials, custom `max_cutoff`) is in
the documentation and in the docstring of `SwapMonteCarlo.Potential`;
`examples/custom_potential.jl` defines the same potential both ways and checks them
against each other.

### Built-in potentials

| Potential | Pair energy for $`r < r_c`$ | $`r_c`$ |
|---|---|---|
| `PolySoftSphere(; epsilon, xc=1.25, mixing=NonAdditive(0.2))` | $`\varepsilon[(\sigma_{ij}/r)^{12} + c_0 + c_2 (r/\sigma_{ij})^2 + c_4 (r/\sigma_{ij})^4]`$, $`C^2`$ at $`r_c`$ | $`x_c \sigma_{ij}`$ |
| `LennardJones(; epsilon, xc=2.5, shift=EnergyShift(), mixing=Additive(), tail_correction=false)` | $`4\varepsilon[(\sigma_{ij}/r)^{12} - (\sigma_{ij}/r)^6]`$, `NoShift()`, `EnergyShift()` or `ForceShift()` | $`x_c \sigma_{ij}`$ |
| `HardSphere(; mixing=Additive())` | $`\infty`$ (overlap) | $`\sigma_{ij}`$ |
| `PseudoHS(; epsilon, mixing=Additive())` | (50, 49) Mie potential shifted at its minimum | $`(50/49)\sigma_{ij}`$ |

### Mixing rules

The pair diameter $`\sigma_{ij}`$ of every built-in potential comes from its mixing rule:
`Additive()`, $`\sigma_{ij} = (\sigma_i + \sigma_j)/2`$, or `NonAdditive(ε)`,
$`\sigma_{ij} = \tfrac{1}{2}(\sigma_i + \sigma_j)(1 - \varepsilon|\sigma_i - \sigma_j|)`$ with
$`0 \le \varepsilon < 1`$ (Ninarello *et al.*). Custom potentials can use them through
`SwapMonteCarlo.mix(rule, σi, σj)`.

```julia
PolySoftSphere(; mixing=Additive())
HardSphere(; mixing=NonAdditive(0.2))
SwapMonteCarlo.mix(NonAdditive(0.2), 1.0, 1.5)  # 1.125
```

## Moves and the acceptance criterion

A `Simulation` combines a system, an ensemble and moves with relative weights. One sweep
is $`N`$ attempts, each of a move chosen at random with probability proportional to its
weight.

| Move | Proposal | $`\ln A`$ |
|---|---|---|
| `Displacement(δ=0.1)` | random particle moved uniformly in $`[-\delta, \delta]^D`$ | $`-\beta\Delta U`$ |
| `Swap(; max_dsigma=Inf)` | diameters of a random pair $`i \ne j`$ exchanged | $`-\beta\Delta U`$ |
| `VolumeMove(δlnV=0.01)` | $`\ln V`$ changed uniformly in $`[-\delta, \delta]`$, positions scaled | $`-\beta(\Delta U + P\Delta V) + (N+1)\,\Delta\ln V`$ |

All moves use the Metropolis rule $`\min(1, A)`$, implemented with the threshold drawn
first: with $`\xi`$ exponentially distributed (`randexp`), a move is accepted if and only
if $`\ln A \ge -\xi`$, that is

```math
U_\text{new} \le U_\text{max} = U_\text{old} + k_BT\,(\text{bias} + \xi),
```

which happens with probability $`P(\xi \ge -\ln A) = \min(1, A)`$. The threshold is known
before the new energy is computed, so for repulsive potentials the energy sum stops as
soon as it exceeds $`U_\text{max}`$ (for hard spheres, at the first overlap). It never
evaluates `exp`, and $`\Delta U = 0`$ is always accepted, also at $`k_BT = 0`$.

The $`(N+1)\,\Delta\ln V`$ term of volume moves is the Jacobian of sampling $`\ln V`$
instead of $`V`$: the NPT density of $`(V, \mathbf{s}^N)`$ with fractional coordinates
$`\mathbf{s}`$ is $`\propto V^N e^{-\beta(U + PV)}`$ and $`dV = V\,d\ln V`$.

**Step sizes** of displacement and volume moves are tuned toward a target acceptance
ratio, 0.4 for displacements and 0.3 for volume moves, **only during `equilibrate!`**:
every `interval` attempts (1000 and 100) the step is multiplied by the ratio of the
measured and target acceptance, limited to $`[2/3, 3/2]`$ and to `bounds`. `run!` keeps the
step sizes fixed, so the production chain satisfies detailed balance. Both are set with
keywords, e.g. `Displacement(0.1; target=0.3, interval=500, bounds=(1e-4, 0.5))`.
`acceptance_rate(sim)` and `step_size(sim)` report them, and `reset_stats!(sim)` restarts
the acceptance counters.

## Pressure

- `Pressure()` is the virial pressure $`P = \rho k_BT + W/(DV)`$ (plus the tail
  correction), which needs `pair_virial`.
- `VirtualPressure(; xi=1e-3)` measures the pressure of hard potentials from the number
  of overlaps after a virtual compression of the volume by the fraction `xi` (Eppenga and
  Frenkel, 1984); its bias is of order `xi`.

In an NPT simulation, $`\langle P_\text{virial}\rangle`$ equals the imposed pressure
exactly when the energy is a continuous function of the volume: potentials continuous at
their cutoff (`PolySoftSphere`, `PseudoHS`, `LennardJones` with `EnergyShift()` or
`ForceShift()`) without tail corrections. A truncated potential with a jump at the cutoff
(`LennardJones(; shift=NoShift())`) has an impulsive contribution
$`\frac{S_D}{2D}\rho^2 r_c^D u(r_c)\, g(r_c)`$ that the pair virial misses. The standard
pressure tail correction contains this term with $`g(r_c) = 1`$, so `NoShift()` with
`tail_correction=true` is consistent up to $`g(r_c) - 1`$, whereas a shifted potential with
`tail_correction=true` is not. The documentation has the details.

## Observers and output

Observers are passed to `run!`/`equilibrate!` and called after every sweep, each with its
own schedule, `Every(n)` or `LogSchedule(; base=1.35, first=1)` for logarithmically spaced
sweeps. Sweeps are counted over the whole life of a simulation.

- `ThermoWriter(path, quantities...; schedule=Every(100))`: text columns preceded by the
  sweep, with `#` header lines giving the package version, the seed, the ensemble and the
  system. Quantities: `PotentialEnergy()` (per particle), `Pressure()`,
  `VirtualPressure()`, `Density()`, `PackingFraction()`, `Volume()`, `Acceptance(:swap)`,
  `StepSize(:displacement)`.
- `TrajectoryWriter(path; schedule=Every(1000))`: extended XYZ frames.
- `TimeSeries(quantities...; schedule=Every(1))`: in memory, `ts["energy"]`.
- `Callback(f; schedule)`: calls `f(sim)`.
- `ProgressPrinter(; schedule=Every(1000))`: prints the energy, the density and the
  acceptance rates.

Files are opened at the first due sweep and stay open across calls of `run!`; `close`
them at the end.

```julia
energies = Float64[]
cb = Callback(s -> push!(energies, s.system.energy); schedule=LogSchedule())
ts = TimeSeries(PotentialEnergy(), StepSize(:displacement); schedule=Every(10))
run!(sim, 1_000; observers=(cb, ts, ProgressPrinter(; schedule=Every(500))))
ts["energy"][end], length(energies)
```

Configurations are written in the extended XYZ format of MolecularDynamics.jl, which
OVITO and ASE also read: a 9-component `Lattice` (2D boxes are padded with a unit $`z`$
vector and `pbc="T T F"`), the columns `type id radius x y z`, and the sweep. Floats are
written with full precision. `write_extxyz(path, sys)` writes one frame and
`read_extxyz(path; frame=:last)` reads one back, also from files written by
MolecularDynamics.jl:

```julia
write_extxyz("output/final.xyz", sim.system; sweep=sim.sweep)
f = read_extxyz("output/final.xyz")  # (; box, positions, diameters, sweep)
sys2 = System(f.box, f.positions, f.diameters, Harmonic(1.0))
```

## Ramps

The temperature of `NVT` and the temperature and pressure of `NPT` can be functions of the
sweep. `LinearRamp(initial, final, nsweeps)` and `ExponentialRamp(initial, final, nsweeps)`
are constant after `nsweeps`; `NVT(0.0)` is a zero temperature quench.

```julia
quench = Simulation(
    sim.system, NVT(ExponentialRamp(1e-3, 1e-6, 500)), Displacement(0.01) => 1.0
)
run!(quench, 1_000)
```

## Checkpoints

`save_checkpoint` stores the complete simulation (system, moves with their step sizes and
counters, the state of the random number generator and the sweep); a run continued from
`load_checkpoint` gives bitwise the same result as the uninterrupted one. Observers are not
saved. Checkpoints use the `Serialization` standard library, so they are restart files for
the same Julia and package versions, not an archive format.

```julia
save_checkpoint("output/checkpoint.jls", sim)
sim = load_checkpoint("output/checkpoint.jls")
run!(sim, 100)
```

## Examples

The scripts in `examples/` run in under a minute with their defaults, from the repository
root with `julia --project examples/<name>.jl`. Output goes to `examples/output/`, which
is ignored by git.

- `ninarello2d.jl [kT] [N] [nsweeps]`: 2D swap Monte Carlo of the Ninarello mixture, with
  thermo, trajectory, final configuration and checkpoint files.
- `hs_npt3d.jl [N]`: hard-sphere compression with pressure ramps, $`\langle\rho\rangle`$
  compared with the Kolafa equation of state.
- `lj_nvt.jl [nsweeps]`: Lennard-Jones fluid truncated at $`3\sigma`$ with tail
  corrections, compared with the NIST reference data.
- `custom_potential.jl`: a user-defined potential, written both with `pair_energy` and
  in the MolecularDynamics.jl style.

## Documentation

The documentation is built locally with Documenter.jl, from the repository root:

```shell
julia --project=docs -e 'import Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

and then open `docs/build/index.html`. It covers the theory of the moves in more detail
(detailed balance of swaps, the volume move Jacobian, pressure estimators) and the full
API.

## Running the tests

Tests run locally (there is no CI). They are split into groups selected with the
environment variable `SWAPMC_TEST_GROUP`: `unit` (default, fast unit tests, allocation and
code quality checks), `stat` (statistical checks of the sampled distributions, a few
minutes) and `all`.

```shell
julia --project -e 'using Pkg; Pkg.test()'
SWAPMC_TEST_GROUP=stat julia --project -e 'using Pkg; Pkg.test()'
SWAPMC_TEST_GROUP=all julia --project -e 'using Pkg; Pkg.test()'
```

A single test file runs on its own with the shared helpers:

```shell
julia --project=test -e 'include("test/helpers.jl"); include("test/test_box.jl")'
```

## Validation and benchmarks

- `validation/`: longer physics checks run by hand against reference data (hard-sphere
  and hard-disk equations of state, Lennard-Jones NIST data). `validation/references.jl`
  holds the reference equations of state and data, and depends only on Base.
- `benchmark/`: timings against the original `swapmc.jl` script that this package
  replaces; see `benchmark/README.md` and `benchmark/baseline/RESULTS.md`.

## Code style

The code follows [Blue style](https://github.com/JuliaDiff/BlueStyle) (see
`.JuliaFormatter.toml`). Format it with

```shell
julia --project=@formatter -e 'using JuliaFormatter; format(".")'
```

where `@formatter` is a shared environment with JuliaFormatter installed
(`julia --project=@formatter -e 'import Pkg; Pkg.add("JuliaFormatter")'`).

## Changes between versions

See [CHANGELOG.md](CHANGELOG.md).
