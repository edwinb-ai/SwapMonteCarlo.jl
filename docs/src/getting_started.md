# Getting started

```@meta
DocTestSetup = :(using SwapMonteCarlo)
```

## Installation

SwapMonteCarlo.jl needs Julia 1.13. It is not registered; add it from a local clone,

```julia
using Pkg: Pkg
Pkg.develop(; path="/path/to/SwapMonteCarlo.jl")
```

or from its repository (placeholder URL) with
`Pkg.add(; url="https://github.com/edwinb-ai/SwapMonteCarlo.jl")`.

The code examples on this page are run when the documentation is built, with small
systems so that the build stays fast.

```@setup start
using SwapMonteCarlo
# Output files of the examples go to a temporary directory.
outdir = mktempdir()
```

## The workflow

A simulation is built in four steps:

1. **Diameters**, from a size distribution, see [Size distributions](@ref).
2. **A box and initial positions**, see [Initial configurations](@ref).
3. **A [`System`](@ref)**: box, positions, diameters and pair potential (see
   [Potentials](@ref)).
4. **A [`Simulation`](@ref)**: the system, an ensemble ([`NVT`](@ref) or [`NPT`](@ref))
   and the moves with their relative weights.

Then [`equilibrate!`](@ref) runs with step size tuning and [`run!`](@ref) runs the
production with fixed step sizes, passing observers that record data.

## Swap Monte Carlo of a 2D polydisperse liquid

The continuously polydisperse soft-sphere mixture of Ninarello, Berthier and Coslovich:
diameters with ``P(\sigma) \propto \sigma^{-3}`` on ``[0.73, 1.62]`` (mean diameter
``\approx 1``), the [`PolySoftSphere`](@ref) potential (``r^{-12}`` repulsion, smooth at
the cutoff ``1.25\,\sigma_{ij}``, non-additivity ``0.2``), at number density ``\rho = 1``.

```@example start
using SwapMonteCarlo
using Random: Xoshiro

N = 256
kT = 0.2
rng = Xoshiro(2026)

σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N)
box = box_for_density(Val(2), N, 1.0)
x = lattice_positions(box, N)
sys = System(box, x, σ, PolySoftSphere())
```

The dimension comes from `Val(2)`; positions are vectors of `SVector{2,Float64}`. The
[`System`](@ref) constructor wraps the positions into the box, checks that the largest
pair cutoff fits in the box (minimum image convention) and that the pair energy is
symmetric in the diameters, builds the cell list and computes the energy.

The simulation attempts displacements with probability 0.8 and swaps with probability
0.2. One sweep is ``N`` attempts.

```@example start
sim = Simulation(sys, NVT(kT), Displacement(0.1) => 0.8, Swap() => 0.2; seed=42)
```

The `seed` makes the run reproducible; without it a random seed is drawn and stored in
`sim.seed` (it is written to the headers of the output files).

During equilibration the maximum displacement is tuned toward 40 % acceptance. Then the
acceptance counters are reset and the production run keeps the step size fixed:

```@example start
equilibrate!(sim, 500)
step_size(sim)
```

```@example start
reset_stats!(sim)
thermo_path = joinpath(outdir, "thermo.txt")
thermo = ThermoWriter(
    thermo_path, PotentialEnergy(), Pressure(), Acceptance(:swap); schedule=Every(100)
)
traj = TrajectoryWriter(joinpath(outdir, "trajectory.xyz"); schedule=Every(500))
run!(sim, 1_000; observers=(thermo, traj))
close(thermo)
close(traj)
acceptance_rate(sim)
```

The thermo file has comment lines with the version, the seed and the state, then one
line per due sweep:

```@example start
print(read(thermo_path, String))
```

`examples/ninarello2d.jl` is the full-size version of this run (``N = 1000``,
temperature and size from the command line).

## Hard spheres at constant pressure

Hard spheres need a start without overlaps: [`insert_hard_spheres`](@ref) places them by
random sequential insertion, largest first, which works up to packing fractions of about
0.38 in 3D (0.55 in 2D). Denser states are reached by compression in the ``NPT``
ensemble. The pressure of hard spheres is measured with virtual compressions,
[`VirtualPressure`](@ref).

```@example start
using Statistics: mean

N = 256
rng = Xoshiro(7)
σ = ones(N)
box = box_for_packing_fraction(Val(3), σ, 0.35)
sys = System(box, insert_hard_spheres(rng, box, σ), σ, HardSphere())

sim = Simulation(
    sys, NPT(1.0, 5.0), Displacement(0.1) => 1.0, VolumeMove(0.01) => 0.01; seed=1
)
equilibrate!(sim, 3_000)
reset_stats!(sim)
ts = TimeSeries(Density(), VirtualPressure(); schedule=Every(10))
run!(sim, 2_000; observers=ts)
(mean(ts["density"]), mean(ts["virtual_pressure"]), acceptance_rate(sim))
```

The Kolafa equation of state gives ``\rho = 0.7504`` at ``\beta P = 5``; this short run
of a small system is only roughly converged. `examples/hs_npt3d.jl` compresses through
several pressures with pressure ramps and compares with the equation of state.

Volume moves recompute the total energy, which costs ``O(N)``, so their weight is
usually chosen such that there are a few of them per sweep; here 1 % of the attempts.

## Size distributions

[`sample_diameters`](@ref) draws diameters from a [`SizeDistribution`](@ref
SwapMonteCarlo.SizeDistribution):

| Distribution | Diameters |
|---|---|
| [`Monodisperse`](@ref)`(σ)` | all equal to `σ` |
| [`Bidisperse`](@ref)`(σa, σb, fraction_a)` | exactly `round(Int, N * fraction_a)` of diameter `σa` |
| [`InversePowerLaw`](@ref)`(σmin, σmax; exponent=3)` | ``P(\sigma) \propto \sigma^{-p}`` on ``[\sigma_\text{min}, \sigma_\text{max}]`` |
| [`UniformSizes`](@ref)`(σmin, σmax)` | uniform on ``[\sigma_\text{min}, \sigma_\text{max}]`` |

With `stratified=true`, the diameters are the quantiles ``F^{-1}((k - 1/2)/N)`` in random
order, so that their moments match the distribution up to ``O(1/N^2)``, which removes
sample-to-sample fluctuations of the polydispersity.

```jldoctest
julia> using Random: Xoshiro

julia> σ = sample_diameters(Xoshiro(1), InversePowerLaw(0.73, 1.62), 10_000; stratified=true);

julia> round(polydispersity(σ); digits=3)
0.229

julia> extrema(sample_diameters(Xoshiro(1), Bidisperse(1.0, 1.4, 0.5), 6))
(1.0, 1.4)
```

## Initial configurations

| Function | Result |
|---|---|
| [`box_for_density`](@ref)`(Val(D), N, ρ)` | cubic box at number density `ρ` |
| [`box_for_packing_fraction`](@ref)`(Val(D), σ, φ)` | cubic box at packing fraction `φ` |
| [`lattice_positions`](@ref)`(box, N)` | hypercubic lattice, ``\lceil N^{1/D} \rceil`` sites per side |
| [`random_positions`](@ref)`(rng, box, N)` | uniformly random (overlaps allowed) |
| [`insert_hard_spheres`](@ref)`(rng, box, σ)` | random sequential insertion without overlaps |

Boxes can also be built directly with [`CubicBox`](@ref)`(Val(D), L)` or
[`OrthorhombicBox`](@ref)`(Lx, Ly[, Lz])`. A configuration saved by
[`write_extxyz`](@ref) (or by MolecularDynamics.jl) is read with
[`read_extxyz`](@ref):

```@example start
path = joinpath(outdir, "final.xyz")
write_extxyz(path, sim.system; sweep=sim.sweep)
frame = read_extxyz(path)
sys2 = System(frame.box, frame.positions, frame.diameters, HardSphere())
```

## Running the examples

The scripts in `examples/` run in under a minute with their defaults:

```shell
julia --project examples/ninarello2d.jl 0.1 1000   # kT and N
julia --project examples/hs_npt3d.jl
julia --project examples/lj_nvt.jl
julia --project examples/custom_potential.jl
```

They write to `examples/output/`, which git ignores.
