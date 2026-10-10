# Simulations and output

```@meta
DocTestSetup = :(using SwapMonteCarlo)
```

## Simulation

```julia
Simulation(system, ensemble, move1 => weight1, move2 => weight2, ...; seed, rng)
```

A [`Simulation`](@ref) holds the [`System`](@ref), the ensemble, the moves with their
normalized cumulative weights, the random number generator, its seed and the number of
sweeps done so far. The constructor checks that

- the weights are positive and finite, and the move names unique (one move of each
  kind);
- [`NPT`](@ref) is used with a [`VolumeMove`](@ref), and a `VolumeMove` only with `NPT`;

and it warns if swap moves are used with identical diameters.

The system and the moves are modified in place. Moves hold acceptance counters and step
sizes, so every simulation needs its own moves; to continue on the same system with new
conditions, build a new `Simulation` and start its moves from the tuned steps:

```julia
δ = step_size(sim).displacement
sim2 = Simulation(sim.system, NVT(0.05), Displacement(δ) => 0.8, Swap() => 0.2)
```

**Random numbers.** The generator is `Xoshiro(seed)`. Without a `seed`, a random one is
drawn from the system's entropy and stored in `sim.seed`, and written to the headers of
the output files, so every run can be reproduced. A different generator can be passed
with `rng` (then `seed` is only informative).

## Ensembles and ramps

[`NVT`](@ref)`(kT)` and [`NPT`](@ref)`(kT, pressure)` take numbers or callables of the
sweep number. The conditions are evaluated once per sweep, before it, at the new sweep
number. [`LinearRamp`](@ref) and [`ExponentialRamp`](@ref) interpolate between two values
over a number of sweeps and stay constant afterwards:

```jldoctest
julia> ramp = LinearRamp(2.0, 0.5, 1000);

julia> ramp(0), ramp(500), ramp(5000)
(2.0, 1.25, 0.5)

julia> ExponentialRamp(1.0, 0.01, 100)(50) ≈ 0.1  # geometric mean of the end points
true
```

Since sweeps are counted from the creation of the simulation, a ramp starts at sweep 0
of the simulation it belongs to. Any function works, e.g. `NVT(t -> 0.5 + 0.5 * exp(-t /
1000))`. `NVT(0.0)` (or a ramp to zero) quenches toward a local energy minimum, since
only moves that do not increase the energy are accepted.

## Running

```julia
run!(sim, nsweeps; observers=(), adapt=false, energy_check=1000)
equilibrate!(sim, nsweeps; kwargs...)   # run! with adapt=true
```

One sweep is ``N`` attempts. [`equilibrate!`](@ref) tunes the step sizes (see
[Step sizes: tuned during equilibration only](@ref)); [`run!`](@ref) keeps them fixed.
After every sweep the observers are called; at the end of every `run!` they are flushed.
[`reset_stats!`](@ref) restarts the acceptance counters reported by
[`acceptance_rate`](@ref), and [`step_size`](@ref) reports the current steps.

## Observers

An observer is called after every sweep and decides with its own schedule whether to
record:

| Observer | Records |
|---|---|
| [`ThermoWriter`](@ref)`(path, quantities...; schedule=Every(100))` | columns of numbers in a text file |
| [`TrajectoryWriter`](@ref)`(path; schedule=Every(1000))` | extended XYZ frames |
| [`TimeSeries`](@ref)`(quantities...; schedule=Every(1))` | values in memory, `ts["energy"]` |
| [`Callback`](@ref)`(f; schedule=Every(1))` | calls `f(sim)` |
| [`ProgressPrinter`](@ref)`(; schedule=Every(1000), io=stdout)` | energy, density and acceptance rates |

Files are created at the first due sweep (overwriting existing ones) and stay open across
calls of `run!`, so the same observers can be passed to consecutive runs; `close` them at
the end. Observers are not part of the simulation and are not saved in checkpoints.

### Schedules

- [`Every`](@ref)`(n)`: at the sweeps that are multiples of `n`.
- [`LogSchedule`](@ref)`(; base=1.35, first=1)`: at
  ``t_1 = \text{first}``, ``t_{k+1} = \max(t_k + 1, \lceil \text{base}\, t_k \rceil)``,
  i.e. `1, 2, 3, 5, 7, 10, 14, 19, 26, 36, …`, for correlation functions and aging.

Since sweeps count from the creation of the simulation, the output of consecutive runs
stays on the same grid.

```jldoctest
julia> s = LogSchedule();

julia> filter(t -> SwapMonteCarlo.isdue(s, t), 1:40)
10-element Vector{Int64}:
  1
  2
  3
  5
  7
 10
 14
 19
 26
 36
```

### Quantities

| Quantity | Label | Value |
|---|---|---|
| [`PotentialEnergy`](@ref)`()` | `energy` | ``U/N``, including long range corrections |
| [`Pressure`](@ref)`()` | `pressure` | virial pressure (not for hard potentials) |
| [`VirtualPressure`](@ref)`(; xi=1e-3)` | `virtual_pressure` | pressure of hard potentials |
| [`Density`](@ref)`()` | `density` | ``N/V`` |
| [`PackingFraction`](@ref)`()` | `packing_fraction` | ``\sum_i v(\sigma_i)/V`` |
| [`Volume`](@ref)`()` | `volume` | ``V`` |
| [`Acceptance`](@ref)`(name)` | `acceptance_<name>` | acceptance since the last `reset_stats!` |
| [`StepSize`](@ref)`(name)` | `step_<name>` | current step size |

Move names are `:displacement`, `:swap` and `:volume`. A custom quantity is a subtype of
[`SwapMonteCarlo.Quantity`](@ref) with methods of
[`measure`](@ref SwapMonteCarlo.measure)`(q, sim)` and
[`label`](@ref SwapMonteCarlo.label)`(q)`; a custom observer is a subtype of
[`SwapMonteCarlo.Observer`](@ref) with a method of
[`observe!`](@ref SwapMonteCarlo.observe!)`(obs, sim)`.

## Output formats

### Thermo files

[`ThermoWriter`](@ref) writes `#` header lines (package version, seed, ensemble, system,
moves and the column labels) and then one line per due sweep, with full precision
numbers:

```text
# SwapMonteCarlo.jl v0.1.0
# seed = 42
# ensemble = NVT{Float64}(0.1)
# system = System{2,Float64}: N=1000, V=1000.0, ρ=1.0, PolySoftSphere
# moves = displacement swap
# sweep energy pressure acceptance_swap
2100 0.30977152690825993 3.5487039370986655 0.13631571019731886
```

It is read with e.g. `readdlm(path; comments=true)` from `DelimitedFiles`.

### Extended XYZ

Configurations are written in the extended XYZ format used by MolecularDynamics.jl,
which OVITO and ASE also read:

```text
2
Lattice="10.0 0.0 0.0 0.0 8.0 0.0 0.0 0.0 1.0" Properties=type:I:1:id:I:1:radius:R:1:pos:R:3 pbc="T T F" Sweep=100
1 1 0.5 1.25 3.5 0.0
1 2 0.7 4.0 0.125 0.0
```

`Lattice` lists the three box vectors; 2D boxes are padded with a unit ``z`` vector and
`pbc="T T F"`, and positions have ``z = 0``. The columns are the type, the particle index,
the *radius* (half the diameter) and the Cartesian position. Floats are written with the
shortest representation that parses back to the same `Float64`, so files round-trip
exactly.

[`write_extxyz`](@ref) writes one frame (`mode="a"` appends), [`TrajectoryWriter`](@ref)
appends one per due sweep, and [`read_extxyz`](@ref)`(path; frame=:last)` reads one
frame (`:first`, `:last` or its number) as `(; box, positions, diameters, sweep)`.
Incomplete frames at the end of a file, e.g. from an interrupted run, are ignored. Files
written by MolecularDynamics.jl, including older 2D files with a 4-component lattice, are
read as well; pass `Val(D)` for a type stable result.

## Checkpoints

[`save_checkpoint`](@ref)`(path, sim)` stores the complete simulation: the system
(positions, diameters, box, running energy, cell list), the ensemble, the moves with
their step sizes and counters, the state of the random number generator, the seed and
the sweep. [`load_checkpoint`](@ref)`(path)` restores it, and continuing the restored
simulation gives bitwise the same results as continuing the original one.

```julia
save_checkpoint("checkpoint.jls", sim)
sim = load_checkpoint("checkpoint.jls")
thermo = ThermoWriter("thermo_part2.txt", PotentialEnergy())
run!(sim, 10_000; observers=thermo)
```

Observers are not saved, and a new [`ThermoWriter`](@ref) or [`TrajectoryWriter`](@ref)
overwrites its file at its first due sweep, so a continued run writes to new files.

The file is written to a temporary name and renamed, so an interrupted save does not
destroy the previous checkpoint. It uses the `Serialization` standard library, so it is
a restart file for the same Julia version and package versions (including packages that
define custom potentials or moves), not an archive format; keep configurations with
[`write_extxyz`](@ref). Loading a checkpoint written by another version warns.
