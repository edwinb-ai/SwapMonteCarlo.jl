# SwapMonteCarlo.jl

Metropolis Monte Carlo of (polydisperse) particles in two and three dimensions, in the
canonical (``NVT``) and isothermal-isobaric (``NPT``) ensembles.

Besides single particle displacements and volume moves, SwapMonteCarlo.jl implements
*swap* moves, which exchange the diameters of two particles. In dense polydisperse
liquids they equilibrate the system many orders of magnitude faster than local moves,
which made it possible to prepare equilibrium supercooled liquids below the
experimental glass transition in simulations (Ninarello, Berthier and Coslovich, Phys.
Rev. X 7, 021039, 2017).

## Features

- 2D and 3D periodic orthorhombic boxes; the dimension and the floating point type are
  part of the types, so every kernel is specialized and type stable.
- Built-in potentials: the smoothed soft spheres of Ninarello *et al.*
  ([`PolySoftSphere`](@ref)), [`LennardJones`](@ref) (truncated with optional tail
  corrections, energy- or force-shifted), [`HardSphere`](@ref) and the
  pseudo hard sphere potential [`PseudoHS`](@ref), each with an additive or non-additive
  mixing rule.
- User-defined pair potentials with two methods, and MolecularDynamics.jl potentials
  (`evaluate`) unchanged; see [Potentials](@ref).
- Exact Metropolis acceptance with the threshold drawn first, which stops energy sums
  of repulsive potentials (and hard spheres at the first overlap) as soon as a move is
  certain to be rejected; see [Moves and acceptance](@ref).
- Step sizes tuned toward a target acceptance ratio during equilibration only.
- Temperature and pressure ramps, observers writing thermodynamic data and extended
  XYZ trajectories compatible with MolecularDynamics.jl, OVITO and ASE, and bitwise
  reproducible checkpoints; see [Simulations and output](@ref).
- Cell lists in fractional coordinates: displacement and swap moves cost ``O(1)`` and
  do not allocate, and a volume move rebuilds the cell list only when the number of
  cells changes.

## A first example

```julia
using SwapMonteCarlo
using Random: Xoshiro

N = 1000
rng = Xoshiro(2026)
σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N)
box = box_for_density(Val(2), N, 1.0)
sys = System(box, lattice_positions(box, N), σ, PolySoftSphere())

sim = Simulation(sys, NVT(0.1), Displacement(0.1) => 0.8, Swap() => 0.2; seed=42)
equilibrate!(sim, 2_000)
reset_stats!(sim)
run!(sim, 10_000; observers=ThermoWriter("thermo.txt", PotentialEnergy(), Pressure()))
acceptance_rate(sim)
```

[Getting started](@ref) explains each step.

## Units and conventions

Reduced units with ``k_B = 1``: energies and temperatures are in units of the energy
scale of the potential, lengths in units of the diameters. A *sweep* is ``N`` move
attempts, and sweeps are counted over the whole life of a [`Simulation`](@ref). The
energy reported by [`PotentialEnergy`](@ref) is per particle and the pressure is
``P`` (not ``\beta P``). Areas in 2D are called volumes.

## Related packages

[MolecularDynamics.jl](https://github.com/edwinb-ai/MolecularDynamics.jl) is the
molecular dynamics sibling of this package. Potentials defined for it work here, and the
two read each other's extended XYZ files, so a configuration equilibrated with swap Monte
Carlo can be used as the starting point of a molecular dynamics run.
