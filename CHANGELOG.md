# Changelog

All notable changes to SwapMonteCarlo.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is below 1.0, a minor version (0.x) may break the API.

## [Unreleased]

## [0.1.0] - 2026-10-10

### Added

- Metropolis Monte Carlo of particles in 2D and 3D periodic orthorhombic boxes
  (`OrthorhombicBox`, `CubicBox`), with positions stored in fractional coordinates and
  the dimension and float type taken from the types.
- `System(box, positions, diameters, potential)` with a cell list, a running total
  energy (summed in `Float64` also for `Float32` systems, see `energy_type`), and the
  accessors `positions`, `diameters`, `volume`, `density`, `packing_fraction`,
  `total_energy` and `pressure`.
- Built-in pair potentials: `PolySoftSphere` (the SS12 potential of Ninarello, Berthier
  and Coslovich, C² at its cutoff), `LennardJones` (`NoShift`, `EnergyShift` or
  `ForceShift`, optional tail corrections of the energy and pressure in 2D and 3D with
  `NoShift`, the convention of the NIST reference data),
  `HardSphere` and `PseudoHS` (the (50, 49) pseudo hard sphere potential).
- Mixing rules `Additive()` and `NonAdditive(ε)` for the pair diameter of every built-in
  potential; a `System` whose diameters give negative pair diameters is rejected.
- Extension interface for user-defined potentials: `pair_energy` and `cutoff`, optional
  `pair_virial`, `is_repulsive`, `is_hard`, `energy_lrc`, `pressure_lrc` and
  `max_cutoff`. Potentials written for MolecularDynamics.jl with
  `evaluate(pot, r, σi, σj) -> (u, f)` work unchanged. The symmetry of the pair energy in
  the diameters is checked when a `System` is built.
- Size distributions `Monodisperse`, `Bidisperse`, `InversePowerLaw` and `UniformSizes`,
  `sample_diameters` (optionally stratified) and `polydispersity`.
- Initial configurations: `box_for_density`, `box_for_packing_fraction`,
  `lattice_positions`, `random_positions` and `insert_hard_spheres` (random sequential
  insertion).
- Ensembles `NVT(kT)` and `NPT(kT, pressure)`, with the temperature and the pressure
  either constant or functions of the sweep, such as `LinearRamp` and `ExponentialRamp`,
  and checked to be finite (and the temperature non-negative) at every sweep.
- Moves `Displacement`, `Swap` (exchange of the diameters of two particles, with an
  optional `max_dsigma`) and `VolumeMove` (isotropic, uniform in `ln V`), all with the
  Metropolis rule with the threshold drawn first, which never evaluates `exp` and stops
  energy sums of repulsive potentials at the first certain rejection.
- `Simulation` with weighted random move selection, `run!`, `equilibrate!` (tunes the
  step sizes toward a target acceptance ratio; `run!` keeps them fixed),
  `reset_stats!`, `acceptance_rate`, `step_size`, and a periodic check of the running
  energy against a full recomputation.
- Observers `ThermoWriter`, `TrajectoryWriter`, `TimeSeries`, `Callback` and
  `ProgressPrinter`, with the schedules `Every` and `LogSchedule`, and the quantities
  `PotentialEnergy`, `Pressure`, `VirtualPressure` (hard potentials), `Density`,
  `PackingFraction`, `Volume`, `Acceptance` and `StepSize`.
- Extended XYZ input and output compatible with MolecularDynamics.jl, OVITO and ASE
  (`write_extxyz`, `read_extxyz`), written with full precision.
- Checkpoints of the complete simulation state, including the random number generator
  (`save_checkpoint`, `load_checkpoint`), for bitwise reproducible restarts; a
  simulation given the task-local default generator owns a copy of it.
- Unit and statistical test groups (`SWAPMC_TEST_GROUP`), validation references,
  a benchmark of the original `swapmc.jl` script, examples and a Documenter.jl manual
  built locally.

[Unreleased]: https://github.com/edwinb-ai/SwapMonteCarlo.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edwinb-ai/SwapMonteCarlo.jl/releases/tag/v0.1.0
