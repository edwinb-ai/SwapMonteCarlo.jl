# API reference

Exported names are available after `using SwapMonteCarlo`. The extension interface
(potentials, moves, observers and quantities) is public but not exported: use it as
`SwapMonteCarlo.pair_energy`, or `import SwapMonteCarlo: pair_energy` to add methods.

```@meta
CurrentModule = SwapMonteCarlo
```

```@docs
SwapMonteCarlo
```

## Boxes

```@docs
OrthorhombicBox
CubicBox
volume
AbstractBox
```

## Pair potentials

```@docs
PolySoftSphere
LennardJones
HardSphere
PseudoHS
ShiftMode
NoShift
EnergyShift
ForceShift
```

### Mixing

```@docs
MixingRule
Additive
NonAdditive
mix
```

### Potential interface

```@docs
Potential
pair_energy
cutoff
pair_virial
evaluate
evaluate_r2
is_repulsive
is_hard
energy_lrc
pressure_lrc
max_cutoff
```

## Diameters

```@docs
SizeDistribution
Monodisperse
Bidisperse
InversePowerLaw
UniformSizes
sample_diameters
polydispersity
quantile
```

## Systems

```@docs
System
Base.length(::System)
positions
diameters
density
packing_fraction
total_energy
pressure
particle_energy
total_virial
check_energy
sync_energy!
```

## Initialization

```@docs
box_for_density
box_for_packing_fraction
lattice_positions
random_positions
insert_hard_spheres
```

## Ensembles and ramps

```@docs
Ensemble
NVT
NPT
conditions
LinearRamp
ExponentialRamp
```

## Moves

```@docs
Displacement
Swap
VolumeMove
MCMove
attempt!
name
MoveStats
AdaptiveStep
record!
adaptive_step
```

## Simulation

```@docs
Simulation
run!
equilibrate!
reset_stats!
acceptance_rate
step_size
```

## Observers

```@docs
Observer
observe!
ThermoWriter
TrajectoryWriter
TimeSeries
Base.getindex(::TimeSeries, ::AbstractString)
Callback
ProgressPrinter
```

### Schedules

```@docs
Schedule
isdue
Every
LogSchedule
```

### Quantities

```@docs
Quantity
measure
label
PotentialEnergy
Pressure
VirtualPressure
Density
PackingFraction
Volume
Acceptance
StepSize
```

## Input and output

```@docs
write_extxyz
read_extxyz
save_checkpoint
load_checkpoint
```

## Writing custom moves

Functions used when implementing a new [`MCMove`](@ref).

```@docs
adapt!
move_stats
check_compatible
```
