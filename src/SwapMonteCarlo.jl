"""
    SwapMonteCarlo

Metropolis Monte Carlo simulations of (polydisperse) particle systems in two and three
dimensions, in the NVT and NPT ensembles, with displacement, swap and volume moves and
user-defined pair potentials.
"""
module SwapMonteCarlo

using LinearAlgebra: dot
using Printf: @printf, @sprintf
using Random: Random, AbstractRNG, Xoshiro, RandomDevice, randexp, shuffle!
using Serialization: serialize, deserialize
using StaticArrays: SVector, @SVector

include("box.jl")
include("mixing.jl")
include("potentials/interface.jl")
include("potentials/softsphere.jl")
include("potentials/lennardjones.jl")
include("potentials/hardsphere.jl")
include("potentials/pseudohs.jl")
include("distributions.jl")
include("celllist.jl")
include("system.jl")
include("energy.jl")
include("initialization.jl")
include("ramps.jl")
include("ensembles.jl")
include("moves/stats.jl")
include("moves/displacement.jl")
include("moves/swap.jl")
include("moves/volume.jl")
include("moves/moveset.jl")
include("io/extxyz.jl")
include("observers/schedules.jl")
include("observers/quantities.jl")
include("observers/observers.jl")
include("simulation.jl")
include("checkpoint.jl")

# Boxes
export OrthorhombicBox, CubicBox, volume
# Potentials and mixing rules
export PolySoftSphere, LennardJones, HardSphere, PseudoHS
export NoShift, EnergyShift, ForceShift
export Additive, NonAdditive
# Size distributions
export Monodisperse, Bidisperse, InversePowerLaw, UniformSizes, sample_diameters
export polydispersity
# Systems and initialization
export System, positions, diameters, density, packing_fraction
export total_energy, pressure
export lattice_positions, random_positions, insert_hard_spheres
export box_for_density, box_for_packing_fraction
# Ensembles, ramps and moves
export NVT, NPT, LinearRamp, ExponentialRamp
export Displacement, Swap, VolumeMove
# Simulation
export Simulation, run!, equilibrate!, reset_stats!, acceptance_rate, step_size
# Observers and IO
export ThermoWriter, TrajectoryWriter, TimeSeries, Callback, ProgressPrinter
export Every, LogSchedule
export PotentialEnergy, Pressure, VirtualPressure, Density, PackingFraction, Volume
export Acceptance, StepSize
export write_extxyz, read_extxyz, save_checkpoint, load_checkpoint

# Extension interface: not exported, but part of the public API.
public AbstractBox, Potential, pair_energy, pair_virial, cutoff, max_cutoff
public evaluate, evaluate_r2, is_repulsive, is_hard, energy_lrc, pressure_lrc
public MixingRule, mix, ShiftMode, SizeDistribution, quantile
public Ensemble, MCMove, attempt!, Observer, observe!, Quantity, measure, label
public Schedule, particle_energy, total_virial, check_energy, sync_energy!

end
