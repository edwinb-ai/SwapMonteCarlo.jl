# User-defined pair potentials.
#
# Usage, from the repository root:
#
#     julia --project examples/custom_potential.jl
#
# Runs in a few seconds. A potential is a subtype of `SwapMonteCarlo.Potential` with two
# required methods, `pair_energy(pot, r2, σi, σj)` (from the squared distance) and
# `cutoff(pot, σi, σj)`. Optional methods add the virial pressure (`pair_virial`) and
# faster rejections for non-negative potentials (`is_repulsive`). Alternatively, a
# potential written for MolecularDynamics.jl, with `evaluate(pot, r, σi, σj) -> (u, f)`,
# works unchanged.
#
# Here the same potential, the Weeks-Chandler-Andersen (WCA) repulsion of polydisperse
# particles, is defined both ways, the two definitions are checked against each other,
# and a swap Monte Carlo simulation is run with the first one.

using SwapMonteCarlo
using Printf: @printf
using Random: Xoshiro
using Statistics: mean

# Always `import` the functions that get new methods.
import SwapMonteCarlo: Potential, pair_energy, pair_virial, cutoff, is_repulsive, evaluate
using SwapMonteCarlo: mix

"""
    WCA(; epsilon=1.0, mixing=Additive())

Weeks-Chandler-Andersen potential, `u = 4ε [(σij/r)^12 - (σij/r)^6] + ε` for
`r < 2^(1/6) σij` and zero beyond, written with the `pair_energy` interface.
"""
struct WCA{M<:SwapMonteCarlo.MixingRule} <: Potential
    epsilon::Float64
    mixing::M
end
WCA(; epsilon=1.0, mixing=Additive()) = WCA(Float64(epsilon), mixing)

# The cutoff may depend on the diameters; here it is the minimum of the LJ potential.
cutoff(p::WCA, σi, σj) = 2^(1 / 6) * mix(p.mixing, σi, σj)

# Energy from the squared distance: no square root is needed. It must be exactly zero
# beyond the cutoff and symmetric in (σi, σj), which the mixing rules guarantee.
function pair_energy(p::WCA, r2, σi, σj)
    σij = mix(p.mixing, σi, σj)
    x2 = σij * σij / r2
    x2 > 2^(-1 / 3) || return zero(r2)
    x6 = x2 * x2 * x2
    return 4 * p.epsilon * x6 * (x6 - 1) + p.epsilon
end

# Pair virial -r du/dr, used by `pressure` and `Pressure()`.
function pair_virial(p::WCA, r2, σi, σj)
    σij = mix(p.mixing, σi, σj)
    x2 = σij * σij / r2
    x2 > 2^(-1 / 3) || return zero(r2)
    x6 = x2 * x2 * x2
    return 24 * p.epsilon * x6 * (2 * x6 - 1)
end

# All pair energies are non-negative: energy sums stop as soon as a move is rejected.
is_repulsive(::WCA) = true

"""
    WCAEvaluate(; epsilon=1.0)

The same potential (with additive mixing) in the MolecularDynamics.jl style:
`evaluate(pot, r, σi, σj)` returns the energy and the force `f = -du/dr`.
"""
struct WCAEvaluate <: Potential
    epsilon::Float64
end
WCAEvaluate(; epsilon=1.0) = WCAEvaluate(Float64(epsilon))

function evaluate(p::WCAEvaluate, r, σi, σj)
    σij = (σi + σj) / 2
    r < 2^(1 / 6) * σij || return (zero(r), zero(r))
    x6 = (σij / r)^6
    u = 4 * p.epsilon * x6 * (x6 - 1) + p.epsilon
    f = 24 * p.epsilon * x6 * (2 * x6 - 1) / r
    return (u, f)
end
cutoff(::WCAEvaluate, σi, σj) = 2^(1 / 6) * (σi + σj) / 2

"""
    main(; N=1000, kT=1.0, φ=0.6, nsweeps=5_000, seed=3)

Check that both definitions give the same energy and pressure, then run swap Monte Carlo
of `N` polydisperse WCA disks at packing fraction `φ` and temperature `kT`.
"""
function main(; N=1000, kT=1.0, φ=0.6, nsweeps=5_000, seed=3)
    rng = Xoshiro(seed)
    σ = sample_diameters(rng, UniformSizes(0.8, 1.2), N)
    box = box_for_packing_fraction(Val(2), σ, φ)
    x = lattice_positions(box, N)

    sys = System(box, x, σ, WCA())
    sys_evaluate = System(box, x, σ, WCAEvaluate())
    @printf(
        "U/N: pair_energy %.12f, evaluate %.12f\n",
        total_energy(sys) / N,
        total_energy(sys_evaluate) / N
    )
    @printf(
        "P:   pair_virial %.12f, evaluate %.12f\n",
        pressure(sys, kT),
        pressure(sys_evaluate, kT)
    )

    sim = Simulation(sys, NVT(kT), Displacement(0.1) => 0.9, Swap() => 0.1; seed=seed)
    equilibrate!(sim, 1_000)
    reset_stats!(sim)
    series = TimeSeries(PotentialEnergy(), Pressure(); schedule=Every(10))
    run!(sim, nsweeps; observers=series)
    rates = acceptance_rate(sim)
    @printf("polydisperse WCA disks, φ = %.2f, kT = %.2f:\n", φ, kT)
    @printf("⟨U/N⟩ = %.5f, ⟨P⟩ = %.5f\n", mean(series["energy"]), mean(series["pressure"]))
    @printf("acceptance: displacement %.3f, swap %.3f\n", rates.displacement, rates.swap)
    return sim
end

main()
