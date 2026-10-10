# Lennard-Jones fluid in the NVT ensemble, truncated at rc = 3σ with long range (tail)
# corrections, compared with the NIST Standard Reference Simulation Website Monte Carlo
# data (N = 500, kT = 0.9, ρ = 0.82: U/N = -5.7456(8), P = 0.8239(29)).
#
# Usage, from the repository root:
#
#     julia --project examples/lj_nvt.jl [nsweeps]
#
# Default nsweeps = 10000 production sweeps (about 30 s). The potential is truncated but
# not shifted (`NoShift`), as in the NIST data, and the tail corrections of the full
# potential beyond rc are added to the energy and to the virial pressure.

using SwapMonteCarlo
using Printf: @printf
using Random: Xoshiro
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "validation", "references.jl"))

"""
    block_error(x; nblocks=20)

Standard error of the mean of the correlated series `x` from `nblocks` block averages.
"""
function block_error(x::AbstractVector; nblocks::Int=20)
    m = length(x) ÷ nblocks
    blocks = [mean(@view x[((k - 1) * m + 1):(k * m)]) for k in 1:nblocks]
    return std(blocks) / sqrt(nblocks)
end

"""
    main(; N=500, kT=0.9, ρ=0.82, nequil=3_000, nsweeps=10_000, seed=1)

Simulate `N` Lennard-Jones particles at temperature `kT` and density `ρ` and print the
average energy per particle and pressure next to the NIST reference values.
"""
function main(; N=500, kT=0.9, ρ=0.82, nequil=3_000, nsweeps=10_000, seed=1)
    σ = ones(N)
    box = box_for_density(Val(3), N, ρ)
    pot = LennardJones(; xc=3.0, shift=NoShift(), tail_correction=true)
    # A simple cubic lattice is a valid (finite energy) start for a soft potential.
    sys = System(box, lattice_positions(box, N), σ, pot)
    sim = Simulation(sys, NVT(kT), Displacement(0.1) => 1.0; seed=seed)

    # Melt the lattice and tune the maximum displacement toward 40 % acceptance.
    equilibrate!(sim, nequil)
    reset_stats!(sim)

    # `PotentialEnergy` and `Pressure` include the tail corrections.
    series = TimeSeries(PotentialEnergy(), Pressure(); schedule=Every(10))
    run!(sim, nsweeps; observers=series)
    (U, P) = (series["energy"], series["pressure"])

    reference = only(r for r in LJ_NIST_MC_RC3_LRC if r.T == kT && r.ρ == ρ)
    @printf(
        "N = %d, kT = %.2f, ρ = %.3f, δ = %.4f, ", N, kT, ρ, step_size(sim).displacement
    )
    @printf("acceptance = %.3f\n", acceptance_rate(sim).displacement)
    @printf("%10s %12s %10s %12s %10s\n", "", "this run", "error", "NIST", "error")
    rows = (("U/N", U, reference.U, reference.dU), ("P", P, reference.P, reference.dP))
    for (label, x, ref, dref) in rows
        @printf(
            "%10s %12.5f %10.5f %12.5f %10.5f\n", label, mean(x), block_error(x), ref, dref
        )
    end
    return sim
end

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
main(; nsweeps=nsweeps)
