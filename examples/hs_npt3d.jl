# Compression of a 3D hard-sphere fluid in the NPT ensemble with a pressure ramp, and
# comparison of the average density with the Kolafa-Labík-Malijevský equation of state.
#
# Usage, from the repository root:
#
#     julia --project examples/hs_npt3d.jl [N]
#
# Default N = 500 (about 30 s). For every pressure βP the box is compressed with a linear
# pressure ramp from the previous pressure, equilibrated, and then sampled with fixed step
# sizes. The reference density solves βP = ρ Z(ρ) with the Kolafa EOS from
# `validation/references.jl`. Near freezing the density decorrelates slowly (about 10³
# sweeps at βP = 8), so the error bars come from 10 blocks of 2000 sweeps each; the runs
# are short and deviations of a couple of error bars are expected.

using SwapMonteCarlo
using Printf: @printf
using Random: Xoshiro
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "validation", "references.jl"))

"""
    block_error(x; nblocks=10)

Standard error of the mean of the correlated series `x` from `nblocks` block averages.
"""
function block_error(x::AbstractVector; nblocks::Int=10)
    m = length(x) ÷ nblocks
    blocks = [mean(@view x[((k - 1) * m + 1):(k * m)]) for k in 1:nblocks]
    return std(blocks) / sqrt(nblocks)
end

"""
    main(; N=500, pressures=(2.0, 4.0, 6.0, 8.0), nramp=2_000, nequil=12_000,
         nprod=20_000, seed=1234)

Compress `N` hard spheres of unit diameter through the reduced pressures `pressures`
(`βP σ³`, at `kT = 1`) and print the average density at each of them next to the Kolafa
equation of state.
"""
function main(;
    N=500,
    pressures=(2.0, 4.0, 6.0, 8.0),
    nramp=2_000,
    nequil=12_000,
    nprod=20_000,
    seed=1234,
)
    rng = Xoshiro(seed)
    σ = ones(N)
    # Random sequential insertion of non-overlapping spheres at packing fraction 0.3
    # (ρ ≈ 0.57, where βP ≈ 2), well below its jamming limit φ ≈ 0.38.
    box = box_for_packing_fraction(Val(3), σ, 0.3)
    sys = System(box, insert_hard_spheres(rng, box, σ), σ, HardSphere())

    (δ, δlnV) = (0.2, 0.01)
    P_previous = 2.0
    @printf(
        "%6s %10s %10s %10s %8s %8s\n", "βP", "⟨ρ⟩", "error", "ρ_Kolafa", "acc_d", "acc_V"
    )
    for P in pressures
        # A new simulation on the same system for every pressure ramp; the moves start
        # from the step sizes tuned at the previous pressure. Volume moves are 1 % of the
        # attempts, i.e. about N/100 per sweep.
        ramp = LinearRamp(P_previous, P, nramp)
        sim = Simulation(
            sys,
            NPT(1.0, ramp),
            Displacement(δ) => 1.0,
            VolumeMove(δlnV) => 0.01;
            seed=rand(rng, UInt64),
        )
        # Compression along the ramp and equilibration at the final pressure, with step
        # size tuning (displacement toward 40 % and volume toward 30 % acceptance).
        equilibrate!(sim, nramp + nequil)
        (δ, δlnV) = (step_size(sim).displacement, step_size(sim).volume)

        reset_stats!(sim)
        series = TimeSeries(Density(), VirtualPressure(); schedule=Every(5))
        run!(sim, nprod; observers=series)
        ρ = series["density"]
        ρ_ref = density_from_pressure(hs_fluid_Z_kolafa, P; bracket=(1e-6, 0.94))
        rates = acceptance_rate(sim)
        @printf(
            "%6.2f %10.5f %10.5f %10.5f %8.3f %8.3f\n",
            P,
            mean(ρ),
            block_error(ρ),
            ρ_ref,
            rates.displacement,
            rates.volume
        )
        P_previous = P
    end
    return sys
end

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
main(; N=N)
