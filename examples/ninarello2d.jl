# Swap Monte Carlo of the 2D continuously polydisperse soft-sphere mixture of Ninarello,
# Berthier and Coslovich (PRX 7, 021039, 2017).
#
# Usage, from the repository root:
#
#     julia --project examples/ninarello2d.jl [kT] [N] [nsweeps]
#
# Defaults: kT = 0.1, N = 1000 and nsweeps = 20000 production sweeps (about 25 s). The
# output goes to `examples/output/ninarello2d_N=<N>_kT=<kT>/`: a thermo file, an extended
# XYZ trajectory (readable by OVITO, ASE and MolecularDynamics.jl), the final
# configuration and a checkpoint to continue the run.

using SwapMonteCarlo
using Printf: @printf
using Random: Xoshiro
using Statistics: mean

"""
    main(; kT=0.1, N=1000, nsweeps=20_000, nequil=5_000, seed=2026)

Equilibrate the mixture at temperature `kT` with step size tuning, then run `nsweeps`
production sweeps with fixed step sizes and write the output files.
"""
function main(; kT=0.1, N=1000, nsweeps=20_000, nequil=5_000, seed=2026)
    rng = Xoshiro(seed)

    # Diameters with P(σ) ∝ σ⁻³ on [0.73, 1.62] (mean diameter ≈ 1) at number density 1.
    σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N)
    box = box_for_density(Val(2), N, 1.0)
    x = lattice_positions(box, N)
    # SS12 soft spheres, C² smooth at the cutoff 1.25 σij, non-additivity 0.2.
    sys = System(box, x, σ, PolySoftSphere(; xc=1.25, mixing=NonAdditive(0.2)))

    # 80 % displacements and 20 % swaps of the diameters of two random particles.
    sim = Simulation(sys, NVT(kT), Displacement(0.1) => 0.8, Swap() => 0.2; seed=seed)
    println(sim)
    @printf("polydispersity = %.4f\n", polydispersity(σ))

    outdir = joinpath(@__DIR__, "output", "ninarello2d_N=$(N)_kT=$(kT)")
    mkpath(outdir)

    # Equilibration: the maximum displacement is tuned toward 40 % acceptance.
    progress = ProgressPrinter(; schedule=Every(max(1, nequil ÷ 5)))
    equilibrate!(sim, nequil; observers=progress)
    @printf("tuned maximum displacement δ = %.4f\n", step_size(sim).displacement)

    # Production: step sizes are frozen, so the chain satisfies detailed balance.
    reset_stats!(sim)
    thermo = ThermoWriter(
        joinpath(outdir, "thermo.txt"),
        PotentialEnergy(),
        Pressure(),
        Acceptance(:displacement),
        Acceptance(:swap);
        schedule=Every(100),
    )
    traj = TrajectoryWriter(joinpath(outdir, "trajectory.xyz"); schedule=LogSchedule())
    series = TimeSeries(PotentialEnergy(), Pressure(); schedule=Every(100))
    progress = ProgressPrinter(; schedule=Every(max(1, nsweeps ÷ 5)))
    elapsed = @elapsed run!(sim, nsweeps; observers=(thermo, traj, series, progress))
    close(thermo)
    close(traj)

    write_extxyz(joinpath(outdir, "final.xyz"), sim.system; sweep=sim.sweep)
    save_checkpoint(joinpath(outdir, "checkpoint.jls"), sim)

    rates = acceptance_rate(sim)
    @printf("⟨U/N⟩ = %.6f, ⟨P⟩ = %.6f\n", mean(series["energy"]), mean(series["pressure"]))
    @printf("acceptance: displacement %.3f, swap %.3f\n", rates.displacement, rates.swap)
    @printf("%.0f ns per attempt\n", 1e9 * elapsed / (nsweeps * N))
    println("output written to ", outdir)
    return sim
end

kT = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.1
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
nsweeps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20_000
main(; kT=kT, N=N, nsweeps=nsweeps)
