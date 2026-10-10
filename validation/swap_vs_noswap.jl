# Swap Monte Carlo vs displacement-only Monte Carlo for the Ninarello soft-sphere mixtures.
#
#     julia --project=validation validation/swap_vs_noswap.jl
#
# Model: `PolySoftSphere()` (SS12 with C² smoothing at 1.25 σij, non-additivity 0.2),
# diameters P(σ) ∝ σ⁻³ on [0.73, 1.62] (stratified), N = 500. 2D at ρ = 1 (Berthier et
# al., Nat. Commun. 10, 1508 (2019)); 3D at ρ = 1.02 (Ninarello, Berthier and Coslovich,
# PRX 7, 021039 (2017), where T_MCT ≈ 0.104 in 3D).
#
# 1. At a higher temperature where both dynamics equilibrate (2D: kT = 0.3,
#    3D: kT = 0.25), the equilibrium energies must agree, and the speedup of the
#    structural relaxation is measured from the self-overlap Q(t) (a = 0.3; cage-relative
#    displacements in 2D) and from the integrated autocorrelation time of U.
# 2. At a low temperature (2D: kT = 0.12, 3D: kT = 0.13), two independent swap runs from
#    different high-temperature configurations must give the same energy and relax
#    within the run (Q decays), while the displacement-only dynamics does not relax on
#    the same time scale.
#
# Moves: displacement/swap weights 0.8/0.2 with swaps, 1.0 without. Times are in sweeps
# (N attempts of any kind). Pass criteria: |ΔU| ≤ 4 √(SE₁² + SE₂²), the swap run longer
# than 20 τ_Q, and, at high temperature, also the displacement-only run longer than
# 20 τ_Q.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "swap_dynamics.jl"))

"""
    ss_run(rng, D, ρ, kT, swap, nequil, nsweeps; N=500) -> NamedTuple

Prepare a fluid of `N` soft spheres at density `ρ` (lattice start, melted at kT = 1 with
swaps), then equilibrate for `nequil` and run for `nsweeps` sweeps at `kT`, with or
without swaps. Returns the energy time series statistics, the self-overlap relaxation
time and the wall time per sweep.
"""
function ss_run(rng, D, ρ, kT, swap, nequil, nsweeps; N=500)
    σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N; stratified=true)
    box = box_for_density(Val(D), N, ρ)
    sys = System(box, lattice_positions(box, N), σ, PolySoftSphere())
    melt = Simulation(sys, NVT(1.0), Displacement(0.1) => 0.8, Swap() => 0.2; rng=rng)
    equilibrate!(melt, 2000)
    moves = swap ? (Displacement(0.1) => 0.8, Swap() => 0.2) : (Displacement(0.1) => 1.0,)
    sim = Simulation(sys, NVT(kT), moves...; rng=rng)
    equilibrate!(sim, nequil)
    reset_stats!(sim)
    every = max(10, nsweeps ÷ 4000)
    traj = Trajectory(sys, every)
    ts = TimeSeries(PotentialEnergy())
    rec = Callback(s -> record!(traj, s.system); schedule=Every(every))
    t = @elapsed run!(sim, nsweeps; observers=(ts, rec))
    (lags, Q) = self_overlap(traj, sys.box.lengths; cage=(D == 2))
    u = ts["energy"]
    h = length(u) ÷ 2
    return (;
        U=mean(u),
        dU=block_stderr(u),
        U1=mean(u[1:h]),
        U2=mean(u[(h + 1):end]),
        τU=integrated_time(u),
        τQ=relaxation_time(lags, Q),
        Qend=Q[end],
        tend=lags[end],
        acc=swap ? acceptance_rate(sim).swap : NaN,
        cost=t / nsweeps,
        nequil=nequil,
        nsweeps=nsweeps,
    )
end

"""
    row(label, kT, swap, r) -> Vector{String}

Table row for the result `r` of [`ss_run`](@ref).
"""
function row(label, kT, swap, r)
    return [
        label,
        @sprintf("%.2f", kT),
        swap ? "0.8 / 0.2" : "1.0 / 0",
        string(r.nequil),
        string(r.nsweeps),
        fmt_pm(r.U, r.dU; digits=4),
        @sprintf("%.4f / %.4f", r.U1, r.U2),
        @sprintf("%.0f", r.τU),
        if isfinite(r.τQ)
            @sprintf("%.3g", r.τQ)
        else
            @sprintf("> %d (Q = %.2f)", r.tend, r.Qend)
        end,
        isnan(r.acc) ? "—" : @sprintf("%.3f", r.acc),
        @sprintf("%.2f", 1e3 * r.cost),
    ]
end

function main()
    seed = getseed(34)
    t0 = time()
    rows = Vector{Vector{String}}()
    checks = Vector{Vector{String}}()
    allok = true
    # Each run is given as (equilibration sweeps, production sweeps): hi0/hi1 without/with
    # swaps at T_hi, lo1 the two swap replicas and lo0 the run without swaps at T_lo.
    setups = (
        (;
            D=2,
            ρ=1.0,
            Thi=0.3,
            hi0=(40_000, 400_000),
            hi1=(20_000, 200_000),
            Tlo=0.12,
            lo1=(40_000, 400_000),
            lo0=(20_000, 200_000),
        ),
        (;
            D=3,
            ρ=1.02,
            Thi=0.25,
            hi0=(20_000, 100_000),
            hi1=(10_000, 40_000),
            Tlo=0.13,
            lo1=(50_000, 120_000),
            lo0=(10_000, 50_000),
        ),
    )
    for (k, (; D, ρ, Thi, hi0, hi1, Tlo, lo1, lo0)) in enumerate(setups)
        base = seed + 100 * k
        # High temperature: both dynamics equilibrate.
        r0 = ss_run(StableRNG(base + 1), D, ρ, Thi, false, hi0...)
        r1 = ss_run(StableRNG(base + 2), D, ρ, Thi, true, hi1...)
        push!(rows, row("$(D)D", Thi, false, r0))
        push!(rows, row("$(D)D", Thi, true, r1))
        sd = sqrt(r0.dU^2 + r1.dU^2)
        ok =
            abs(r0.U - r1.U) <= 4 * sd &&
            r0.nsweeps >= 20 * r0.τQ &&
            r1.nsweeps >= 20 * r1.τQ
        allok &= ok
        push!(
            checks,
            [
                "$(D)D, kT = $(Thi): swap vs no swap",
                @sprintf("%+.4f (%.1f σ)", r1.U - r0.U, (r1.U - r0.U) / sd),
                @sprintf("%.1f", r0.τQ / r1.τQ),
                @sprintf("%.1f", r0.τU / r1.τU),
                passfail(ok),
            ],
        )
        println("$(D)D high T done: speedup τ_Q $(r0.τQ / r1.τQ)")
        # Low temperature: two swap replicas and one displacement-only run.
        a = ss_run(StableRNG(base + 3), D, ρ, Tlo, true, lo1...)
        b = ss_run(StableRNG(base + 4), D, ρ, Tlo, true, lo1...)
        c = ss_run(StableRNG(base + 5), D, ρ, Tlo, false, lo0...)
        push!(rows, row("$(D)D (replica 1)", Tlo, true, a))
        push!(rows, row("$(D)D (replica 2)", Tlo, true, b))
        push!(rows, row("$(D)D", Tlo, false, c))
        sd = sqrt(a.dU^2 + b.dU^2)
        ok = abs(a.U - b.U) <= 4 * sd && a.nsweeps >= 20 * a.τQ && b.nsweeps >= 20 * b.τQ
        allok &= ok
        τswap = (a.τQ + b.τQ) / 2
        push!(
            checks,
            [
                "$(D)D, kT = $(Tlo): swap replicas",
                @sprintf("%+.4f (%.1f σ)", b.U - a.U, (b.U - a.U) / sd),
                if isfinite(c.τQ)
                    @sprintf("%.1f", c.τQ / τswap)
                else
                    @sprintf("> %.0f", c.tend / τswap)
                end,
                @sprintf("%.1f", c.τU / ((a.τU + b.τU) / 2)),
                passfail(ok),
            ],
        )
        println("$(D)D low T done")
    end
    runtime = time() - t0
    intro = """
    Ninarello soft-sphere mixtures (`PolySoftSphere()`, non-additivity 0.2, P(σ) ∝ σ⁻³ on
    [0.73, 1.62]), N = 500, NVT. 2D at ρ = 1, 3D at ρ = 1.02. Every run starts from a
    lattice melted at kT = 1 and is equilibrated at the target temperature for the
    number of sweeps given in the "equil." column (the step size is tuned during
    equilibration, then frozen). τ_Q: time at which the self-overlap Q(t) (a = 0.3,
    cage-relative in 2D) falls below 1/e; τ_U: integrated autocorrelation time of U/N;
    both in sweeps. "U halves": means of the two halves of the production run (a drift
    signals a run that is not equilibrated). Cost: wall time per sweep.

    Overall: $(allok ? "all checks pass" : "SOME CHECKS FAIL").
    """
    t1 =
        "## Runs\n\n" * md_table(
            [
                "system",
                "kT",
                "disp / swap",
                "equil.",
                "sweeps",
                "U/N ± SE",
                "U halves",
                "τ_U",
                "τ_Q",
                "swap acc.",
                "ms / sweep",
            ],
            rows,
        )
    t2 =
        """
   ## Checks

   Pass: |ΔU| ≤ 4 √(SE₁² + SE₂²) and runs longer than 20 τ_Q (both runs at high
   temperature, the swap replicas at low temperature). Speedup: ratio of the relaxation
   times without and with swaps (at low temperature a lower bound when Q does not decay
   within the displacement-only run).

   """ * md_table(["comparison", "ΔU/N", "speedup τ_Q", "speedup τ_U", "check"], checks)
    notes = """
    ## Notes

    - The energy autocorrelation time is much shorter than the structural relaxation
      time and underestimates the speedup; τ_Q is the relevant measure.
    - In 2D the plain self-overlap decays through long-wavelength (Mermin–Wagner)
      fluctuations that are not structural relaxation, hence the cage-relative
      displacements (the mean displacement of the neighbors closer than 1.5 at the time
      origin is subtracted).
    - The displacement-only run at low temperature starts from a configuration
      equilibrated at kT = 1 and is out of equilibrium: its energy is higher than the
      swap one and drifts down during the run (aging, see "U halves"), so neither its
      energy nor its τ_Q is an equilibrium value, and the speedup at low temperature is
      only indicative (a lower bound when Q does not decay).
    - The speedup is modest at the higher temperatures (a few) and grows quickly on
      cooling, as reported by Ninarello et al.; at kT = 0.12 in 2D the
      displacement-only dynamics does not relax within 2·10⁵ sweeps, while the swap
      dynamics relaxes in about 2·10⁴.
    """
    write_results(
        "swap_vs_noswap",
        "Swap vs displacement-only Monte Carlo",
        intro,
        t1,
        t2,
        notes;
        seed=seed,
        runtime=runtime,
    )
    return allok
end

main()
