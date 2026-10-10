# Lennard-Jones fluid against the NIST Standard Reference Simulation Website (SRSW).
#
#     julia --project=validation validation/lj_nist.jl
#
# NVT Monte Carlo of N = 500 Lennard-Jones particles truncated (not shifted) at
# r_c = 3σ with the standard long-range corrections, as in the NIST table
# `LJ_NIST_MC_RC3_LRC` (see `references.jl`). The potential energy per particle and the
# virial pressure (both including the tail corrections) are compared with NIST at a few
# vapor and liquid state points of the T* = 0.85 and 0.90 isotherms.
#
# Pressure convention. A potential truncated without shift jumps by -u(r_c) at the
# cutoff; the pressure *of the truncated model* therefore contains an impulsive term
# ΔP_imp = (2π/3) ρ² r_c³ u(r_c) g(r_c) that the pair virial inside the cutoff misses.
# NIST states that its pressure was "calculated using the virial expression" with
# "standard long range corrections" and that "the above pressure correction does not
# account for impulsive effects at r_c" (https://mmlapps.nist.gov/srs/LJ_PURE/index.htm).
# The standard tail correction of the pressure, P_lrc = (16π/3) ρ² [(2/3) r_c⁻⁹ - r_c⁻³],
# is the virial of the *full* potential beyond r_c (with g = 1); integrating it by parts
# shows that it already equals ΔP_imp(g = 1) plus the volume derivative of the energy
# correction, so "virial inside r_c + P_lrc" estimates the pressure of the full
# Lennard-Jones fluid, and no separate impulsive term must be added. This is what
# `Pressure()` computes (`pressure_lrc` of `LennardJones` uses the same standard
# formula). The table also shows what adding ΔP_imp (with g(r_c) = 1) would give, which
# is inconsistent with the NIST data at liquid densities.
#
# Pass criterion: |x - x_NIST| ≤ 4 √(SE² + σ_NIST²) for U/N and P.

include(joinpath(@__DIR__, "common.jl"))

"""
    lj_state(rng, T, ρ; N, nequil, nsweeps, every=10) -> NamedTuple

NVT run of truncated Lennard-Jones particles (r_c = 3σ, no shift, tail corrections) at
temperature `T` and density `ρ`, started from a simple cubic lattice. Returns the means
and blocked standard errors of `U/N` and of the virial pressure.
"""
function lj_state(rng, T, ρ; N, nequil, nsweeps, every=10)
    pot = LennardJones(; xc=3.0, shift=NoShift(), tail_correction=true)
    box = box_for_density(Val(3), N, ρ)
    sys = System(box, lattice_positions(box, N), ones(N), pot)
    sim = Simulation(sys, NVT(T), Displacement(0.1) => 1.0; rng=rng)
    equilibrate!(sim, nequil)
    reset_stats!(sim)
    ts = TimeSeries(PotentialEnergy(), Pressure(); schedule=Every(every))
    run!(sim, nsweeps; observers=ts)
    u = ts["energy"]
    p = ts["pressure"]
    return (;
        U=mean(u),
        dU=block_stderr(u),
        P=mean(p),
        dP=block_stderr(p),
        acc=acceptance_rate(sim).displacement,
        δ=step_size(sim).displacement,
    )
end

"""
    impulsive_pressure(ρ; xc=3.0)

Impulsive pressure term `(2π/3) ρ² r_c³ u(r_c)` of the Lennard-Jones potential truncated
without shift at `r_c = xc σ`, assuming `g(r_c) = 1`.
"""
function impulsive_pressure(ρ; xc=3.0)
    uc = 4 * (xc^-12 - xc^-6)
    return 2π / 3 * ρ^2 * xc^3 * uc
end

function main()
    seed = getseed(33)
    t0 = time()
    N = 500
    states = (
        (0.85, 5.00e-3),
        (0.85, 0.82),
        (0.85, 0.86),
        (0.90, 9.00e-3),
        (0.90, 0.78),
        (0.90, 0.90),
    )
    rows = Vector{Vector{String}}()
    imp_rows = Vector{Vector{String}}()
    allok = true
    for (k, (T, ρ)) in enumerate(states)
        ref = only(r for r in LJ_NIST_MC_RC3_LRC if r.T == T && r.ρ == ρ)
        liquid = ρ > 0.5
        nequil = liquid ? 5_000 : 10_000
        nsweeps = liquid ? 40_000 : 100_000
        rng = StableRNG(seed + k)
        r = lj_state(rng, T, ρ; N=N, nequil=nequil, nsweeps=nsweeps)
        sU = sqrt(r.dU^2 + ref.dU^2)
        sP = sqrt(r.dP^2 + ref.dP^2)
        okU = abs(r.U - ref.U) <= 4 * sU
        okP = abs(r.P - ref.P) <= 4 * sP
        allok &= okU & okP
        digits = liquid ? 4 : 6
        push!(
            rows,
            [
                @sprintf("%.2f", T),
                @sprintf("%g", ρ),
                @sprintf("%d", nsweeps),
                fmt_pm(r.U, r.dU; digits=digits),
                fmt_pm(ref.U, ref.dU; digits=digits),
                @sprintf("%+.1f", (r.U - ref.U) / sU),
                fmt_pm(r.P, r.dP; digits=digits),
                fmt_pm(ref.P, ref.dP; digits=digits),
                @sprintf("%+.1f", (r.P - ref.P) / sP),
                @sprintf("%.2f", r.acc),
                passfail(okU & okP),
            ],
        )
        Δimp = impulsive_pressure(ρ)
        push!(
            imp_rows,
            [
                @sprintf("%.2f", T),
                @sprintf("%g", ρ),
                @sprintf("%.4g", Δimp),
                fmt_pm(r.P + Δimp, r.dP; digits=digits),
                @sprintf("%+.1f", (r.P + Δimp - ref.P) / sP),
            ],
        )
        println(
            "T = $(T), ρ = $(ρ) done: U = $(fmt_pm(r.U, r.dU; digits=digits)), " *
            "P = $(fmt_pm(r.P, r.dP; digits=digits))",
        )
    end
    runtime = time() - t0
    intro = """
    Lennard-Jones, `LennardJones(xc=3.0, shift=NoShift(), tail_correction=true)`, N = 500,
    NVT with displacements only (step tuned to 40 % acceptance during equilibration, then
    frozen). Simple cubic lattice start; 5000 (liquid) or 10000 (vapor) equilibration
    sweeps; U/N and P recorded every 10 sweeps during the production. NIST: 5·10⁵
    production sweeps (2.5·10⁸ trials), uncertainties from block averages. z: difference
    in units of √(SE² + σ_NIST²).

    Overall: $(allok ? "all checks pass" : "SOME CHECKS FAIL").
    """
    t1 =
        "## U/N and P vs NIST (pass: |z| ≤ 4 for both)\n\n" * md_table(
            [
                "T*",
                "ρ*",
                "sweeps",
                "U/N ± SE",
                "U/N NIST",
                "z_U",
                "P ± SE",
                "P NIST",
                "z_P",
                "acc.",
                "check",
            ],
            rows,
        )
    t2 =
        """
   ## Impulsive correction (not part of the check)

   `P + ΔP_imp` adds the impulsive term of the truncated potential, `ΔP_imp = (2π/3) ρ²
   r_c³ u(r_c)` with g(r_c) = 1, to the measured pressure (which already includes the
   standard tail correction). At liquid densities it is far outside the NIST
   uncertainty, which confirms that the NIST pressures are "virial + standard tail
   correction" without an impulsive term, the convention of `Pressure()`.

   """ * md_table(["T*", "ρ*", "ΔP_imp", "P + ΔP_imp", "z vs NIST"], imp_rows)
    write_results(
        "lj_nist",
        "Lennard-Jones vs NIST SRSW (r_c = 3σ, tail corrections)",
        intro,
        t1,
        t2;
        seed=seed,
        runtime=runtime,
    )
    return allok
end

main()
