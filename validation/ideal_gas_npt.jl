# Ideal gas in the NPT ensemble: the full volume distribution against the exact result.
#
#     julia --project=validation validation/ideal_gas_npt.jl
#
# For N non-interacting particles the NPT weight of the volume is `V^N e^{-βPV}`, so
# `βPV ~ Gamma(N + 1, 1)`: ⟨V⟩ = (N + 1) kT/P, Var V = (N + 1)(kT/P)² and
# ⟨ln V⟩ = ψ(N + 1) + ln(kT/P). `VolumeMove` samples `ln V`, whose density carries the
# extra Jacobian factor V, hence the `(N + 1) Δln V` term of its acceptance; without it
# the distribution would be Gamma(N, 1).
#
# The full distribution is tested with a Kolmogorov–Smirnov test on samples thinned by
# about six integrated autocorrelation times (so they are nearly independent).
#
# Pass criteria: the means of V, (V - ⟨V⟩)² and ln V within 4 SE of the exact values,
# and a KS p-value above 1e-3.

include(joinpath(@__DIR__, "common.jl"))

"""
Non-interacting particles: zero pair energy and zero cutoff.
"""
struct IdealGas <: SwapMonteCarlo.Potential end
SwapMonteCarlo.pair_energy(::IdealGas, r2, σi, σj) = zero(r2)
SwapMonteCarlo.pair_virial(::IdealGas, r2, σi, σj) = zero(r2)
SwapMonteCarlo.cutoff(::IdealGas, σi, σj) = 0.0

"""
    digamma_int(n)

Digamma function at a positive integer, `ψ(n) = H_{n-1} - γ`.
"""
digamma_int(n::Integer) = sum((1 / k for k in 1:(n - 1)); init=0.0) - Base.MathConstants.γ

"""
    ideal_gas_volumes(rng, N, kT, P, moves...; attempts) -> Vector{Float64}

Volume after every sweep of an NPT run of `N` ideal-gas particles in 3D with about
`attempts` move attempts in total.
"""
function ideal_gas_volumes(rng, N, kT, P, moves...; attempts)
    box = CubicBox(Val(3), (N * kT / P)^(1 / 3))
    sys = System(box, random_positions(rng, box, N), ones(N), IdealGas())
    sim = Simulation(sys, NPT(kT, P), moves...; rng=rng)
    equilibrate!(sim, max(10_000 ÷ N, 100))
    ts = TimeSeries(Volume())
    run!(sim, attempts ÷ N; observers=ts)
    return ts["volume"]
end

function main()
    seed = getseed(35)
    t0 = time()
    attempts = 40_000_000
    cases = (
        (1, 1.0, 1.0, false),
        (2, 1.0, 1.0, false),
        (4, 1.0, 1.0, false),
        (16, 1.0, 1.0, false),
        (64, 1.0, 1.0, false),
        (16, 2.0, 0.5, false),
        (16, 1.0, 1.0, true),
    )
    rows = Vector{Vector{String}}()
    allok = true
    for (k, (N, kT, P, with_disp)) in enumerate(cases)
        rng = StableRNG(seed + k)
        vmove = VolumeMove(0.5; bounds=(1e-4, 3.0))
        moves = with_disp ? (Displacement(0.5) => 0.5, vmove => 0.5) : (vmove => 1.0,)
        v = ideal_gas_volumes(rng, N, kT, P, moves...; attempts=attempts)
        s = kT / P
        # Exact moments of Gamma(N + 1, s).
        Vref = (N + 1) * s
        varref = (N + 1) * s^2
        lnref = digamma_int(N + 1) + log(s)
        dv2 = (v .- Vref) .^ 2
        lnv = log.(v)
        zV = (mean(v) - Vref) / block_stderr(v)
        zvar = (mean(dv2) - varref) / block_stderr(dv2)
        zln = (mean(lnv) - lnref) / block_stderr(lnv)
        τ = integrated_time(v)
        thin = max(1, ceil(Int, 6 * τ))
        samples = v[thin:thin:end] ./ s
        d = ks_stat(samples, x -> gamma_cdf_int(N + 1, x))
        p = ks_pvalue(d, length(samples))
        ok = all(abs.((zV, zvar, zln)) .<= 4) && p > 1e-3
        allok &= ok
        push!(
            rows,
            [
                string(N),
                @sprintf("%g / %g", kT, P),
                with_disp ? "disp + vol" : "vol",
                @sprintf("%.4f (%.4f)", mean(v), Vref),
                @sprintf("%+.1f", zV),
                @sprintf("%.4f (%.4f)", mean(dv2), varref),
                @sprintf("%+.1f", zvar),
                @sprintf("%.4f (%.4f)", mean(lnv), lnref),
                @sprintf("%+.1f", zln),
                @sprintf("%.1f", τ),
                string(length(samples)),
                @sprintf("%.4f", d),
                @sprintf("%.3f", p),
                passfail(ok),
            ],
        )
        println("N = $(N), kT = $(kT), P = $(P) done: KS p = $(p)")
    end
    runtime = time() - t0
    intro = """
    Ideal gas (zero pair energy, zero cutoff) in a cubic 3D box, NPT with `VolumeMove`
    (step tuned toward 30 % acceptance in `ln V` with an upper bound of 3), and in one
    case also displacements (which are always accepted). About $(attempts) move attempts
    per case (`$(attempts) ÷ N` sweeps), volume recorded every sweep. Exact values in
    parentheses; z: deviation in blocked standard errors; τ: integrated autocorrelation
    time of V in sweeps; KS: Kolmogorov–Smirnov statistic and p-value of βPV against
    Gamma(N + 1, 1) on samples thinned by ⌈6τ⌉ sweeps.

    Overall: $(allok ? "all checks pass" : "SOME CHECKS FAIL").
    """
    t1 =
        "## Volume distribution vs Gamma(N + 1, kT/P)\n\n" * md_table(
            [
                "N",
                "kT / P",
                "moves",
                "⟨V⟩ (exact)",
                "z",
                "⟨(V - V̄)²⟩ (exact)",
                "z",
                "⟨ln V⟩ (exact)",
                "z",
                "τ_V",
                "KS samples",
                "KS D",
                "KS p",
                "check",
            ],
            rows,
        )
    write_results(
        "ideal_gas_npt",
        "Ideal gas in the NPT ensemble",
        intro,
        t1;
        seed=seed,
        runtime=runtime,
    )
    return allok
end

main()
