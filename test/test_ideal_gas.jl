# The ideal gas in the NPT ensemble: the volume distribution is known exactly.
#
# With non-interacting particles the NPT weight of a volume `V` is
#
#     p(V) ∝ V^N e^{-βPV},
#
# where `V^N` comes from integrating the positions over the box. This is a Gamma
# distribution with shape `N + 1` and scale `kT/P`, so that
#
#     ⟨V⟩ = (N + 1) kT / P,   ⟨ln V⟩ = ψ(N + 1) + ln(kT/P),   ⟨1/V⟩ = P / (N kT).
#
# `VolumeMove` proposes uniform steps in `ln V`, not in `V`. The density of `ln V` is
# `p(V) dV/d(ln V) = V p(V) ∝ V^(N+1) e^{-βPV}`: the extra factor `V` (the Jacobian of the
# change of variables) is why the acceptance contains `(N + 1) Δln V` rather than
# `N Δln V`. A sampler that forgot the Jacobian would give a Gamma(N) volume distribution,
# i.e. `⟨V⟩ = N kT/P`, which these tests detect easily.

"""
    ig_digamma_int(n)

Digamma function at a positive integer, `ψ(n) = H_{n-1} - γ`.
"""
function ig_digamma_int(n::Integer)
    return sum((1 / k for k in 1:(n - 1)); init=0.0) - Base.MathConstants.γ
end

"""
    ig_run(rng, D, N, kT, P, moves; nsweeps)

Equilibrate and run an ideal gas of `N` particles in `D` dimensions in the NPT ensemble
with the given moves (pairs `move => weight`). Returns the volume time series, recorded
every sweep.
"""
function ig_run(rng, D, N, kT, P, moves...; nsweeps)
    box = CubicBox(Val(D), (N * kT / P)^(1 / D))
    sys = System(box, random_positions(rng, box, N), ones(N), Ideal())
    sim = Simulation(sys, NPT(kT, P), moves...; rng=rng)
    equilibrate!(sim, max(1000 ÷ N, 100))
    ts = TimeSeries(Volume())
    run!(sim, nsweeps; observers=ts)
    return ts["volume"]
end

"""
    ig_check(v, N, kT, P; thin)

Check the volume samples `v` of an ideal gas of `N` particles at temperature `kT` and
pressure `P` against the exact Gamma(N + 1, kT/P) distribution: the means of `V`,
`ln V` and (for `N ≥ 2`) `1/V`, and a Kolmogorov–Smirnov test of `βPV` on samples thinned
by `thin` sweeps so that they are nearly independent.
"""
function ig_check(v, N, kT, P; thin)
    scale = kT / P
    # ⟨V⟩ = (N + 1) kT/P; standard deviation of V is √(N + 1) kT/P.
    Vref = (N + 1) * scale
    se = block_stderr(v)
    @test check_mean(mean(v), se, Vref; se_max=0.01 * Vref)
    # ⟨ln V⟩ = ψ(N + 1) + ln(kT/P).
    lnv = log.(v)
    @test check_mean(
        mean(lnv), block_stderr(lnv), ig_digamma_int(N + 1) + log(scale); se_max=0.01
    )
    # ⟨1/V⟩ = P/(N kT) for N ≥ 1, but for N = 1 the variance of 1/V diverges
    # (∫ V^{-2} V e^{-V} dV = ∞), so the sample mean has no reliable error bar.
    if N >= 2
        inv_v = inv.(v)
        ref = 1 / (N * scale)
        @test check_mean(mean(inv_v), block_stderr(inv_v), ref; se_max=0.02 * ref)
    end
    # Kolmogorov–Smirnov test of the full distribution of βPV ~ Gamma(N + 1, 1).
    samples = v[thin:thin:end] ./ scale
    @test length(samples) >= 2000
    @test ks_stat(samples, x -> gamma_cdf_int(N + 1, x)) < ks_crit(length(samples))
    return nothing
end

@testset "ideal gas NPT, volume moves only (D = $D, N = $N, kT = $kT, P = $P)" for (
    k, D, N, kT, P
) in (
    (1, 3, 1, 1.0, 1.0), (2, 3, 4, 1.0, 1.0), (3, 2, 4, 1.0, 1.0), (4, 3, 4, 1.5, 0.6)
)
    rng = test_rng(2100 + k)
    # A sweep is N attempts. Larger steps than the default bound decorrelate faster.
    v = ig_run(
        rng, D, N, kT, P, VolumeMove(0.5; bounds=(1e-3, 3.0)) => 1.0; nsweeps=240_000 ÷ N
    )
    ig_check(v, N, kT, P; thin=max(40 ÷ N, 10))
end

@testset "ideal gas NPT, displacement and volume moves (N = 4)" begin
    rng = test_rng(2110)
    N = 4
    # The displacements do not change the energy (always accepted) but are interleaved
    # with the volume moves, which checks that they do not bias the volume.
    v = ig_run(
        rng,
        3,
        N,
        1.0,
        1.0,
        Displacement(0.3) => 0.5,
        VolumeMove(0.5; bounds=(1e-3, 3.0)) => 0.5;
        nsweeps=120_000,
    )
    ig_check(v, N, 1.0, 1.0; thin=20)
end
