# Sanity checks of the statistics helpers of `helpers.jl` on synthetic data with known
# properties. The statistical physics tests rely on these helpers for their tolerances,
# so a helper that underestimates errors would make those tests meaningless.

"""
    sh_ar1(rng, n, φ)

AR(1) series `x_t = φ x_{t-1} + √(1 - φ²) ε_t` of length `n` with unit stationary variance,
started in the stationary distribution.
"""
function sh_ar1(rng, n, φ)
    x = zeros(n)
    x[1] = randn(rng)
    c = sqrt(1 - φ^2)
    for t in 2:n
        x[t] = φ * x[t - 1] + c * randn(rng)
    end
    return x
end

"""
    sh_ar1_stderr(n, φ)

Exact standard error of the mean of `n` consecutive values of the stationary AR(1)
series with unit variance and coefficient `φ`.
"""
function sh_ar1_stderr(n, φ)
    v = (1 + φ) / (1 - φ) - 2 * φ * (1 - φ^n) / (n * (1 - φ)^2)
    return sqrt(v / n)
end

"""
    sh_simpson(f, a, b, n)

Composite Simpson rule for `∫_a^b f(x) dx` with an even number `n` of intervals.
"""
function sh_simpson(f, a, b, n)
    h = (b - a) / n
    s = f(a) + f(b)
    for k in 1:(n - 1)
        s += (isodd(k) ? 4 : 2) * f(a + k * h)
    end
    return s * h / 3
end

@testset "block_stderr on AR(1) series" begin
    rng = test_rng(2000)
    n = 2^16
    φ = 0.9
    se = sh_ar1_stderr(n, φ)
    ratios = Float64[]
    naive = Float64[]
    for _ in 1:20
        x = sh_ar1(rng, n, φ)
        push!(ratios, block_stderr(x) / se)
        push!(naive, std(x) / sqrt(n) / se)
    end
    # A single estimate fluctuates by about 10 % (the last blocking levels have few
    # blocks) and the maximum over levels is biased upward by a few percent, so the
    # average over the replicas is checked tightly and the individual ones loosely.
    @test 0.92 <= mean(ratios) <= 1.15
    @test all(r -> 0.7 <= r <= 1.6, ratios)
    # The naive estimate ignores the correlations: it is too small by a factor
    # √((1 + φ)/(1 - φ)) ≈ 4.4, which the blocking method must recover.
    @test all(<(0.3), naive)
end

@testset "block_stderr on independent samples" begin
    rng = test_rng(2001)
    n = 2^15
    ratios = [block_stderr(randn(rng, n)) / (1 / sqrt(n)) for _ in 1:20]
    @test 0.92 <= mean(ratios) <= 1.15
    @test all(r -> 0.75 <= r <= 1.5, ratios)
end

@testset "check_mean" begin
    @test check_mean(1.0, 0.1, 1.3; k=4)
    @test !check_mean(1.0, 0.1, 1.5; k=4)
    @test check_mean(1.0, 0.1, 1.5; k=4, sys=0.1)
    # A noisy estimate must fail even when it agrees with the reference.
    @test !check_mean(1.0, 0.1, 1.0; se_max=0.05)
end

@testset "chi2_crit" begin
    # Upper 5 % quantiles of the χ² distribution (standard tables).
    for (k, q) in
        ((5, 11.0705), (10, 18.3070), (20, 31.4104), (50, 67.5048), (100, 124.342))
        @test isapprox(chi2_crit(k; z=1.6449), q; rtol=3e-3)
    end
    # For even k the χ² CDF is a Gamma CDF with integer shape k/2 and scale 2, so the
    # tail probability at the default critical value must be close to 1e-4
    # (Wilson–Hilferty is approximate, especially for few degrees of freedom).
    for k in (4, 10, 30, 100, 400)
        p = 1 - gamma_cdf_int(k ÷ 2, chi2_crit(k) / 2)
        @test 0.5e-4 <= p <= 2e-4
    end
    @test chi2_crit(20) > chi2_crit(20; z=1.6449) > 20
end

@testset "gamma_cdf_int" begin
    @test gamma_cdf_int(3, 0.0) == 0.0
    @test gamma_cdf_int(3, -1.0) == 0.0
    for x in (0.1, 1.0, 5.0)
        @test gamma_cdf_int(1, x) ≈ 1 - exp(-x)
    end
    for k in (1, 2, 5, 17), x in (0.3, 1.0, 4.5, 20.0)
        gpdf(t) = t^(k - 1) * exp(-t) / factorial(k - 1)
        @test isapprox(gamma_cdf_int(k, x), sh_simpson(gpdf, 0.0, x, 2000); atol=1e-9)
    end
    @test gamma_cdf_int(5, 1e3) ≈ 1.0
end

@testset "ks_stat and ks_crit" begin
    # One sample at 1/2 against the uniform CDF: the empirical CDF jumps from 0 to 1.
    @test ks_stat([0.5], identity) ≈ 0.5
    @test ks_stat([0.25, 0.75], identity) ≈ 0.25
    rng = test_rng(2002)
    n = 10^4
    # √n D_n converges to the Kolmogorov distribution, whose mean is √(π/2) ln 2.
    d = [sqrt(n) * ks_stat(rand(rng, n), identity) for _ in 1:100]
    @test isapprox(mean(d), sqrt(π / 2) * log(2); rtol=0.1)
    @test count(>(sqrt(n) * ks_crit(n; α=0.05)), d) <= 15
    @test all(<(sqrt(n) * ks_crit(n)), d)
    # A slightly wrong distribution (u^0.9 instead of u) is detected.
    @test ks_stat(rand(rng, n) .^ 0.9, identity) > ks_crit(n)
    # Gamma samples (sums of exponentials) against gamma_cdf_int.
    g = [sum(Random.randexp(rng) for _ in 1:3) for _ in 1:n]
    @test ks_stat(g, x -> gamma_cdf_int(3, x)) < ks_crit(n)
    @test ks_stat(g, x -> gamma_cdf_int(4, x)) > ks_crit(n)
end
