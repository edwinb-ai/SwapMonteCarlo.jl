# Particle size distributions and sampling of diameters.

@testset "InversePowerLaw analytic moments" begin
    (a, b) = (0.73, 1.62)
    d = InversePowerLaw(a, b)
    @test d isa SMC.SizeDistribution
    @test SMC.mean_diameter(d) ≈ 2a * b / (a + b) rtol = 1e-14
    @test SMC.mean_diameter(d) ≈ 1.006468085106383 rtol = 1e-14
    @test SMC.diameter_moment(d, 2) ≈ 2a^2 * b^2 * log(b / a) / (b^2 - a^2) rtol = 1e-14
    @test polydispersity(d) ≈ 0.2289100960577779 rtol = 1e-12
    @test SMC.quantile(d, 0.0) == a
    @test SMC.quantile(d, 1.0) ≈ b
    @test SMC.quantile(d, 0.5) ≈ 1 / sqrt(0.5 / a^2 + 0.5 / b^2)
    @test_throws DomainError SMC.quantile(d, 1.5)
    @test_throws DomainError SMC.quantile(d, -0.1)
    # The generic exponent branch agrees with the closed form at p = 3.
    d3 = InversePowerLaw(a, b; exponent=3 + 1e-12)
    @test SMC.quantile(d3, 0.3) ≈ SMC.quantile(d, 0.3) rtol = 1e-9
end

@testset "random sampling: $(name)" for (name, d) in [
    ("InversePowerLaw p = 3", InversePowerLaw(0.73, 1.62)),
    ("InversePowerLaw p = 2.5", InversePowerLaw(0.8, 1.4; exponent=2.5)),
    ("InversePowerLaw p = 0", InversePowerLaw(0.8, 1.4; exponent=0)),
    ("UniformSizes", UniformSizes(0.9, 1.3)),
]
    n = 10^6
    σs = sample_diameters(test_rng(20), d, n)
    @test σs isa Vector{Float64}
    @test length(σs) == n
    @test all(σ -> d.sigma_min <= σ <= d.sigma_max, σs)
    for k in (1, 2, 3)
        x = σs .^ k
        @test abs(mean(x) - SMC.diameter_moment(d, k)) < 4 * std(x) / sqrt(n)
    end
    @test polydispersity(σs) ≈ polydispersity(d) rtol = 5e-3
    # The empirical distribution matches the inverse of the quantile function.
    grid = sort([SMC.quantile(d, u) for u in 0:1e-4:1])
    cdf(σ) = searchsortedlast(grid, σ) / length(grid)
    @test ks_stat(σs[1:10_000], cdf) < ks_crit(10_000) + 1e-4
end

@testset "stratified sampling: $(name)" for (name, d) in [
    ("InversePowerLaw", InversePowerLaw(0.73, 1.62)),
    ("UniformSizes", UniformSizes(0.9, 1.3)),
]
    for N in (100, 1000)
        σs = sample_diameters(test_rng(21), d, N; stratified=true)
        @test length(σs) == N
        qs = [SMC.quantile(d, (k - 0.5) / N) for k in 1:N]
        @test sort(σs) == sort(qs)
        # Shuffled, not sorted.
        @test !issorted(σs)
        for k in (1, 2)
            @test abs(sum(σs .^ k) / N - SMC.diameter_moment(d, k)) < 1 / N^2
        end
        @test polydispersity(σs) ≈ polydispersity(d) rtol = 10 / N^2
    end
end

@testset "UniformSizes" begin
    d = UniformSizes(0.9, 1.3)
    @test SMC.quantile(d, 0.25) ≈ 1.0
    @test SMC.mean_diameter(d) ≈ 1.1
    @test polydispersity(d) ≈ 0.4 / (sqrt(12) * 1.1)
end

@testset "Bidisperse: fraction = $f, stratified = $strat" for f in
                                                              (0.0, 0.3, 0.5, 0.8, 1.0),
    strat in (false, true)

    d = Bidisperse(1.0, 1.4, f)
    for N in (1, 7, 100, 1001)
        σs = sample_diameters(test_rng(22), d, N; stratified=strat)
        @test length(σs) == N
        @test count(==(1.0), σs) == round(Int, N * f)
        @test count(==(1.4), σs) == N - round(Int, N * f)
    end
    if 0 < f < 1
        σs = sample_diameters(test_rng(23), d, 1000; stratified=strat)
        @test !issorted(σs) && !issorted(σs; rev=true)
    end
    @test SMC.quantile(d, 0.0) == (f > 0 ? 1.0 : 1.4)
    @test SMC.mean_diameter(d) ≈ f * 1.0 + (1 - f) * 1.4
end

@testset "Monodisperse" begin
    d = Monodisperse(1.2)
    @test sample_diameters(test_rng(24), d, 5) == fill(1.2, 5)
    @test sample_diameters(test_rng(24), d, 5; stratified=true) == fill(1.2, 5)
    @test SMC.quantile(d, 0.3) == 1.2
    @test polydispersity(fill(1.2, 10)) < 1e-14
    @test polydispersity(d) == 0
end

@testset "reproducibility: $(name)" for (name, d) in [
    ("Monodisperse", Monodisperse(1.0)),
    ("Bidisperse", Bidisperse(1.0, 1.4, 0.5)),
    ("InversePowerLaw", InversePowerLaw(0.73, 1.62)),
    ("UniformSizes", UniformSizes(0.9, 1.3)),
]
    for stratified in (false, true)
        a = sample_diameters(StableRNG(42), d, 500; stratified)
        b = sample_diameters(StableRNG(42), d, 500; stratified)
        @test a == b
    end
    @test isempty(sample_diameters(StableRNG(42), d, 0))
    @test_throws ArgumentError sample_diameters(StableRNG(42), d, -1)
end

@testset "polydispersity" begin
    @test polydispersity([1.0, 3.0]) ≈ 0.5
    @test polydispersity([2.0]) == 0
    @test polydispersity([1, 2, 3]) ≈ sqrt(2 / 3) / 2
    @test_throws ArgumentError polydispersity(Float64[])
end

@testset "invalid inputs" begin
    @test_throws ArgumentError Monodisperse(0)
    @test_throws ArgumentError Monodisperse(-1.0)
    @test_throws ArgumentError Monodisperse(Inf)
    @test_throws ArgumentError Bidisperse(1.0, 1.4, 1.5)
    @test_throws ArgumentError Bidisperse(1.0, 1.4, -0.1)
    @test_throws ArgumentError Bidisperse(0.0, 1.4, 0.5)
    @test_throws ArgumentError Bidisperse(1.0, NaN, 0.5)
    @test_throws ArgumentError InversePowerLaw(1.0, 0.5)
    @test_throws ArgumentError InversePowerLaw(1.0, 1.0)
    @test_throws ArgumentError InversePowerLaw(0.0, 1.0)
    @test_throws ArgumentError InversePowerLaw(0.5, 1.0; exponent=1)
    @test_throws ArgumentError InversePowerLaw(0.5, Inf)
    @test_throws ArgumentError UniformSizes(2.0, 1.0)
    @test_throws ArgumentError UniformSizes(-1.0, 1.0)
end
