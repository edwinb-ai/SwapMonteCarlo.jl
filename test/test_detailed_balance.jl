# Stationary distributions sampled by the moves: the pair distance of two particles under
# displacements, and the diameter permutations of three particles under swaps.

using SwapMonteCarlo: Displacement, Swap

"""
    db_simpson(f, a, b, n)

Composite Simpson rule for `∫_a^b f(x) dx` with an even number `n` of intervals.
"""
function db_simpson(f, a, b, n)
    h = (b - a) / n
    s = f(a) + f(b)
    for k in 1:(n - 1)
        s += (isodd(k) ? 4 : 2) * f(a + k * h)
    end
    return s * h / 3
end

"""
    db_chi2(observed, expected_probabilities) -> χ²

Pearson χ² statistic of the counts `observed` against the probabilities.
"""
function db_chi2(observed, probabilities)
    n = sum(observed)
    return sum((o - n * q)^2 / (n * q) for (o, q) in zip(observed, probabilities))
end

@testset "pair distance of two Gaussian cores" begin
    for D in (2, 3)
        rng = test_rng(1100 + D)
        pot = GaussianCore(; A=2.0, rc=3.0)
        β = 1.0
        L = 7.0
        box = CubicBox(Val(D), L)
        x = [SVector(ntuple(_ -> 1.0, Val(D))), SVector(ntuple(_ -> 2.0, Val(D)))]
        sys = System(box, x, [1.0, 1.0], pot)
        # With δ = L/2 a displacement is uniform over the periodic box.
        move = Displacement(L / 2)
        nsamples = 10^5
        thin = 3

        # Equal volume bins of the distance r < L/2, plus one bin for r ≥ L/2. The ball
        # r < L/2 lies inside the box, so the density of r there is S_D r^(D-1) e^{-βu}/Z.
        nbins = 19
        rmax = L / 2
        edges = [rmax * (k / nbins)^(1 / D) for k in 0:nbins]
        S = D == 2 ? 2π : 4π
        w(r) = r < pot.rc ? exp(-β * pot.A * exp(-r^2)) : 1.0
        g(r) = S * r^(D - 1) * w(r)
        ball(r) = S * r^D / D
        Z = L^D - ball(pot.rc) + db_simpson(g, 0.0, pot.rc, 2000)
        probabilities = [db_simpson(g, edges[k], edges[k + 1], 200) / Z for k in 1:nbins]
        push!(probabilities, 1 - sum(probabilities))
        @test minimum(probabilities) * nsamples > 100

        counts = zeros(Int, nbins + 1)
        for _ in 1:nsamples
            for _ in 1:thin
                SMC.attempt!(move, sys, (; kT=1 / β), rng)
            end
            dx = SMC.min_image(sys.box, sys.frac[1] - sys.frac[2])
            r = sqrt(dot(dx, dx))
            k = r < rmax ? clamp(searchsortedlast(edges, r), 1, nbins) : nbins + 1
            counts[k] += 1
        end
        χ2 = db_chi2(counts, probabilities)
        @test χ2 < chi2_crit(nbins)
        @test sys.energy ≈ total_energy(sys) atol = 1e-10
    end
end

@testset "diameter permutations of three soft spheres" begin
    rng = test_rng(1200)
    pot = PolySoftSphere(; mixing=NonAdditive(0.2))
    L = SVector(4.0, 4.0)
    box = OrthorhombicBox(L)
    # The diameters σ0 sit in the lowest energy arrangement.
    x = [SVector(2.0, 1.1), SVector(1.5, 1.9), SVector(1.0, 1.0)]
    σ0 = [0.8, 1.0, 1.3]
    sys = System(box, x, σ0, pot)
    kT = 0.7
    perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    energies = [brute_energy(L, x, σ0[q], pot) for q in perms]
    weights = exp.(-(energies .- minimum(energies)) ./ kT)
    probabilities = weights ./ sum(weights)
    nsamples = 10^5
    thin = 10
    @test minimum(probabilities) * nsamples > 100

    move = Swap()
    counts = zeros(Int, length(perms))
    for _ in 1:nsamples
        for _ in 1:thin
            SMC.attempt!(move, sys, (; kT=kT), rng)
        end
        k = findfirst(q -> σ0[q] == sys.diameters, perms)
        counts[k] += 1
    end
    χ2 = db_chi2(counts, probabilities)
    @test χ2 < chi2_crit(length(perms) - 1)
    @test 0.1 < SMC.acceptance_rate(move) < 0.9
    @test sys.energy ≈ total_energy(sys) rtol = 1e-10
end
