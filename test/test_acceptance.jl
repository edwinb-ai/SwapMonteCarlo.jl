# Empirical acceptance probabilities of single moves from a fixed state against exact
# values obtained by quadrature or enumeration.

using SwapMonteCarlo: Displacement, Swap, VolumeMove

"""
    acc_tolerance(p, M)

Allowed deviation of an empirical acceptance ratio from the exact `p` after `M`
independent attempts: four standard deviations plus a small allowance for the
quadrature error.
"""
acc_tolerance(p, M) = 4 * sqrt(p * (1 - p) / M) + 1e-3

"""
    acc_simpson(f, a, b, n)

Composite Simpson rule for `∫_a^b f(x) dx` with an even number `n` of intervals.
"""
function acc_simpson(f, a, b, n)
    h = (b - a) / n
    s = f(a) + f(b)
    for k in 1:(n - 1)
        s += (isodd(k) ? 4 : 2) * f(a + k * h)
    end
    return s * h / 3
end

const ACC_M = 2 * 10^5

@testset "displacement acceptance (two Gaussian cores)" begin
    rng = test_rng(1000)
    pot = GaussianCore(; A=2.0, rc=3.0)
    L = 7.0
    box = CubicBox(Val(2), L)
    x = [SVector(2.0, 3.0), SVector(3.0, 3.4)]
    sys = System(box, x, [1.0, 1.0], pot)
    kT = 0.5
    δ = 1.5
    # Exact acceptance: average over the moved particle and over the displacement square
    # of min(1, e^{-βΔu}), by the midpoint rule.
    Lv = SVector(L, L)
    u_old = pair_energy(pot, sum(abs2, min_image_brute(x[1] - x[2], Lv)), 1.0, 1.0)
    n = 400
    h = 2δ / n
    p = 0.0
    for (i, j) in ((1, 2), (2, 1)), a in 1:n, b in 1:n
        d = SVector(-δ + (a - 0.5) * h, -δ + (b - 0.5) * h)
        r2 = sum(abs2, min_image_brute(x[i] + d - x[j], Lv))
        p += min(1.0, exp(-(pair_energy(pot, r2, 1.0, 1.0) - u_old) / kT))
    end
    p /= 2 * n^2
    @test 0.2 < p < 0.9

    move = Displacement(δ)
    frac0 = copy(sys.frac)
    E0 = sys.energy
    accepted = 0
    for _ in 1:ACC_M
        accepted += SMC.attempt!(move, sys, (; kT=kT), rng)
        # Restore the initial state.
        for k in 1:2
            sys.frac[k] = frac0[k]
            SMC.cell_move!(sys.cells, k, frac0[k])
        end
        sys.energy = E0
    end
    @test abs(accepted / ACC_M - p) <= acc_tolerance(p, ACC_M)
    @test SMC.move_stats(move).attempts == ACC_M
    @test SMC.move_stats(move).accepts == accepted
end

@testset "swap acceptance (three soft spheres)" begin
    rng = test_rng(1001)
    pot = PolySoftSphere(; mixing=NonAdditive(0.2))
    box = CubicBox(Val(2), 4.0)
    # The diameters σ0 sit in the lowest energy arrangement.
    x = [SVector(2.0, 1.1), SVector(1.5, 1.9), SVector(1.0, 1.0)]
    σ0 = [0.8, 1.0, 1.3]
    sys = System(box, x, σ0, pot)
    kT = 0.5
    L = SVector(4.0, 4.0)
    E0 = brute_energy(L, x, σ0, pot)
    @test sys.energy ≈ E0 rtol = 1e-12
    # Exact acceptance: average over the three pairs of min(1, e^{-βΔE}).
    p = 0.0
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        σ = copy(σ0)
        σ[i], σ[j] = σ[j], σ[i]
        p += min(1.0, exp(-(brute_energy(L, x, σ, pot) - E0) / kT)) / 3
    end
    @test 0.2 < p < 0.9

    move = Swap()
    accepted = 0
    for _ in 1:ACC_M
        accepted += SMC.attempt!(move, sys, (; kT=kT), rng)
        sys.diameters .= σ0
        sys.energy = E0
    end
    @test abs(accepted / ACC_M - p) <= acc_tolerance(p, ACC_M)
end

@testset "volume acceptance (ideal gas)" begin
    rng = test_rng(1002)
    N = 4
    box0 = CubicBox(Val(2), 2.0)
    sys = System(box0, lattice_positions(box0, N), ones(N), Ideal())
    kT = 1.0
    P = 1.5
    Δ = 0.5
    V0 = volume(box0)
    # Exact acceptance: average over Δln V = x ∈ [-Δ, Δ] of
    # min(1, exp((N + 1) x - βP V0 (e^x - 1))).
    f(x) = min(1.0, exp((N + 1) * x - P * V0 * expm1(x) / kT))
    p = acc_simpson(f, -Δ, Δ, 20_000) / (2Δ)
    @test 0.2 < p < 0.9

    move = VolumeMove(Δ)
    accepted = 0
    for _ in 1:ACC_M
        accepted += SMC.attempt!(move, sys, (; kT=kT, pressure=P), rng)
        sys.box = box0
        SMC.cells_valid(sys.cells, sys.box, sys.rc) || SMC.rebuild_cells!(sys)
        sys.energy = 0.0
    end
    @test abs(accepted / ACC_M - p) <= acc_tolerance(p, ACC_M)
end
