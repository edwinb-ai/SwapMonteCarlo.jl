# Energy and virial sums against brute force, early exit, allocations and type stability.

import SwapMonteCarlo: is_repulsive

"""
Polydisperse soft spheres, `u = (σij/r)^12 - xc^-12` for `r < xc σij`, with a mixing rule.
File-local stand-in for the package potentials.
"""
struct ESoftSphere{M} <: Potential
    mixing::M
    xc::Float64
end
ESoftSphere(mixing=SMC.NonAdditive(0.2)) = ESoftSphere(mixing, 1.25)
function pair_energy(p::ESoftSphere, r2, σi, σj)
    σ = SMC.mix(p.mixing, σi, σj)
    r2 < (p.xc * σ)^2 || return zero(r2)
    x6 = (σ^2 / r2)^3
    return oftype(r2, x6 * x6 - p.xc^-12)
end
function pair_virial(p::ESoftSphere, r2, σi, σj)
    σ = SMC.mix(p.mixing, σi, σj)
    r2 < (p.xc * σ)^2 || return zero(r2)
    x6 = (σ^2 / r2)^3
    return oftype(r2, 12 * x6 * x6)
end
cutoff(p::ESoftSphere, σi, σj) = p.xc * SMC.mix(p.mixing, σi, σj)
is_repulsive(::ESoftSphere) = true

"""
    brute_particle_energy(L, x, σ, pot, xi, σi, skips) -> U

Energy of a particle of diameter `σi` at the Cartesian position `xi` with all particles
except those in `skips`.
"""
function brute_particle_energy(L, x, σ, pot, xi, σi, skips)
    U = 0.0
    for j in eachindex(x)
        j in skips && continue
        d = min_image_brute(x[j] - xi, L)
        U += pair_energy(pot, dot(d, d), σi, σ[j])
    end
    return U
end

"""
    make_system(rng, D, N, ρ, pot; T=Float64)

Random polydisperse system at density `ρ` (overlaps are allowed for soft potentials).
"""
function make_system(rng, D, N, ρ, pot; T=Float64)
    box = SMC.box_for_density(Val(D), N, ρ)
    box = OrthorhombicBox(T.(box.lengths))
    x = SMC.lattice_positions(box, N)
    x = [xi + T(0.3) * (rand(rng, SVector{D,T}) .- T(0.5)) for xi in x]
    σ = T(0.8) .+ T(0.4) .* rand(rng, T, N)
    return System(box, x, σ, pot)
end

energy_potentials = Any[
    ("soft sphere", ESoftSphere()),
    ("Gaussian core", GaussianCore()),
    ("evaluate Gaussian", EvaluateGaussian(1.0)),
]

@testset "particle energies vs brute force" begin
    rng = test_rng(11)
    for D in (2, 3), (name, pot) in energy_potentials
        @testset "$(D)D $(name)" begin
            sys = make_system(rng, D, D == 2 ? 300 : 400, D == 2 ? 0.6 : 0.4, pot)
            L = sys.box.lengths
            x = positions(sys)
            σ = sys.diameters
            for i in rand(rng, 1:length(sys), 20)
                ref = brute_particle_energy(L, x, σ, pot, x[i], σ[i], (i,))
                @test SMC.particle_energy(sys, i) ≈ ref rtol = 1e-12 atol = 1e-12
            end
            # Virtual particles at random positions, with skips.
            for _ in 1:20
                s = rand(rng, SVector{D,Float64})
                σv = 0.8 + 0.4 * rand(rng)
                (i, j) = (rand(rng, 1:length(sys)), rand(rng, 0:length(sys)))
                ref = brute_particle_energy(
                    L, x, σ, pot, SMC.cartesian(sys.box, s), σv, (i, j)
                )
                e = SMC.particle_energy(sys, s, σv, i, j, Inf)
                @test e ≈ ref rtol = 1e-12 atol = 1e-12
            end
            # Total energy is the half sum of the particle energies.
            @test total_energy(sys) ≈ brute_energy(L, x, σ, pot) rtol = 1e-12
            @test total_energy(sys) ≈
                sum(SMC.particle_energy(sys, i) for i in 1:length(sys)) / 2 rtol = 1e-10
        end
    end
end

@testset "early exit" begin
    rng = test_rng(12)
    for D in (2, 3)
        sys = make_system(rng, D, 300, 1.0, ESoftSphere())
        for i in 1:20
            exact = SMC.particle_energy(sys, sys.frac[i], sys.diameters[i], i, 0, Inf)
            @test exact == SMC.particle_energy(sys, i)
            exact > 0 || continue
            emax = 0.3 * exact
            e = SMC.particle_energy(sys, sys.frac[i], sys.diameters[i], i, 0, emax)
            @test emax < e <= exact * (1 + 1e-12)
            # Below the threshold, the energy is exact.
            @test SMC.particle_energy(sys, sys.frac[i], sys.diameters[i], i, 0, exact) ==
                exact
        end
        U = total_energy(sys)
        @test U == total_energy(sys; emax=Inf)
        @test U == total_energy(sys; emax=2U)
        @test total_energy(sys; emax=0.5U) == Inf

        # Non-repulsive potentials ignore the threshold.
        sysg = make_system(rng, D, 200, 0.5, GaussianCore())
        exact = SMC.particle_energy(sysg, 1)
        @test SMC.particle_energy(sysg, sysg.frac[1], sysg.diameters[1], 1, 0, -1.0) ==
            exact
        @test total_energy(sysg; emax=-1.0) == total_energy(sysg)
    end
end

@testset "virial and pressure" begin
    rng = test_rng(13)
    # GaussianCore has no pair virial; Ideal checks the ideal gas term alone.
    for D in (2, 3), pot in (ESoftSphere(), EvaluateGaussian(1.0), Ideal())
        sys = make_system(rng, D, 250, 0.7, pot)
        L = sys.box.lengths
        x = positions(sys)
        W = brute_virial(L, x, sys.diameters, pot)
        @test SMC.total_virial(sys) ≈ W rtol = 1e-12 atol = 1e-12
        kT = 0.7
        V = SMC.volume(sys)
        @test pressure(sys, kT) ≈ length(sys) * kT / V + W / (D * V) rtol = 1e-12
    end
end

@testset "check_energy and sync_energy!" begin
    rng = test_rng(14)
    sys = make_system(rng, 2, 200, 0.8, ESoftSphere())
    U = total_energy(sys)
    @test sys.energy == U
    sys.energy += 1
    (running, recomputed) = SMC.check_energy(sys)
    @test running == U + 1
    @test recomputed == U
    @test SMC.sync_energy!(sys) == U
    @test sys.energy == U
end

@testset "allocations and inference" begin
    rng = test_rng(15)
    for D in (2, 3), T in (Float64, Float32)
        sys = make_system(rng, D, 200, 0.8, ESoftSphere(); T=T)
        s = rand(rng, SVector{D,T})
        @test allocs(SMC.particle_energy, sys, 5) == 0
        @test allocs(SMC.particle_energy, sys, s, T(1), 5, 7, T(Inf)) == 0
        @test allocs(SMC.total_energy, sys) == 0
        @test allocs(SMC.total_virial, sys) == 0
        @test allocs(SMC.cell_move!, sys.cells, 5, sys.frac[5]) == 0
        @test allocs(SMC.cells_valid, sys.cells, sys.box, sys.rc) == 0
        # A move to another cell (and back) does not allocate either.
        f = (cells, i, a, b) -> (SMC.cell_move!(cells, i, a); SMC.cell_move!(cells, i, b))
        @test allocs(f, sys.cells, 5, s, sys.frac[5]) == 0
        @test (@inferred SMC.particle_energy(sys, 5)) isa T
        @test (@inferred SMC.particle_energy(sys, s, T(1), 0, 0, T(Inf))) isa T
        @test (@inferred total_energy(sys)) isa T
        @test (@inferred SMC.total_virial(sys)) isa T
        @test (@inferred pressure(sys, 1.0)) isa T
        @test (@inferred SMC.cells_valid(sys.cells, sys.box, sys.rc)) isa Bool
    end
end

@testset "Float32 systems" begin
    rng = test_rng(16)
    for D in (2, 3), (name, pot) in energy_potentials
        sys = make_system(rng, D, 200, 0.6, pot; T=Float32)
        @test eltype(sys.frac) == SVector{D,Float32}
        @test sys.energy isa Float32
        x = [Float64.(xi) for xi in positions(sys)]
        L = Float64.(sys.box.lengths)
        σ = Float64.(sys.diameters)
        @test total_energy(sys) ≈ brute_energy(L, x, σ, pot) rtol = 1e-4
        @test SMC.particle_energy(sys, 3) ≈
            brute_particle_energy(L, x, σ, pot, x[3], σ[3], (3,)) rtol = 1e-4 atol = 1e-4
    end
end
