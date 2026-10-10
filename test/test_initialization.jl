# System construction and validation, accessors and initial configurations.

import SwapMonteCarlo: is_repulsive, is_hard

"""Hard spheres with additive diameters (file-local stand-in for `HardSphere`)."""
struct IHardSphere <: Potential end
function pair_energy(::IHardSphere, r2, σi, σj)
    σ = (σi + σj) / 2
    return r2 < σ^2 ? oftype(r2, Inf) : zero(r2)
end
cutoff(::IHardSphere, σi, σj) = (σi + σj) / 2
is_repulsive(::IHardSphere) = true
is_hard(::IHardSphere) = true

"""A potential whose pair energy is not symmetric in the diameters."""
struct AsymmetricStep <: Potential end
pair_energy(::AsymmetricStep, r2, σi, σj) = r2 < 1 ? σi : zero(r2)
cutoff(::AsymmetricStep, σi, σj) = 1.0

"""
    min_distance(L, x) -> d

Smallest minimum-image distance between two of the Cartesian positions `x`.
"""
function min_distance(L, x)
    d2 = Inf
    for i in eachindex(x), j in (i + 1):lastindex(x)
        d = min_image_brute(x[i] - x[j], L)
        d2 = min(d2, dot(d, d))
    end
    return sqrt(d2)
end

"""
    no_overlaps(L, x, σ) -> Bool

Whether no two spheres overlap with additive diameters.
"""
function no_overlaps(L, x, σ)
    for i in eachindex(x), j in (i + 1):lastindex(x)
        d = min_image_brute(x[i] - x[j], L)
        dot(d, d) < ((σ[i] + σ[j]) / 2)^2 && return false
    end
    return true
end

hard =
    if isdefined(SMC, :HardSphere) &&
        hasmethod(pair_energy, Tuple{SMC.HardSphere,Float64,Float64,Float64})
        SMC.HardSphere()
    else
        IHardSphere()
    end

@testset "System validation" begin
    rng = test_rng(21)
    box = SMC.CubicBox(Val(3), 10.0)
    x = SMC.random_positions(rng, box, 10)
    @test_throws ArgumentError System(box, x, ones(9), GaussianCore())
    @test_throws ArgumentError System(box, SVector{3,Float64}[], Float64[], GaussianCore())
    @test_throws ArgumentError System(box, [rand(2) for _ in 1:10], ones(10), Ideal())
    @test_throws ArgumentError System(box, x, -ones(10), Ideal())
    # The cutoff (5.5) exceeds half the box width.
    @test_throws ArgumentError System(box, x, ones(10), GaussianCore(; rc=5.5))
    @test System(box, x, ones(10), GaussianCore(; rc=5.0)) isa System
    @test_throws ArgumentError System(box, x, rand(rng, 10) .+ 0.5, AsymmetricStep())
    # Overlapping hard cores.
    y = copy(x)
    y[2] = y[1] + SVector(0.5, 0.0, 0.0)
    @test_throws ArgumentError System(box, y, ones(10), hard)
    err = try
        System(box, y, ones(10), hard)
    catch e
        e
    end
    @test occursin("overlapping", sprint(showerror, err))
end

@testset "positions, accessors and show" begin
    rng = test_rng(22)
    for D in (2, 3), T in (Float64, Float32)
        L = SVector{D,T}(ntuple(d -> T(8 + d), D))
        box = OrthorhombicBox(L)
        N = 50
        x = uniform_positions(rng, L, N)
        σ = T(0.5) .+ rand(rng, T, N)
        # Positions outside the box are wrapped.
        shifted = [xi + L .* SVector{D,T}(ntuple(_ -> T(rand(rng, -2:2)), D)) for xi in x]
        sys = System(box, shifted, σ, Ideal())
        @test length(sys) == N
        @test SMC.dimension(sys) == D
        @test eltype(positions(sys)) == SVector{D,T}
        @test all(isapprox.(positions(sys), x; rtol=0, atol=50 * eps(T)))
        @test all(s -> all(0 .<= s .< 1), sys.frac)
        @test diameters(sys) == σ
        @test diameters(sys) !== sys.diameters
        @test SMC.volume(sys) ≈ prod(L)
        @test density(sys) ≈ N / prod(L)
        φ = D == 2 ? π / 4 * sum(abs2, σ) / prod(L) : π / 6 * sum(σ .^ 3) / prod(L)
        @test packing_fraction(sys) ≈ φ rtol = 1e-5
        @test packing_fraction(sys) isa T
        str = sprint(show, sys)
        @test occursin("System{$(D),$(T)}", str)
        @test occursin("N=$(N)", str)
        @test occursin("Ideal", str)
    end
    # Any vectors work as positions.
    box = SMC.CubicBox(Val(2), 5.0)
    sys = System(box, [[1, 2], [3.0, 4.0]], [1, 1], Ideal())
    @test positions(sys) ≈ [SVector(1.0, 2.0), SVector(3.0, 4.0)]
end

@testset "boxes for a density or packing fraction" begin
    for D in (2, 3)
        box = SMC.box_for_density(Val(D), 1000, 0.8)
        @test box isa OrthorhombicBox{D,Float64}
        @test SMC.volume(box) ≈ 1000 / 0.8
        @test all(==(box.lengths[1]), box.lengths)
        σ = rand(test_rng(23), 500) .+ 0.5
        box = SMC.box_for_packing_fraction(Val(D), σ, 0.4)
        sys = System(box, SMC.lattice_positions(box, 500), σ, Ideal())
        @test packing_fraction(sys) ≈ 0.4
    end
    @test_throws ArgumentError SMC.box_for_density(Val(2), 10, 0.0)
    @test_throws ArgumentError SMC.box_for_packing_fraction(Val(2), ones(10), -0.1)
end

@testset "lattice positions" begin
    for (box, N, m) in (
        (SMC.CubicBox(Val(3), 6.0), 27, 3),
        (SMC.CubicBox(Val(3), 6.0), 28, 4),
        (SMC.CubicBox(Val(2), 10.0), 100, 10),
        (OrthorhombicBox(10.0, 20.0), 37, 7),
        (OrthorhombicBox(4.0f0, 5.0f0, 6.0f0), 64, 4),
        (SMC.CubicBox(Val(2), 3.0), 1, 1),
    )
        D = SMC.dimension(box)
        T = eltype(box)
        x = SMC.lattice_positions(box, N)
        L = box.lengths
        @test length(x) == N
        @test eltype(x) == SVector{D,T}
        @test all(xi -> all(0 .< xi .< L), x)
        @test length(unique(x)) == N
        # The first site is offset by half a spacing along each direction.
        @test x[1] ≈ L ./ (2m)
        N > 1 && @test min_distance(L, x) ≈ minimum(L) / m
    end
    @test isempty(SMC.lattice_positions(SMC.CubicBox(Val(2), 3.0), 0))
end

@testset "random positions" begin
    rng = test_rng(24)
    box = OrthorhombicBox(3.0f0, 4.0f0)
    x = SMC.random_positions(rng, box, 1000)
    @test length(x) == 1000
    @test eltype(x) == SVector{2,Float32}
    @test all(xi -> all(0 .<= xi .< box.lengths), x)
end

@testset "random sequential insertion of hard spheres" begin
    rng = test_rng(25)
    for (D, N) in ((2, 400), (3, 400))
        σ = 0.7 .+ 0.6 .* rand(rng, N)
        box = SMC.box_for_packing_fraction(Val(D), σ, 0.3)
        x = SMC.insert_hard_spheres(rng, box, σ)
        @test length(x) == N
        @test eltype(x) == SVector{D,Float64}
        @test no_overlaps(box.lengths, x, σ)
        sys = System(box, x, σ, hard)
        @test isfinite(sys.energy)
        @test packing_fraction(sys) ≈ 0.3
    end
    # Monodisperse, Float32.
    box = SMC.box_for_packing_fraction(Val(2), ones(200), 0.3)
    box = OrthorhombicBox(Float32.(box.lengths))
    x = SMC.insert_hard_spheres(rng, box, ones(200))
    @test eltype(x) == SVector{2,Float32}
    @test no_overlaps(Float64.(box.lengths), [Float64.(xi) for xi in x], ones(200))
    # Impossible packings fail.
    box = SMC.box_for_packing_fraction(Val(2), ones(100), 0.9)
    @test_throws ErrorException SMC.insert_hard_spheres(rng, box, ones(100); max_tries=1000)
end

# A potential that is not truncated by its own `pair_energy`.
struct Untruncated <: Potential end
pair_energy(::Untruncated, r2, σi, σj) = inv(r2)^6
cutoff(::Untruncated, σi, σj) = 1.5

@testset "potentials must vanish beyond their cutoff" begin
    box = CubicBox(Val(2), 10.0)
    x = lattice_positions(box, 16)
    @test_throws ArgumentError System(box, x, ones(16), Untruncated())
    @test System(box, x, ones(16), GaussianCore(; rc=3.0)) isa System
end

# A potential whose cutoff is negative for unlike pairs only, which are not among the
# pairs of extreme diameters checked first by `System`.
struct NegativeCutoff <: Potential end
pair_energy(::NegativeCutoff, r2, σi, σj) = zero(r2)
cutoff(::NegativeCutoff, σi, σj) = σi + σj == 2.5 ? -1.0 : 1.0

@testset "negative pair diameters and cutoffs" begin
    # With NonAdditive(0.9), ε (σmax - σmin) = 1.08 > 1 gives a negative pair diameter.
    m = NonAdditive(0.9)
    @test SMC.mix(m, 1.0, 2.2) < 0
    box = CubicBox(Val(2), 30.0)
    x = lattice_positions(box, 100)
    σ = [isodd(k) ? 1.0 : 2.2 for k in 1:100]
    for pot in
        (PolySoftSphere(; mixing=m), HardSphere(; mixing=m), LennardJones(; mixing=m))
        @test_throws ArgumentError System(box, x, σ, pot)
    end
    msg = try
        System(box, x, σ, PolySoftSphere(; mixing=m))
    catch err
        sprint(showerror, err)
    end
    @test occursin("NonAdditive", msg) && occursin("1.0 and 2.2", msg)
    # Diameters closer together are fine.
    @test System(
        box, x, [isodd(k) ? 1.0 : 1.5 for k in 1:100], PolySoftSphere(; mixing=m)
    ) isa System
    # Negative cutoffs of sampled pairs are rejected by `check_cutoff` itself.
    @test_throws ArgumentError SMC.check_cutoff(NegativeCutoff(), [1.0, 1.5], Xoshiro(1))
    @test SMC.check_cutoff(NegativeCutoff(), [1.0, 2.0], Xoshiro(1)) === nothing
    @test_throws ArgumentError System(
        box, x, [isodd(k) ? 1.0 : 1.5 for k in 1:100], NegativeCutoff()
    )
end
