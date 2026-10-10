# Periodic boxes, coordinate transformations, minimum image and mixing rules.

@testset "box constructors" begin
    b = OrthorhombicBox(2.0, 3.0, 4.0)
    @test b isa SMC.AbstractBox{3,Float64}
    @test SMC.dimension(b) == 3
    @test eltype(b) == Float64
    @test volume(b) == 24.0
    @test SMC.perpendicular_widths(b) == SVector(2.0, 3.0, 4.0)
    @test OrthorhombicBox(SVector(2, 3)) isa OrthorhombicBox{2,Float64}
    @test OrthorhombicBox(2, 3.0f0) isa OrthorhombicBox{2,Float32}
    @test OrthorhombicBox(SVector(1.0f0, 2.0f0)) isa OrthorhombicBox{2,Float32}

    c2 = CubicBox(Val(2), 3)
    @test c2 isa OrthorhombicBox{2,Float64}
    @test volume(c2) == 9.0
    c3 = CubicBox(Val(3), 2.0f0)
    @test c3 isa OrthorhombicBox{3,Float32}
    @test volume(c3) === 8.0f0

    @test_throws ArgumentError OrthorhombicBox(1.0, 0.0)
    @test_throws ArgumentError OrthorhombicBox(1.0, -2.0, 1.0)
    @test_throws ArgumentError CubicBox(Val(3), -1.0)
    @test_throws ArgumentError OrthorhombicBox(SVector(1.0, NaN))
end

@testset "scale" begin
    b = OrthorhombicBox(2.0, 3.0)
    s = SMC.scale(b, 1.5)
    @test s.lengths == SVector(3.0, 4.5)
    @test volume(s) ≈ 1.5^2 * volume(b)
    @test SMC.scale(b, SVector(2.0, 0.5)).lengths == SVector(4.0, 1.5)
    @test s.inv_lengths ≈ inv.(s.lengths)
    @test_throws ArgumentError SMC.scale(b, -1.0)
end

@testset "cartesian/fractional round trip: D = $D" for D in (2, 3)
    rng = test_rng(1)
    L = SVector(ntuple(k -> 1.0 + k + rand(rng), Val(D)))
    b = OrthorhombicBox(L)
    nbad = 0
    for _ in 1:1000
        s = SVector(ntuple(_ -> rand(rng), Val(D)))
        x = SMC.cartesian(b, s)
        nbad += !isapprox(SMC.fractional(b, x), s; atol=4eps())
        nbad += !(SMC.cartesian(b, SMC.fractional(b, x)) ≈ x)
    end
    @test nbad == 0
end

@testset "wrap_fractional" begin
    cases = [
        -1e-17,
        1.0,
        -1.0,
        3.7,
        -0.0,
        0.0,
        prevfloat(1.0),
        nextfloat(1.0),
        prevfloat(0.0),
        nextfloat(-1.0),
        prevfloat(-1.0),
        -prevfloat(1.0),
        -2.5,
        1e10 + 0.25,
        -eps(),
    ]
    for x in cases
        w = SMC.wrap_fractional(SVector(x, x))
        @test all(v -> 0 <= v < 1, w)
        @test w[1] === w[2]
    end
    @test SMC.wrap_fractional(SVector(-1e-17, 0.5))[1] == 0.0
    @test SMC.wrap_fractional(SVector(1.0, 0.5))[1] == 0.0
    @test SMC.wrap_fractional(SVector(-1.0, 0.5))[1] == 0.0
    @test SMC.wrap_fractional(SVector(3.7, 0.5))[1] ≈ 0.7
    @test SMC.wrap_fractional(SVector(-0.25, 1.25)) == SVector(0.75, 0.25)
    @test SMC.wrap_fractional(SVector(prevfloat(1.0), 0.5))[1] == prevfloat(1.0)
    w32 = SMC.wrap_fractional(SVector(-1.0f-9, 1.0f0, 2.5f0))
    @test eltype(w32) == Float32
    @test all(v -> 0 <= v < 1, w32)
    rng = test_rng(2)
    nbad = 0
    for _ in 1:10_000
        x = SVector((rand(rng, 3) .- 0.5) .* 10.0^rand(rng, -20:3)...)
        nbad += !all(v -> 0 <= v < 1, SMC.wrap_fractional(x))
    end
    @test nbad == 0
end

"""
    brute_min_image(L, d)

Shortest periodic image of the Cartesian separation `d` among the `3^D` images.
"""
function brute_min_image(L::SVector{D}, d::SVector{D}) where {D}
    best = d
    for n in Iterators.product(ntuple(_ -> -1:1, Val(D))...)
        e = d + L .* SVector(n)
        dot(e, e) < dot(best, best) && (best = e)
    end
    return best
end

@testset "min_image vs brute force: D = $D" for D in (2, 3)
    rng = test_rng(3)
    L = SVector(ntuple(k -> 1.0 + k * rand(rng), Val(D)))
    b = OrthorhombicBox(L)
    nbad = 0
    for _ in 1:10_000
        si = SVector(ntuple(_ -> rand(rng), Val(D)))
        sj = SVector(ntuple(_ -> rand(rng), Val(D)))
        d = SMC.min_image(b, sj - si)
        ref = brute_min_image(L, SMC.cartesian(b, sj - si))
        nbad += !(d ≈ ref) || !(norm(d) ≈ norm(ref))
        # The minimum image lies inside the Wigner-Seitz cell of the box.
        nbad += any(abs.(d) .> L ./ 2 .+ 1e-12)
    end
    @test nbad == 0
end

@testset "fits_cutoff" begin
    b = OrthorhombicBox(4.0, 6.0)
    @test SMC.fits_cutoff(b, 2.0)
    @test !SMC.fits_cutoff(b, nextfloat(2.0))
    @test SMC.fits_cutoff(b, 0.0)
    @test SMC.fits_cutoff(CubicBox(Val(3), 10.0), 4.9)
    @test !SMC.fits_cutoff(CubicBox(Val(3), 10.0), 5.1)
end

@testset "mixing rules" begin
    rng = test_rng(4)
    rules = [Additive(), NonAdditive(0.0), NonAdditive(0.2), NonAdditive(0.9)]
    @testset "symmetry: $(rule)" for rule in rules
        nbad = 0
        for _ in 1:10_000
            a = 0.5 + rand(rng)
            b = 0.5 + rand(rng)
            nbad += SMC.mix(rule, a, b) !== SMC.mix(rule, b, a)
            nbad += SMC.mix(rule, a, a) != a
        end
        @test nbad == 0
    end
    @testset "NonAdditive(0) equals Additive" begin
        pairs = [(0.5 + rand(rng), 0.5 + rand(rng)) for _ in 1:1000]
        @test all(p -> SMC.mix(NonAdditive(0), p...) == SMC.mix(Additive(), p...), pairs)
    end
    @test SMC.mix(NonAdditive(0.2), 0.73, 1.62) ≈ 0.96585
    @test SMC.mix(Additive(), 0.73, 1.62) ≈ 1.175
    # With ε ≥ 0 the pair diameter never exceeds the larger diameter.
    @test SMC.mix(NonAdditive(0.2), 0.73, 1.62) <= 1.62
    @test NonAdditive(0.2) isa SMC.MixingRule
    @test_throws ArgumentError NonAdditive(-0.1)
    @test_throws ArgumentError NonAdditive(1.0)
    @test_throws ArgumentError NonAdditive(1.5)
end
