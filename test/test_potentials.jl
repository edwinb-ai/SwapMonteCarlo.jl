# Built-in pair potentials, tail corrections and the user extension interface.

"""A deliberately asymmetric potential, to check that `check_symmetry` catches it."""
struct Asymmetric <: Potential end
pair_energy(::Asymmetric, r2, σi, σj) = r2 < (σi + σj)^2 ? σi / sqrt(r2) : zero(r2)
cutoff(::Asymmetric, σi, σj) = σi + σj

"""
    fd_virial(pot, r, σi, σj; h=1e-6r)

Central finite difference estimate of the pair virial `-r du/dr`.
"""
function fd_virial(pot, r, σi, σj; h=1e-6 * r)
    up = pair_energy(pot, (r + h)^2, σi, σj)
    um = pair_energy(pot, (r - h)^2, σi, σj)
    return -r * (up - um) / (2h)
end

"""
    lj_full(r, σ)

Untruncated Lennard-Jones energy and virial with `ε = 1`.
"""
function lj_full(r, σ)
    x6 = (σ / r)^6
    return 4 * x6 * (x6 - 1), 24 * x6 * (2x6 - 1)
end

"""
    simpson(f, a, b; n=4000)

Composite Simpson quadrature of `f` on `[a, b]` with `n` (even) intervals.
"""
function simpson(f, a, b; n=4000)
    h = (b - a) / n
    s = f(a) + f(b)
    for k in 1:(n - 1)
        s += (isodd(k) ? 4 : 2) * f(a + k * h)
    end
    return s * h / 3
end

"""
    tail_reference(diameters, box, xc) -> (U_lrc, P_lrc)

Tail corrections of the full Lennard-Jones potential beyond `xc σij`, by quadrature over
all ordered pairs `(i, j)` including `i = j`, with the substitution `t = 1/r`.
"""
function tail_reference(diameters, box, xc)
    D = SMC.dimension(box)
    V = volume(box)
    S = D == 2 ? 2π : 4π
    U = 0.0
    P = 0.0
    for σi in diameters, σj in diameters
        σ = (σi + σj) / 2
        rc = xc * σ
        # ∫_rc^∞ g(r) r^(D-1) dr = ∫_0^(1/rc) g(1/t) t^(-1-D) dt.
        U += simpson(t -> t == 0 ? 0.0 : lj_full(1 / t, σ)[1] * t^(-1 - D), 0.0, 1 / rc)
        P += simpson(t -> t == 0 ? 0.0 : lj_full(1 / t, σ)[2] * t^(-1 - D), 0.0, 1 / rc)
    end
    return S * U / (2V), S * P / (2D * V^2)
end

@testset "PolySoftSphere known values" begin
    p = PolySoftSphere(; mixing=Additive())
    @test p.c0 ≈ -1.924145348608 rtol = 1e-14
    @test p.c2 ≈ 2.11106232532992 rtol = 1e-14
    @test p.c4 ≈ -0.5910974510923775 rtol = 1e-14
    @test pair_energy(p, 1.0, 1.0, 1.0) ≈ 0.5958195256295423 rtol = 1e-14
    @test pair_energy(p, 2^(1 / 3), 1.0, 1.0) ≈ 0.04731779703542338 rtol = 1e-12
    @test abs(pair_energy(p, prevfloat(1.25^2), 1.0, 1.0)) < 1e-12
    @test cutoff(p, 1.0, 1.0) == 1.25
    @test SMC.is_repulsive(p)
    @test !SMC.is_hard(p)
    @test PolySoftSphere().mixing == NonAdditive(0.2)

    # Exactly zero at and beyond the cutoff, also for non-trivial diameters.
    @testset "zero beyond cutoff: σ = ($σi, $σj)" for (σi, σj) in
                                                      [(1.0, 1.0), (0.73, 1.62), (1.1, 0.9)]
        q = PolySoftSphere()
        rc2 = cutoff(q, σi, σj)^2
        for r2 in (rc2, nextfloat(rc2), 1.5rc2, 100rc2)
            @test pair_energy(q, r2, σi, σj) === 0.0
            @test pair_virial(q, r2, σi, σj) === 0.0
        end
        @test pair_energy(q, prevfloat(rc2), σi, σj) >= 0
        @test all(r2 -> pair_energy(q, r2, σi, σj) >= 0, range(0.99rc2, rc2; length=2000))
    end

    # The energy vanishes as (rc - r)^3 at the cutoff.
    h = 1e-3
    ratio = pair_energy(p, (1.25 - h)^2, 1.0, 1.0) / pair_energy(p, (1.25 - 2h)^2, 1.0, 1.0)
    @test ratio ≈ 1 / 8 atol = 2e-3

    @test_throws ArgumentError PolySoftSphere(; epsilon=-1.0)
    @test_throws ArgumentError PolySoftSphere(; xc=0.0)
    @test PolySoftSphere(; epsilon=1.0f0, xc=1.25f0).c0 isa Float32
end

@testset "LennardJones" begin
    @testset "values: $(name)" for (name, shift) in [
        ("NoShift", NoShift()), ("EnergyShift", EnergyShift()), ("ForceShift", ForceShift())
    ]
        p = LennardJones(; shift)
        rc = cutoff(p, 1.0, 1.0)
        @test rc == 2.5
        @test pair_energy(p, rc^2, 1.0, 1.0) === 0.0
        @test pair_energy(p, nextfloat(rc^2), 1.0, 1.0) === 0.0
        @test pair_virial(p, rc^2, 1.0, 1.0) === 0.0
        @test !SMC.is_repulsive(p)
        @test !SMC.is_hard(p)
    end
    p0 = LennardJones(; shift=NoShift())
    @test pair_energy(p0, 2^(1 / 3), 1.0, 1.0) ≈ -1 rtol = 1e-14
    @test pair_energy(p0, 1.0, 1.0, 1.0) == 0
    @test pair_virial(p0, 2^(1 / 3), 1.0, 1.0) ≈ 0 atol = 1e-13
    @test pair_energy(p0, prevfloat(2.5^2), 1.0, 1.0) ≈ 4 * (2.5^-12 - 2.5^-6)

    pe = LennardJones(; shift=EnergyShift())
    @test abs(pair_energy(pe, prevfloat(2.5^2), 1.0, 1.0)) < 1e-14
    @test pair_energy(pe, 2^(1 / 3), 1.0, 1.0) ≈ -1 - 4 * (2.5^-12 - 2.5^-6)
    @test pair_virial(pe, 1.3, 1.0, 1.0) == pair_virial(p0, 1.3, 1.0, 1.0)

    pf = LennardJones(; shift=ForceShift(), epsilon=2.0)
    σi, σj = 0.9, 1.3
    rc = cutoff(pf, σi, σj)
    @test rc ≈ 2.5 * 1.1
    @test abs(pair_energy(pf, prevfloat(rc^2), σi, σj)) < 1e-14
    @test abs(pair_virial(pf, prevfloat(rc^2), σi, σj)) < 1e-13
    # The force vanishes linearly at the cutoff.
    w4 = pair_virial(pf, (rc * (1 - 1e-4))^2, σi, σj)
    w5 = pair_virial(pf, (rc * (1 - 1e-5))^2, σi, σj)
    @test abs(w4) < 1e-3
    @test w4 / w5 ≈ 10 rtol = 1e-3

    # Default keyword arguments.
    p = LennardJones()
    @test p.shift === EnergyShift()
    @test p.mixing === Additive()
    @test !p.tail_correction
    @test_throws ArgumentError LennardJones(; tail_correction=true, mixing=NonAdditive(0.1))
    @test_throws ArgumentError LennardJones(; epsilon=0.0)
    @test_throws ArgumentError LennardJones(; xc=-1.0)
end

@testset "LennardJones tail corrections: $(name), D = $D" for (name, σs) in [
        ("monodisperse", fill(1.0, 4)), ("polydisperse", [0.8, 0.95, 1.0, 1.1, 1.3])
    ],
    D in (2, 3)

    box = CubicBox(Val(D), 7.5)
    sys = (; diameters=σs, box)
    for xc in (2.5, 3.0)
        p = LennardJones(; xc, tail_correction=true, shift=NoShift())
        (Uref, Pref) = tail_reference(σs, box, xc)
        @test SMC.energy_lrc(p, sys) ≈ Uref rtol = 1e-9
        @test SMC.pressure_lrc(p, sys) ≈ Pref rtol = 1e-9
        @test SMC.energy_lrc(p, sys) < 0
        @test SMC.pressure_lrc(p, sys) < 0
        # The corrections do not depend on the shift mode.
        q = LennardJones(; xc, tail_correction=true, shift=ForceShift())
        @test SMC.energy_lrc(q, sys) == SMC.energy_lrc(p, sys)
        # Without tail corrections they vanish.
        r = LennardJones(; xc)
        @test SMC.energy_lrc(r, sys) === 0.0
        @test SMC.pressure_lrc(r, sys) === 0.0
    end
    # Textbook monodisperse formulas in 3D.
    if name == "monodisperse" && D == 3
        p = LennardJones(; tail_correction=true)
        N = length(σs)
        ρ = N / volume(box)
        sr3 = 1 / 2.5^3
        @test SMC.energy_lrc(p, sys) ≈ N * 8π / 3 * ρ * (sr3^3 / 3 - sr3)
        @test SMC.pressure_lrc(p, sys) ≈ 16π / 3 * ρ^2 * (2 * sr3^3 / 3 - sr3)
    end
    # Float32 systems get Float32 corrections.
    sys32 = (; diameters=Float32.(σs), box=CubicBox(Val(D), 7.5f0))
    p = LennardJones(; tail_correction=true)
    @test SMC.energy_lrc(p, sys32) isa Float32
    @test SMC.energy_lrc(p, sys32) ≈ SMC.energy_lrc(p, sys) rtol = 1e-5
end

@testset "HardSphere" begin
    p = HardSphere()
    @test pair_energy(p, prevfloat(1.0), 1.0, 1.0) === Inf
    @test pair_energy(p, 0.0, 1.0, 1.0) === Inf
    @test pair_energy(p, 1.0, 1.0, 1.0) === 0.0
    @test pair_energy(p, 2.0, 1.0, 1.0) === 0.0
    @test pair_energy(p, 1.0f0, 1.0f0, 1.0f0) === 0.0f0
    @test pair_energy(p, 0.5f0, 1.0f0, 1.0f0) === Inf32
    @test cutoff(p, 0.8, 1.2) == 1.0
    @test SMC.is_hard(p)
    @test SMC.is_repulsive(p)
    @test_throws ArgumentError pair_virial(p, 1.0, 1.0, 1.0)
    msg = try
        pair_virial(p, 1.0, 1.0, 1.0)
    catch err
        sprint(showerror, err)
    end
    @test occursin("VirtualPressure", msg)
    q = HardSphere(; mixing=NonAdditive(0.2))
    σij = SMC.mix(NonAdditive(0.2), 0.73, 1.62)
    @test cutoff(q, 0.73, 1.62) == σij
    @test pair_energy(q, prevfloat(σij^2), 0.73, 1.62) === Inf
    @test pair_energy(q, σij^2, 0.73, 1.62) === 0.0
end

@testset "PseudoHS" begin
    p = PseudoHS()
    @test p.A ≈ 50 * (50 / 49)^49
    @test pair_energy(p, 1.0, 1.0, 1.0) ≈ 1.0 rtol = 1e-13
    @test pair_energy(PseudoHS(; epsilon=2.5), 1.0, 1.0, 1.0) ≈ 2.5 rtol = 1e-13
    rc = cutoff(p, 1.0, 1.0)
    @test rc ≈ 50 / 49
    @test abs(pair_energy(p, prevfloat(rc^2), 1.0, 1.0)) < 1e-10
    @test abs(pair_virial(p, prevfloat(rc^2), 1.0, 1.0)) < 1e-6
    @test pair_energy(p, rc^2, 1.0, 1.0) === 0.0
    @test pair_energy(p, nextfloat(rc^2), 1.0, 1.0) === 0.0
    @test pair_virial(p, rc^2, 1.0, 1.0) === 0.0
    # Steep but finite inside the core, positive everywhere.
    @test pair_energy(p, 0.81, 1.0, 1.0) > 1e3
    @test all(r -> pair_energy(p, r^2, 1.0, 1.0) >= 0, range(0.9, 1.05; length=2000))
    @test all(r2 -> pair_energy(p, r2, 1.0, 1.0) >= 0, range(0.99rc^2, rc^2; length=2000))
    @test SMC.is_repulsive(p)
    @test !SMC.is_hard(p)
    # Agreement with a direct evaluation of the Mie form.
    for r in (0.95, 1.0, 1.01)
        u = p.A * (r^-50 - r^-49) + 1
        @test pair_energy(p, r^2, 1.0, 1.0) ≈ u rtol = 1e-12
    end
    @test_throws ArgumentError PseudoHS(; epsilon=0)
end

const BUILTIN_POTENTIALS = [
    ("SS12 non-additive", PolySoftSphere()),
    ("SS12 additive", PolySoftSphere(; mixing=Additive(), epsilon=1.5, xc=1.4)),
    ("LJ NoShift", LennardJones(; shift=NoShift())),
    ("LJ EnergyShift", LennardJones(; epsilon=0.7)),
    ("LJ ForceShift", LennardJones(; shift=ForceShift(), mixing=NonAdditive(0.1))),
    ("PseudoHS", PseudoHS()),
    ("PseudoHS non-additive", PseudoHS(; mixing=NonAdditive(0.2), epsilon=2.0)),
]

const ALL_POTENTIALS = [
    BUILTIN_POTENTIALS
    [
        ("HardSphere", HardSphere()),
        ("HardSphere non-additive", HardSphere(; mixing=NonAdditive(0.2))),
    ]
]

@testset "pair_virial vs finite differences: $(name)" for (name, pot) in BUILTIN_POTENTIALS
    rng = test_rng(10)
    nbad = 0
    for _ in 1:50
        σi = 0.73 + 0.89 * rand(rng)
        σj = 0.73 + 0.89 * rand(rng)
        σij = SMC.mix(pot.mixing, σi, σj)
        rc = cutoff(pot, σi, σj)
        r = 0.85σij + rand(rng) * (0.99rc - 0.85σij)
        w = pair_virial(pot, r^2, σi, σj)
        nbad += !isapprox(w, fd_virial(pot, r, σi, σj); rtol=1e-6, atol=1e-7)
    end
    @test nbad == 0
end

@testset "symmetry and max_cutoff: $(name)" for (name, pot) in ALL_POTENTIALS
    rng = test_rng(11)
    σs = 0.73 .+ 0.89 .* rand(rng, 200)
    @test SMC.check_symmetry(pot, σs, rng; npairs=2000) === nothing
    rcmax = SMC.max_cutoff(pot, σs)
    @test all(cutoff(pot, a, b) <= rcmax for a in σs, b in σs)
    @test rcmax == maximum(cutoff(pot, a, b) for a in σs, b in σs)
    # Energies are bitwise symmetric on a grid of distances too.
    nbad = 0
    for _ in 1:1000
        a, b = rand(rng, σs), rand(rng, σs)
        r2 = (rand(rng) * 1.1 * cutoff(pot, a, b))^2
        nbad += !isequal(pair_energy(pot, r2, a, b), pair_energy(pot, r2, b, a))
    end
    @test nbad == 0
end

@testset "type stability: $(name)" for (name, pot) in ALL_POTENTIALS
    for T in (Float64, Float32)
        r2 = T(1.05)
        σi, σj = T(0.9), T(1.1)
        @test @inferred(pair_energy(pot, r2, σi, σj)) isa T
        @test @inferred(cutoff(pot, σi, σj)) isa T
        if !(pot isa HardSphere)
            @test @inferred(pair_virial(pot, r2, σi, σj)) isa T
            # Float32 kernels agree with the Float64 ones.
            u64 = pair_energy(pot, Float64(r2), Float64(σi), Float64(σj))
            @test pair_energy(pot, r2, σi, σj) ≈ u64 rtol = 1e-4
        end
    end
end

@testset "check_symmetry detects asymmetric potentials" begin
    @test_throws ArgumentError SMC.check_symmetry(
        Asymmetric(), [0.8, 0.9, 1.0, 1.2], test_rng(12)
    )
end

@testset "user-defined potentials" begin
    g = GaussianCore()
    @test pair_energy(g, 1.0, 1.0, 1.0) ≈ 2 * exp(-1)
    @test pair_energy(g, 9.0, 1.0, 1.0) == 0
    @test SMC.max_cutoff(g, [0.8, 1.2]) == 3.0
    @test !SMC.is_repulsive(g)
    @test SMC.energy_lrc(g, (; diameters=[1.0, 1.0], box=CubicBox(Val(3), 10.0))) === 0.0

    e = EvaluateGaussian(1.5)
    rng = test_rng(13)
    nbad = 0
    for _ in 1:50
        σi = 0.8 + 0.4 * rand(rng)
        σj = 0.8 + 0.4 * rand(rng)
        r = 0.5 + 2rand(rng)
        σ = (σi + σj) / 2
        u = r < 2.5σ ? 1.5 * exp(-(r / σ)^2) : 0.0
        nbad += !(pair_energy(e, r^2, σi, σj) ≈ u)
        nbad +=
            !isapprox(
                pair_virial(e, r^2, σi, σj), fd_virial(e, r, σi, σj); rtol=1e-6, atol=1e-10
            )
    end
    @test nbad == 0
    @test SMC.max_cutoff(e, [0.8, 1.2]) ≈ 3.0

    @test_throws ErrorException pair_energy(NoMethods(), 1.0, 1.0, 1.0)
    msg = try
        pair_energy(NoMethods(), 1.0, 1.0, 1.0)
    catch err
        sprint(showerror, err)
    end
    @test occursin("pair_energy", msg)
    @test occursin("NoMethods", msg)
    @test_throws ErrorException cutoff(NoMethods(), 1.0, 1.0)
end
