# Hard spheres (3D) and hard disks (2D): NPT densities against accurate equations of state,
# and the virtual-compression pressure in NVT.
#
# References (see `validation/references.jl`):
#   - 3D: Kolafa, Labík and Malijevský (2004) fluid EOS; βP = 4.00887 ⇔ ρ = 0.700.
#   - 2D: Henderson (1975) EOS; βP = 1.381829 ⇔ ρ = 0.500 (η = 0.393).
#
# The systematic allowances cover the finite-size effects of the NPT ensemble at
# N = 256 (of order 1/N) and the accuracy of the equations of state (Kolafa: ~1e-4;
# Henderson: ~0.1 % at η ≈ 0.4).

include(joinpath(@__DIR__, "..", "validation", "references.jl"))

"""
    hs_npt_state(rng, D, N, βP; φ0=0.3, nequil, nsweeps)

Monodisperse hard spheres (`D = 3`) or disks (`D = 2`) of unit diameter, inserted at
random at the packing fraction `φ0` and compressed in the NPT ensemble at `kT = 1` and
pressure `βP`. Returns the simulation and the density time series of the production run.
"""
function hs_npt_state(rng, D, N, βP; φ0=0.3, nequil, nsweeps)
    σ = ones(N)
    box = box_for_packing_fraction(Val(D), σ, φ0)
    sys = System(box, insert_hard_spheres(rng, box, σ), σ, HardSphere())
    # About four volume attempts per sweep; compressions stop at the first overlap and
    # expansions need no energy, so they are cheap.
    sim = Simulation(
        sys, NPT(1.0, βP), Displacement(0.1) => 1.0, VolumeMove(0.01) => 4 / N; rng=rng
    )
    equilibrate!(sim, nequil)
    ts = TimeSeries(Density())
    run!(sim, nsweeps; observers=ts)
    return sim, ts["density"]
end

"""
    hs_nvt_from(rng, sim, ρ)

NVT simulation of the hard particles of the NPT simulation `sim` at the density `ρ`. The
NPT run is continued until its density is at least `ρ`, so that the configuration can be
expanded to `ρ` without creating overlaps.
"""
function hs_nvt_from(rng, sim, ρ)
    sys0 = sim.system
    while density(sys0) < ρ
        run!(sim, 1)
    end
    D = SMC.dimension(sys0)
    box = box_for_density(Val(D), length(sys0), ρ)
    f = (volume(box) / volume(sys0.box))^(1 / D)
    sys = System(box, positions(sys0) .* f, diameters(sys0), HardSphere())
    return Simulation(sys, NVT(1.0), Displacement(0.1) => 1.0; rng=rng)
end

"""
    hs_check(rng, D, N, βP, ρref, Zfun; rel_sys, nequil, nsweeps)

Check the NPT density of hard particles at `βP` against `ρref` (within 4 standard errors
plus the relative allowance `rel_sys`), then the virtual pressure in NVT at the measured
density against `ρ Zfun(ρ)`.
"""
function hs_check(rng, D, N, βP, ρref, Zfun; rel_sys, nequil, nsweeps)
    sim, ρ = hs_npt_state(rng, D, N, βP; nequil=nequil, nsweeps=nsweeps)
    ρm = mean(ρ)
    se = block_stderr(ρ)
    @test check_mean(ρm, se, ρref; sys=rel_sys * ρref, se_max=0.003 * ρref)
    # No drift between the two halves of the production run (equilibrated start). Each
    # half has a standard error of about √2 se, so their difference has about 2 se.
    h = length(ρ) ÷ 2
    @test abs(mean(ρ[1:h]) - mean(ρ[(h + 1):end])) <= 8 * se

    # Virtual compressions by the volume fraction ξ count the pairs that would overlap,
    # i.e. the pairs in a thin shell beyond contact. Since the shell has a finite width
    # the estimator has a relative bias of order ξ: expanding the shell integral of
    # r^(D-1) g(r) gives ξ (1 + σ g'(σ) / (2D g(σ))) to first order, a few 0.1 % at
    # ξ = 0.01 (`validation/hs3d_eos.jl` finds agreement with Kolafa within 0.3 % for
    # 0.5 ≤ ρ ≤ 0.9, consistent with ξ = 0.003). A smaller ξ has a smaller bias but
    # much more noise. The allowance of 1 % covers the bias with a wide margin.
    nvt = hs_nvt_from(rng, sim, ρm)
    equilibrate!(nvt, 2000)
    ts = TimeSeries(VirtualPressure(; xi=1e-2); schedule=Every(2))
    run!(nvt, nsweeps ÷ 3; observers=ts)
    p = ts["virtual_pressure"]
    Pref = ρm * Zfun(ρm)
    @test check_mean(mean(p), block_stderr(p), Pref; sys=0.01 * Pref, se_max=0.006 * Pref)
    return nothing
end

@testset "hard spheres 3D: NPT density vs Kolafa (βP = 4.00887)" begin
    rng = test_rng(2300)
    hs_check(
        rng,
        3,
        256,
        4.00887,
        0.700,
        hs_fluid_Z_kolafa;
        rel_sys=0.003,
        nequil=10_000,
        nsweeps=150_000,
    )
end

@testset "hard disks 2D: NPT density vs Henderson (βP = 1.381829)" begin
    rng = test_rng(2301)
    hs_check(
        rng,
        2,
        256,
        1.381829,
        0.500,
        ρ -> hd_Z_henderson(hd_packing_fraction(ρ));
        rel_sys=0.005,
        nequil=10_000,
        nsweeps=150_000,
    )
end
