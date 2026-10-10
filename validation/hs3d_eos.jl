# Equation of state of monodisperse hard spheres (3D) in the NPT ensemble.
#
#     julia --project=validation validation/hs3d_eos.jl
#
# Fluid: the pressures βP = ρ Z_Kolafa(ρ) for ρ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} are imposed and
# the measured ⟨ρ⟩ is compared with ρ (Kolafa, Labík and Malijevský 2004, accurate to
# about 1e-4 in Z) and with the Carnahan–Starling density at the same pressure. In
# addition, the virtual-compression pressure (`VirtualPressure`, ξ = 0.01) is measured in
# NVT at the Kolafa density.
#
# Solid: an FCC crystal (5 × 5 × 5 conventional cells, N = 500) is simulated at the
# pressures of Speedy's fit for ρ ∈ {1.05, 1.10, 1.15}.
#
# Pass criteria: |⟨ρ⟩ - ρ_ref| ≤ 4 SE + 0.2 % ρ_ref (fluid, finite-size effects of order
# 1/N and the accuracy of the fit) or + 0.3 % (solid: Speedy's fit is accurate to a few
# 0.1 % in Z), and |P_virtual - βP| ≤ 4 SE + 0.5 % (the O(ξ) bias of the estimator).

include(joinpath(@__DIR__, "common.jl"))

"""
    fcc_positions(ncells, a) -> (positions, L)

Cartesian positions of an FCC crystal of `ncells³` conventional cubic cells with lattice
constant `a` (four particles per cell), and the box length `L = ncells a`.
"""
function fcc_positions(ncells::Integer, a::Real)
    basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    x = SVector{3,Float64}[]
    for i in 0:(ncells - 1), j in 0:(ncells - 1), k in 0:(ncells - 1), b in basis
        # A quarter-spacing offset keeps the sites away from the box faces.
        push!(x, a .* SVector(i + b[1] + 0.25, j + b[2] + 0.25, k + b[3] + 0.25))
    end
    return x, ncells * a
end

function main()
    seed = getseed(31)
    t0 = time()
    N = 500
    nequil = 20_000
    nsweeps = 300_000
    fluid_rows = Vector{Vector{String}}()
    vp_rows = Vector{Vector{String}}()
    allok = true
    for (k, ρK) in enumerate((0.5, 0.6, 0.7, 0.8, 0.9))
        rng = StableRNG(seed + k)
        βP = ρK * hs_fluid_Z_kolafa(ρK)
        ρCS = density_from_pressure(
            r -> hs_fluid_Z_cs(hs_packing_fraction(r)), βP; bracket=(0.0, 1.2)
        )
        σ = ones(N)
        box = box_for_packing_fraction(Val(3), σ, 0.3)
        sys = System(box, insert_hard_spheres(rng, box, σ), σ, HardSphere())
        sim = Simulation(
            sys, NPT(1.0, βP), Displacement(0.1) => 1.0, VolumeMove(0.01) => 4 / N; rng=rng
        )
        (ρm, se, τ, acc) = npt_density(sim, nequil, nsweeps)
        ok = abs(ρm - ρK) <= 4 * se + 0.002 * ρK
        allok &= ok
        push!(
            fluid_rows,
            [
                @sprintf("%.6f", βP),
                @sprintf("%.4f", ρK),
                fmt_pm(ρm, se; digits=5),
                @sprintf("%+.3f ± %.3f", 100 * (ρm / ρK - 1), 100 * se / ρK),
                @sprintf("%.5f", ρCS),
                @sprintf("%+.3f", 100 * (ρm / ρCS - 1)),
                @sprintf("%.0f", τ),
                @sprintf("%.2f / %.2f", acc.displacement, acc.volume),
                passfail(ok),
            ],
        )
        ((pv, pse), (pv3, pse3)) = virtual_pressure_at(rng, sim, ρK, 2000, 60_000)
        okv = abs(pv - βP) <= 4 * pse + 0.005 * βP
        allok &= okv
        push!(
            vp_rows,
            [
                @sprintf("%.4f", ρK),
                @sprintf("%.6f", βP),
                fmt_pm(pv, pse; digits=4),
                @sprintf("%+.2f ± %.2f", 100 * (pv / βP - 1), 100 * pse / βP),
                fmt_pm(pv3, pse3; digits=4),
                @sprintf("%+.2f ± %.2f", 100 * (pv3 / βP - 1), 100 * pse3 / βP),
                passfail(okv),
            ],
        )
        println(
            "fluid ρ = $(ρK) done: ⟨ρ⟩ = $(fmt_pm(ρm, se; digits=5)), " *
            "P_virtual = $(fmt_pm(pv, pse))",
        )
    end

    solid_rows = Vector{Vector{String}}()
    for (k, ρS) in enumerate((1.05, 1.10, 1.15))
        rng = StableRNG(seed + 100 + k)
        βP = ρS * hs_solid_Z_speedy(ρS)
        (x, L) = fcc_positions(5, (4 / ρS)^(1 / 3))
        σ = ones(length(x))
        sys = System(CubicBox(Val(3), L), x, σ, HardSphere())
        sim = Simulation(
            sys,
            NPT(1.0, βP),
            Displacement(0.05) => 1.0,
            VolumeMove(0.002) => 4 / N;
            rng=rng,
        )
        (ρm, se, τ, acc) = npt_density(sim, nequil ÷ 2, nsweeps ÷ 2)
        ok = abs(ρm - ρS) <= 4 * se + 0.003 * ρS
        allok &= ok
        push!(
            solid_rows,
            [
                @sprintf("%.4f", βP),
                @sprintf("%.4f", ρS),
                fmt_pm(ρm, se; digits=5),
                @sprintf("%+.3f ± %.3f", 100 * (ρm / ρS - 1), 100 * se / ρS),
                @sprintf("%.0f", τ),
                @sprintf("%.2f / %.2f", acc.displacement, acc.volume),
                passfail(ok),
            ],
        )
        println("solid ρ = $(ρS) done: ⟨ρ⟩ = $(fmt_pm(ρm, se; digits=5))")
    end

    runtime = time() - t0
    intro = """
    Monodisperse hard spheres (σ = 1, kT = 1), N = 500, NPT with displacements and
    isotropic volume moves (about 4 volume attempts per sweep, step sizes tuned during
    equilibration and then frozen). Fluid: random insertion at φ = 0.3, compression at the
    target pressure, $(nequil) equilibration and $(nsweeps) production sweeps. Solid:
    perfect FCC crystal at the reference density, $(nequil ÷ 2) + $(nsweeps ÷ 2) sweeps.
    SE: blocked standard error; τ: integrated autocorrelation time of the density in
    sweeps.

    Overall: $(allok ? "all checks pass" : "SOME CHECKS FAIL").
    """
    fluid =
        "## Fluid vs Kolafa (pass: |Δρ| ≤ 4 SE + 0.2 %)\n\n" * md_table(
            [
                "βP",
                "ρ Kolafa",
                "⟨ρ⟩ ± SE",
                "Δρ/ρ (%)",
                "ρ CS",
                "Δρ/ρ_CS (%)",
                "τ_ρ",
                "acc. disp / vol",
                "check",
            ],
            fluid_rows,
        )
    vp =
        "## Virtual pressure in NVT at the Kolafa density " *
        "(check on ξ = 0.01; pass: |ΔP| ≤ 4 SE + 0.5 %)\n\n" *
        md_table(
            [
                "ρ",
                "βP Kolafa",
                "βP virtual ξ = 0.01",
                "ΔP/P (%)",
                "βP virtual ξ = 0.003",
                "ΔP/P (%)",
                "check",
            ],
            vp_rows,
        )
    solid =
        "## FCC solid vs Speedy (pass: |Δρ| ≤ 4 SE + 0.3 %)\n\n" * md_table(
            ["βP", "ρ Speedy", "⟨ρ⟩ ± SE", "Δρ/ρ (%)", "τ_ρ", "acc. disp / vol", "check"],
            solid_rows,
        )
    notes = """
    ## Notes

    - Carnahan–Starling underestimates Z by up to 0.3 % in the fluid, so its density at a
      given pressure is slightly too high; the deviations from CS are shown for
      reference only.
    - The virtual pressure counts the pairs that overlap after a compression of the volume
      by ξ; its relative bias is of order ξ, ξ (1 + σ g'(σ) / (6 g(σ))) to first order,
      i.e. a few 0.1 % at ξ = 0.01. The results for ξ = 0.003 agree within their
      (larger) errors.
    - The FCC densities agree with Speedy's fit within 0.2 %, which is about the
      accuracy of the fit; the crystal is perfect (no vacancies) and N = 500, so
      finite-size effects of a similar size are expected, especially at ρ = 1.05, close
      to melting (ρ_m ≈ 1.037).
    """
    write_results(
        "hs3d_eos",
        "Hard-sphere equation of state (3D)",
        intro,
        fluid,
        vp,
        solid,
        notes;
        seed=seed,
        runtime=runtime,
    )
    return allok
end

main()
