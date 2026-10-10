# Equation of state of monodisperse hard disks (2D) in the NPT ensemble.
#
#     julia --project=validation validation/hd2d_eos.jl
#
# The pressures βP = ρ Z_Henderson(η) for the area fractions η ∈ {0.2, 0.3, 0.4, 0.5, 0.6}
# are imposed and the measured ⟨ρ⟩ is compared with the Henderson density ρ = 4η/π. The
# virtual-compression pressure (ξ = 0.01) is also measured in NVT at the Henderson
# density.
#
# Henderson's equation of state is an approximation: it reproduces the second and third
# virial coefficients and is accurate to a few 0.1 % in Z up to η ≈ 0.5, but deviates
# more at higher density. Pass criteria: |⟨ρ⟩ - ρ_H| ≤ 4 SE + 0.5 % for η ≤ 0.5 and
# + 1 % for η = 0.6; |P_virtual - βP| ≤ 4 SE + 1 % (+ 2 % at η = 0.6), which includes
# the O(ξ) bias of the estimator.

include(joinpath(@__DIR__, "common.jl"))

function main()
    seed = getseed(32)
    t0 = time()
    N = 400
    nequil = 20_000
    nsweeps = 200_000
    rows = Vector{Vector{String}}()
    vp_rows = Vector{Vector{String}}()
    allok = true
    for (k, η) in enumerate((0.2, 0.3, 0.4, 0.5, 0.6))
        rng = StableRNG(seed + k)
        ρH = 4η / π
        βP = ρH * hd_Z_henderson(η)
        rel = η <= 0.5 ? 0.005 : 0.01
        σ = ones(N)
        box = box_for_packing_fraction(Val(2), σ, min(η, 0.3))
        sys = System(box, insert_hard_spheres(rng, box, σ), σ, HardSphere())
        sim = Simulation(
            sys, NPT(1.0, βP), Displacement(0.1) => 1.0, VolumeMove(0.01) => 4 / N; rng=rng
        )
        (ρm, se, τ, acc) = npt_density(sim, nequil, nsweeps)
        ok = abs(ρm - ρH) <= 4 * se + rel * ρH
        allok &= ok
        push!(
            rows,
            [
                @sprintf("%.2f", η),
                @sprintf("%.6f", βP),
                @sprintf("%.5f", ρH),
                fmt_pm(ρm, se; digits=5),
                @sprintf("%+.3f ± %.3f", 100 * (ρm / ρH - 1), 100 * se / ρH),
                @sprintf("%.0f", τ),
                @sprintf("%.2f / %.2f", acc.displacement, acc.volume),
                passfail(ok),
            ],
        )
        ((pv, pse), (pv3, pse3)) = virtual_pressure_at(rng, sim, ρH, 2000, 60_000)
        okv = abs(pv - βP) <= 4 * pse + 2 * rel * βP
        allok &= okv
        push!(
            vp_rows,
            [
                @sprintf("%.2f", η),
                @sprintf("%.6f", βP),
                fmt_pm(pv, pse; digits=4),
                @sprintf("%+.2f ± %.2f", 100 * (pv / βP - 1), 100 * pse / βP),
                fmt_pm(pv3, pse3; digits=4),
                @sprintf("%+.2f ± %.2f", 100 * (pv3 / βP - 1), 100 * pse3 / βP),
                passfail(okv),
            ],
        )
        println(
            "η = $(η) done: ⟨ρ⟩ = $(fmt_pm(ρm, se; digits=5)), " *
            "P_virtual = $(fmt_pm(pv, pse))",
        )
    end
    runtime = time() - t0
    intro = """
    Monodisperse hard disks (σ = 1, kT = 1), N = $(N), NPT with displacements and
    isotropic volume moves (about 4 volume attempts per sweep). Random insertion at
    η = min(η, 0.3), compression at the target pressure, $(nequil) equilibration and
    $(nsweeps) production sweeps. SE: blocked standard error; τ: integrated
    autocorrelation time of the density in sweeps.

    Overall: $(allok ? "all checks pass" : "SOME CHECKS FAIL").
    """
    t1 =
        "## NPT density vs Henderson (pass: |Δρ| ≤ 4 SE + 0.5 %, 1 % at η = 0.6)\n\n" *
        md_table(
            [
                "η Henderson",
                "βP",
                "ρ Henderson",
                "⟨ρ⟩ ± SE",
                "Δρ/ρ (%)",
                "τ_ρ",
                "acc. disp / vol",
                "check",
            ],
            rows,
        )
    t2 =
        "## Virtual pressure in NVT at the Henderson density (check on ξ = 0.01)\n\n" *
        md_table(
            [
                "η",
                "βP Henderson",
                "βP virtual ξ = 0.01",
                "ΔP/P (%)",
                "βP virtual ξ = 0.003",
                "ΔP/P (%)",
                "check",
            ],
            vp_rows,
        )
    notes = """
    ## Notes

    - A positive Δρ means that Henderson's equation of state overestimates the pressure
      at that density (the simulated fluid is denser at the same pressure).
    - At η ≥ 0.5 both measurements point the same way: the NPT density is higher than
      Henderson's (by about 0.3 %) and the NVT virtual pressure at the Henderson density
      is lower (by about 1 %; d ln P / d ln ρ ≈ 4 there, so the two are consistent),
      with the same result for ξ = 0.01 and ξ = 0.003 (no visible O(ξ) bias). We
      attribute this to the limited accuracy of Henderson's approximate equation at
      high density (it is built to reproduce the low-order virial coefficients) rather
      than to a sampling error, since the 3D hard-sphere runs against the much more
      accurate Kolafa equation (`hs3d_eos.md`) test the same code paths and agree at
      the 0.1 % level. A more accurate hard-disk reference (e.g. Kolafa and Rottner,
      Mol. Phys. 104, 3435 (2006)) would settle it.
    - The liquid–hexatic transition of hard disks is at η ≈ 0.70, so all states are in
      the isotropic fluid.
    """
    write_results(
        "hd2d_eos",
        "Hard-disk equation of state (2D)",
        intro,
        t1,
        t2,
        notes;
        seed=seed,
        runtime=runtime,
    )
    return allok
end

main()
