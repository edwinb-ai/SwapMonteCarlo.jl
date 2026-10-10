# Sanity check of `references.jl` against values computed independently while
# planning. Base only; run from the repository root:
#
#     julia validation/check_references.jl

include(joinpath(@__DIR__, "references.jl"))

using Printf: @printf

"""
    check(label, value, expected, digits)

Print `value` next to `expected` and report a mismatch if they differ after rounding to
`digits` decimals (the precision of `expected`). Returns `true` on agreement.
"""
function check(label, value, expected, digits)
    ok = abs(value - expected) <= 0.5 * 10.0^(-digits) * (1 + 1e-9)
    @printf(
        "%-28s %14.8f  expected %12.*f  %s\n",
        label,
        value,
        digits,
        expected,
        ok ? "ok" : "MISMATCH"
    )
    return ok
end

function main()
    allok = true

    println("Hard-sphere fluid, Kolafa: βP = ρ Z(ρ)")
    for (ρ, βP, d) in (
        (0.3, 0.590469, 6),
        (0.5, 1.634705, 6),
        (0.6, 2.576990, 6),
        (0.7, 4.00887, 5),
        (0.8, 6.216072, 6),
        (0.9, 9.686465, 6),
    )
        allok &= check("ρ = $ρ", ρ * hs_fluid_Z_kolafa(ρ), βP, d)
    end

    println("\nHard disks, Henderson: βP = ρ Z(η = πρ/4)")
    ρ = 0.5
    allok &= check("ρ = $ρ", ρ * hd_Z_henderson(hd_packing_fraction(ρ)), 1.381829, 6)

    println("\nHard-sphere FCC solid, Speedy: βP = ρ Z(ρ)")
    for (ρ, βP) in ((1.04, 11.662), (1.10, 14.593), (1.15, 18.126))
        allok &= check("ρ = $ρ", ρ * hs_solid_Z_speedy(ρ), βP, 3)
    end

    println("\nCarnahan–Starling vs Kolafa (relative difference in Z)")
    for ρ in (0.3, 0.6, 0.9, 0.939)
        zk = hs_fluid_Z_kolafa(ρ)
        zc = hs_fluid_Z_cs(hs_packing_fraction(ρ))
        @printf(
            "ρ = %-6g  Z_Kolafa = %.6f  Z_CS = %.6f  (CS - K)/K = %+.2e\n",
            ρ,
            zk,
            zc,
            (zc - zk) / zk
        )
    end

    println("\nRound trip density_from_pressure")
    for (Z, ρ, br) in (
        (hs_fluid_Z_kolafa, 0.85, (0.0, 1.0)),
        (hs_solid_Z_speedy, 1.12, (1.0, 1.4)),
        (r -> hd_Z_henderson(hd_packing_fraction(r)), 0.7, (0.0, 0.9)),
    )
        βP = ρ * Z(ρ)
        ρ2 = density_from_pressure(Z, βP; bracket=br)
        ok = abs(ρ2 - ρ) < 1e-10
        allok &= ok
        @printf(
            "βP = %10.6f -> ρ = %.12f (expected %g) %s\n", βP, ρ2, ρ, ok ? "ok" : "MISMATCH"
        )
    end

    println(
        "\nNIST LJ tables: $(length(LJ_NIST_MC_RC3_LRC)) NVT MC rows, " *
        "$(length(LJ_NIST_CUBOID_REFERENCE.lrc3)) cuboid configurations",
    )

    println(allok ? "\nAll reference values agree." : "\nSOME VALUES DISAGREE.")
    return allok
end

main() || exit(1)
