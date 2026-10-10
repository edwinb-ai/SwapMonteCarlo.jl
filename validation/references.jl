# Reference equations of state and literature data for the validation runs.
#
# This file depends only on Base, so it can be included from any environment:
#
#     include("validation/references.jl")
#
# Conventions: reduced units, σ = 1 (hard-core / Lennard-Jones diameter), k_B = 1.
# `ρ` is the number density, `η` the packing (area) fraction and `Z = βP/ρ` the
# compressibility factor.

# ---------------------------------------------------------------------------------------
# Hard spheres (3D)
# ---------------------------------------------------------------------------------------

"""
    hs_packing_fraction(ρ)

Packing fraction `η = π ρ / 6` of monodisperse hard spheres of unit diameter at number
density `ρ`.
"""
hs_packing_fraction(ρ) = π * ρ / 6

"""
    hs_fluid_Z_kolafa(ρ)

Compressibility factor `Z = βP/ρ` of the monodisperse hard-sphere fluid at number
density `ρ` (σ = 1), from the accurate fit of Kolafa, Labík and Malijevský to simulation
data in the stable and metastable fluid (it reproduces the exact virial coefficients
up to B₄),

    Z = 1 + 4x + 6x² + 2.3647684x³ − 0.8698551x⁴ + 1.1062803x⁵ − 1.1014221x⁶
          + 0.66605866x⁷ − 0.03633431x⁸ − 0.20965164x¹⁰ + 0.10555569x¹⁴ − 0.00872380x²²,

with `x = η/(1 − η)` and `η = πρ/6`. Use it up to the freezing density (ρ ≈ 0.939,
η ≈ 0.492) and in the metastable fluid; it is the most accurate of the fluid
equations of state in this file.

The implementation is copied verbatim (renamed) from `liquid_eos` in
`free-energy-hs/free_energy.jl`.

Reference: J. Kolafa, S. Labík and A. Malijevský, "Accurate equation of state of the
hard sphere fluid in stable and metastable regions", Phys. Chem. Chem. Phys. 6, 2335
(2004).
"""
function hs_fluid_Z_kolafa(rho)
    eta = pi * rho / 6.0
    x = eta / (1.0 - eta)
    term_1 = 1.0 + 4.0 * x + 6.0 * x^2 + 2.3647684 * x^3 - 0.8698551 * x^4
    term_2 = 1.1062803 * x^5 - 1.1014221 * x^6 + 0.66605866 * x^7 - 0.03633431 * x^8
    term_3 = -0.20965164 * x^10 + 0.10555569 * x^14 - 0.00872380 * x^22

    return term_1 + term_2 + term_3
end

"""
    hs_fluid_Z_cs(η)

Carnahan–Starling compressibility factor of the hard-sphere fluid as a function of the
packing fraction `η`,

    Z = (1 + η + η² − η³) / (1 − η)³.

Less accurate than [`hs_fluid_Z_kolafa`](@ref): it underestimates Z by up to about
0.3 % in the fluid (−0.08 % at ρ = 0.3, −0.27 % at ρ = 0.6, −0.12 % at freezing; see
`check_references.jl`), which matters for long, precise runs. Note the argument is `η`,
not `ρ`.

Reference: N. F. Carnahan and K. E. Starling, "Equation of state for nonattracting rigid
spheres", J. Chem. Phys. 51, 635 (1969).
"""
function hs_fluid_Z_cs(η)
    return (1 + η + η^2 - η^3) / (1 - η)^3
end

"""
    hs_solid_Z_speedy(ρ)

Compressibility factor `Z = βP/ρ` of the hard-sphere FCC crystal at number density `ρ`
(σ = 1), from Speedy's fit to simulation data,

    Z = 3 / (1 − z) − a (z − b) / (z − c),   z = ρ / ρ_cp,  ρ_cp = √2,

with `a = 0.5921`, `b = 0.7072`, `c = 0.601`. Valid from melting (ρ ≈ 1.04, η ≈ 0.545)
up to close packing (η = π√2/6 ≈ 0.7405), where it diverges with the free-volume form
`3/(1 − z)`.

The implementation is copied (renamed) from `solid_eos` in
`free-energy-hs/free_energy.jl`.

Reference: R. J. Speedy, "Pressure and entropy of hard-sphere crystals", J. Phys.:
Condens. Matter 10, 4387 (1998).
"""
function hs_solid_Z_speedy(rho)
    y = rho / sqrt(2.0)
    term_1 = 3.0 / (1.0 - y)
    term_2 = (y - 0.7072) / (y - 0.601)
    result = term_1 - 0.5921 * term_2

    return result
end

# ---------------------------------------------------------------------------------------
# Hard disks (2D)
# ---------------------------------------------------------------------------------------

"""
    hd_packing_fraction(ρ)

Area fraction `η = π ρ / 4` of monodisperse hard disks of unit diameter at number
density `ρ`.
"""
hd_packing_fraction(ρ) = π * ρ / 4

"""
    hd_Z_henderson(η)

Henderson's compressibility factor of the hard-disk fluid as a function of the area
fraction `η = πρ/4`,

    Z = (1 + η²/8) / (1 − η)².

A good approximation in the isotropic fluid, i.e. below the liquid–hexatic transition
(η ≈ 0.70); check the size of its error against accurate hard-disk data before using it
for high-precision comparisons near η ≈ 0.65. Note the argument is `η`, not `ρ`.

Reference: D. Henderson, "A simple equation of state for hard discs", Mol. Phys. 30,
971 (1975).
"""
function hd_Z_henderson(η)
    return (1 + η^2 / 8) / (1 - η)^2
end

# ---------------------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------------------

"""
    density_from_pressure(Zfun, βP; bracket, rtol=1e-13, maxiter=200)

Solve `βP = ρ Zfun(ρ)` for the number density `ρ` by bisection. `Zfun` must take the
number density (wrap packing-fraction forms, e.g.
`ρ -> hd_Z_henderson(hd_packing_fraction(ρ))`). `ρ Z(ρ)` must be increasing on
`bracket = (ρlo, ρhi)` and the root must lie inside it; otherwise an `ArgumentError` is
thrown.

# Keyword arguments
- `bracket`: tuple `(ρlo, ρhi)` that contains the solution.
- `rtol`: stop once the bracket is narrower than `rtol * ρhi`.
- `maxiter`: maximum number of bisection steps.
"""
function density_from_pressure(Zfun, βP; bracket, rtol=1e-13, maxiter=200)
    lo, hi = float.(bracket)
    f(ρ) = ρ * Zfun(ρ) - βP
    flo, fhi = f(lo), f(hi)
    if !(flo <= 0 <= fhi)
        throw(
            ArgumentError(
                "βP = $βP is not bracketed by ρ ∈ $bracket " * "(ρZ - βP = $flo, $fhi)"
            ),
        )
    end
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        if f(mid) < 0
            lo = mid
        else
            hi = mid
        end
        (hi - lo) <= rtol * hi && break
    end
    return (lo + hi) / 2
end

# ---------------------------------------------------------------------------------------
# Lennard-Jones (3D), NIST Standard Reference Simulation Website (SRSW)
# ---------------------------------------------------------------------------------------

"""
    LJ_NIST_MC_RC3_LRC

NIST SRSW NVT Monte Carlo results for the Lennard-Jones fluid, truncated at
`r_c = 3σ` with standard long-range (tail) corrections for energy and pressure. N = 500,
5.0e7 equilibration trials, 2.5e8 production trials, maximum displacement adjusted to
0.40 acceptance, virial pressure. Uncertainties (`dU`, `dP`) are one standard
deviation from block averages of 5.0e7 trials.

Each row is `(; T, ρ, U, dU, P, dP)`: temperature `T*`, density `ρ*`, potential energy
per particle `U*` and pressure `p*` (both including the tail corrections).

Note: the T* = 0.85 and 0.90 isotherms cross the liquid–vapor coexistence region
(T_c ≈ 1.31); the rows at ρ* ≤ 0.009 are vapor-like and ρ* ≥ 0.776 liquid-like. The
lowest liquid densities (ρ* = 0.776, 0.78 at T* = 0.85) are close to coexistence and may
cavitate in long or large runs.

Transcribed on 2026-10-10 from the raw HTML of
https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm, linked from the NIST "Lennard-Jones Fluid
Properties" page (https://www.nist.gov/mml/csd/chemical-informatics-research-group/lennard-jones-fluid-properties).
NIST provides no cut-and-shifted NVT MC table there (only LAMMPS MD and TMMC data with a
linear-force shift).
"""
const LJ_NIST_MC_RC3_LRC = (
    (T=0.85, ρ=1.00e-3, U=-1.0317e-2, dU=2.34e-5, P=8.4402e-4, dP=4.66e-8),
    (T=0.85, ρ=3.00e-3, U=-3.1019e-2, dU=5.91e-5, P=2.4965e-3, dP=4.99e-7),
    (T=0.85, ρ=5.00e-3, U=-5.1901e-2, dU=7.53e-5, P=4.1003e-3, dP=5.05e-7),
    (T=0.85, ρ=7.00e-3, U=-7.2834e-2, dU=1.34e-4, P=5.6565e-3, dP=7.96e-7),
    (T=0.85, ρ=9.00e-3, U=-9.3973e-2, dU=1.29e-4, P=7.1641e-3, dP=2.24e-6),
    (T=0.85, ρ=7.76e-1, U=-5.5121e+0, dU=4.55e-4, P=6.7714e-3, dP=1.77e-3),
    (T=0.85, ρ=7.80e-1, U=-5.5386e+0, dU=7.26e-4, P=4.7924e-2, dP=3.18e-3),
    (T=0.85, ρ=8.20e-1, U=-5.7947e+0, dU=6.03e-4, P=5.5355e-1, dP=4.13e-3),
    (T=0.85, ρ=8.60e-1, U=-6.0305e+0, dU=2.38e-3, P=1.2660e+0, dP=1.36e-2),
    (T=0.85, ρ=9.00e-1, U=-6.2391e+0, dU=5.27e-3, P=2.2314e+0, dP=2.72e-2),
    (T=0.90, ρ=1.00e-3, U=-9.9165e-3, dU=1.89e-5, P=8.9429e-4, dP=2.48e-8),
    (T=0.90, ρ=3.00e-3, U=-2.9787e-2, dU=3.21e-5, P=2.6485e-3, dP=2.54e-7),
    (T=0.90, ρ=5.00e-3, U=-4.9771e-2, dU=3.80e-5, P=4.3569e-3, dP=2.19e-7),
    (T=0.90, ρ=7.00e-3, U=-6.9805e-2, dU=7.66e-5, P=6.0193e-3, dP=1.02e-6),
    (T=0.90, ρ=9.00e-3, U=-8.9936e-2, dU=2.44e-5, P=7.6363e-3, dP=1.44e-6),
    (T=0.90, ρ=7.76e-1, U=-5.4689e+0, dU=4.20e-4, P=2.4056e-1, dP=2.74e-3),
    (T=0.90, ρ=7.80e-1, U=-5.4956e+0, dU=7.86e-4, P=2.7851e-1, dP=2.97e-3),
    (T=0.90, ρ=8.20e-1, U=-5.7456e+0, dU=7.51e-4, P=8.2386e-1, dP=2.85e-3),
    (T=0.90, ρ=8.60e-1, U=-5.9753e+0, dU=5.53e-4, P=1.5781e+0, dP=3.29e-3),
    (T=0.90, ρ=9.00e-1, U=-6.1773e+0, dU=1.57e-3, P=2.5848e+0, dP=9.54e-3),
)

"""
    LJ_NIST_CUBOID_REFERENCE

NIST SRSW "Lennard-Jones Fluid Reference Calculations: Cuboid Cell": pair energy
`Upair*`, pair virial `Wpair* = -Σ r V'(r)` and energy tail correction `ULRC*` for four
sample configurations (download the gzipped tarball linked on that page; it includes a
metadata file with the format). Values are totals as printed by NIST (not per particle).

Truncation schemes (keys):
- `:lrc3`: potential cut at r_c = 3σ (not shifted), with long-range correction;
- `:lrc4`: cut at r_c = 4σ, with long-range correction;
- `:lfs3`: cut at r_c = 3σ with a linear-force shift (`ULRC = 0`).

Each entry is a tuple over configurations 1–4 of `(; U, W, ULRC)`.

Transcribed on 2026-10-10 from
https://www.nist.gov/mml/csd/chemical-informatics-research-group/lennard-jones-fluid-reference-calculations
(which redirects to `.../chemical-informatics-group/lennard-jones-fluid-reference-calculations-cuboid-cell`).
"""
const LJ_NIST_CUBOID_REFERENCE = (
    lrc3=(
        (U=-4.3515e+3, W=-5.6867e+2, ULRC=-1.9849e+2),
        (U=-6.9000e+2, W=-5.6846e+2, ULRC=-2.4230e+1),
        (U=-1.1467e+3, W=-1.1649e+3, ULRC=-4.9622e+1),
        (U=-1.6790e+1, W=-4.6249e+1, ULRC=-5.4517e-1),
    ),
    lrc4=(
        (U=-4.4675e+3, W=-1.2639e+3, ULRC=-8.3769e+1),
        (U=-7.0460e+2, W=-6.5599e+2, ULRC=-1.0226e+1),
        (U=-1.1754e+3, W=-1.3371e+3, ULRC=-2.0942e+1),
        (U=-1.7060e+1, W=-4.7869e+1, ULRC=-2.3008e-1),
    ),
    lfs3=(
        (U=-3.8709e+3, W=3.1754e+2, ULRC=0.0),
        (U=-6.2012e+2, W=-4.4533e+2, ULRC=0.0),
        (U=-1.0210e+3, W=-9.3578e+2, ULRC=0.0),
        (U=-1.5001e+1, W=-4.3096e+1, ULRC=0.0),
    ),
)

# TODO: a cut-and-shifted (energy shift only) LJ reference, e.g. r_c = 2.5σ, is not
# provided by the NIST SRSW pages above. If needed, take it from the literature (e.g.
# Johnson, Zollweg and Gubbins, Mol. Phys. 78, 591 (1993) for the full potential, or a
# dedicated cut-and-shifted EOS) after checking the source.
