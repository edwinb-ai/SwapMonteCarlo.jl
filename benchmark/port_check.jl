# Port check: SwapMonteCarlo.jl reproduces the energies of the original `swapmc.jl`.
#
# Run from the repository root (a few seconds):
#
#     julia --project=benchmark benchmark/port_check.jl
#
# The configuration is the RNG-free lattice of `baseline/RESULTS.md`: the square lattice
# of `Orig.initialize_lattice(N, √N)` (density ρ = 1) with the quantile diameters
# `σ_k = F⁻¹((k - 1/2)/N)` of P(σ) ∝ σ⁻³ on [0.73, 1.62], assigned to the lattice sites in
# order. Large particles overlap (U/N ≈ 30), so the check is sensitive to the pair
# potential, the non-additive mixing rule and the cutoff. The same configuration is built
# in SwapMonteCarlo.jl with `PolySoftSphere(; xc=1.25, mixing=NonAdditive(0.2))` and the
# energies per particle are compared with each other and with the reference numbers
# recorded in `baseline/RESULTS.md`.

using Test: @test, @testset
using Printf: @printf
using StaticArrays: SVector
using SwapMonteCarlo: SwapMonteCarlo, CubicBox, System, PolySoftSphere, NonAdditive
using SwapMonteCarlo: total_energy

include(joinpath(@__DIR__, "baseline", "swapmc_original.jl"))

"""Cutoff of the original script in units of the pair diameter."""
const R_CUT = 1.25

"""Reference `U/N` of the RNG-free lattice from `baseline/RESULTS.md`."""
const REFERENCE = Dict(1000 => 30.092752732640, 10_000 => 30.390921820627)

"""
    lattice_configuration(N) -> (L, x, y, σ)

Box side, lattice coordinates and diameters of the RNG-free reference configuration with
`N` particles at density 1.
"""
function lattice_configuration(N::Int)
    L = sqrt(N / 1.0)
    (x, y) = Orig.initialize_lattice(N, L)
    (smin, smax) = (0.73, 1.62)
    σ = [1.0 / sqrt((1.0 - u) / smin^2 + u / smax^2) for u in ((k - 0.5) / N for k in 1:N)]
    return L, x, y, σ
end

"""
    original_energy(N) -> Float64

Energy per particle of the reference configuration computed by the original script
(`Orig.total_energy` with the script's own cell list).
"""
function original_energy(N::Int)
    (L, x, y, σ) = lattice_configuration(N)
    particles = [Orig.Particle(Orig.SVector(x[k], y[k]), σ[k]) for k in 1:N]
    (cells, _) = Orig.build_cell_list(particles, L, R_CUT * maximum(σ) * 1.1)
    return Orig.total_energy(particles, cells, R_CUT) / N
end

"""
    package_energy(N) -> Float64

Energy per particle of the reference configuration computed by SwapMonteCarlo.jl.
"""
function package_energy(N::Int)
    (L, x, y, σ) = lattice_configuration(N)
    positions = [SVector(x[k], y[k]) for k in 1:N]
    potential = PolySoftSphere(; xc=R_CUT, mixing=NonAdditive(0.2))
    sys = System(CubicBox(Val(2), L), positions, σ, potential)
    return total_energy(sys) / N
end

@testset "port check against swapmc.jl" begin
    for N in (1000, 10_000)
        orig = original_energy(N)
        ours = package_energy(N)
        @printf(
            "N = %-6d U/N original %.12f  package %.12f  reference %.12f  rel. diff %.1e\n",
            N,
            orig,
            ours,
            REFERENCE[N],
            abs(ours - orig) / abs(orig),
        )
        @test isapprox(ours, orig; rtol=1e-12)
        @test isapprox(ours, REFERENCE[N]; rtol=1e-12)
        @test isapprox(orig, REFERENCE[N]; rtol=1e-12)
    end
end
