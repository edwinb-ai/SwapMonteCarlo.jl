# Cell list: neighbor searches against brute force, incremental updates and rebuilds.

import SwapMonteCarlo: is_repulsive

"""
Polydisperse soft spheres, `u = (σij/r)^12 - xc^-12` for `r < xc σij`, with a mixing rule.
File-local stand-in for the package potentials.
"""
struct CLSoftSphere{M} <: Potential
    mixing::M
    xc::Float64
end
CLSoftSphere(mixing=SMC.NonAdditive(0.2)) = CLSoftSphere(mixing, 1.25)
function pair_energy(p::CLSoftSphere, r2, σi, σj)
    σ = SMC.mix(p.mixing, σi, σj)
    r2 < (p.xc * σ)^2 || return zero(r2)
    x6 = (σ^2 / r2)^3
    return oftype(r2, x6 * x6 - p.xc^-12)
end
function pair_virial(p::CLSoftSphere, r2, σi, σj)
    σ = SMC.mix(p.mixing, σi, σj)
    r2 < (p.xc * σ)^2 || return zero(r2)
    x6 = (σ^2 / r2)^3
    return oftype(r2, 12 * x6 * x6)
end
cutoff(p::CLSoftSphere, σi, σj) = p.xc * SMC.mix(p.mixing, σi, σj)
is_repulsive(::CLSoftSphere) = true

"""
    cell_pairs(sys, cells=sys.cells) -> Vector{Tuple{Int,Int}}

All pairs `i < j` within their pair cutoff found through the cell list `cells` (duplicates
kept).
"""
function cell_pairs(sys, cells=sys.cells)
    pairs = Tuple{Int,Int}[]
    for i in eachindex(sys.frac)
        c = SMC.cell_coordinates(cells, sys.frac[i])
        for I in CartesianIndices(cells.stencil)
            cell = SMC.neighbor_cell(cells.wrap, c, I)
            for k in 1:cells.counts[cell]
                j = Int(cells.members[k, cell])
                j > i || continue
                dx = SMC.min_image(sys.box, sys.frac[j] - sys.frac[i])
                rc = cutoff(sys.potential, sys.diameters[i], sys.diameters[j])
                dot(dx, dx) < rc^2 && push!(pairs, (i, j))
            end
        end
    end
    return pairs
end

"""
    check_cells(sys) -> Bool

Whether the cell list is consistent: every particle is stored exactly once, in the slot
and cell recorded for it, and the cells contain the same particles as a fresh build.
"""
function check_cells(sys)
    cells = sys.cells
    N = length(sys.frac)
    seen = zeros(Int, N)
    for cell in eachindex(cells.counts), k in 1:cells.counts[cell]
        j = cells.members[k, cell]
        seen[j] += 1
        cells.cell_of[j] == cell || return false
        cells.slot_of[j] == k || return false
    end
    all(==(1), seen) || return false
    fresh = SMC.CellList(sys.frac, sys.box, sys.rc; range=cells.range)
    fresh.ncells == cells.ncells || return false
    for cell in eachindex(cells.counts)
        a = Set(cells.members[1:cells.counts[cell], cell])
        b = Set(fresh.members[1:fresh.counts[cell], cell])
        a == b || return false
    end
    return true
end

"""
    matches_brute(sys) -> Bool

Whether the total energy and the set of interacting pairs agree with brute force.
"""
function matches_brute(sys; rtol=1e-12)
    L = sys.box.lengths
    x = positions(sys)
    σ = sys.diameters
    pairs = cell_pairs(sys)
    ok_pairs =
        length(pairs) == length(Set(pairs)) &&
        Set(pairs) == brute_neighbors(L, x, σ, sys.potential)
    E = brute_energy(L, x, σ, sys.potential)
    return ok_pairs && isapprox(total_energy(sys), E; rtol=rtol, atol=1e-10)
end

potentials = Any[
    ("mono", CLSoftSphere(SMC.Additive()), :mono),
    ("poly non-additive", CLSoftSphere(SMC.NonAdditive(0.2)), :poly),
]
if isdefined(SMC, :PolySoftSphere) &&
    hasmethod(pair_energy, Tuple{SMC.PolySoftSphere,Float64,Float64,Float64})
    push!(potentials, ("PolySoftSphere", SMC.PolySoftSphere(), :poly))
end

@testset "random configurations vs brute force" begin
    rng = test_rng(1)
    for D in (2, 3), (name, pot, kind) in potentials
        @testset "$(D)D $(name)" begin
            for _ in 1:20
                N = D == 2 ? 250 : 300
                σ = kind === :mono ? ones(N) : 0.7 .+ 0.6 .* rand(rng, N)
                box = SMC.box_for_density(Val(D), N, 0.4 + 0.6 * rand(rng))
                x = uniform_positions(rng, box.lengths, N)
                sys = System(box, x, σ, pot)
                @test all(>=(3), sys.cells.ncells)
                @test matches_brute(sys)
                # The other cell range finds the same pairs.
                other = SMC.CellList(sys.frac, sys.box, sys.rc; range=3 - sys.cells.range)
                L = sys.box.lengths
                @test Set(cell_pairs(sys, other)) ==
                    brute_neighbors(L, positions(sys), sys.diameters, sys.potential)
            end
        end
    end
end

@testset "few cells per side: no double counting" begin
    rng = test_rng(2)
    pot = GaussianCore(; A=1.0, rc=1.0)
    # Expected cell counts with range 1 (cells at least rc wide) and range 2 (cells at
    # least rc/2 wide, so up to 5 cells per side are still "few"). The total is capped at
    # 2N cells.
    for (lengths, expected1, expected2) in (
        ((2.01, 2.5, 3.5), (1, 2, 3), (3, 4, 6)),
        ((2.5, 2.01, 2.9), (2, 1, 2), (4, 3, 5)),
        ((3.5, 3.9, 2.02), (3, 3, 1), (6, 6, 3)),
        ((2.01, 2.01, 2.01), (1, 1, 1), (3, 3, 3)),
        ((2.01, 3.5), (1, 3), (3, 6)),
        ((2.5, 2.5), (2, 2), (4, 4)),
        ((2.02, 2.03), (1, 1), (3, 3)),
        ((3.2, 7.0), (3, 6), (6, 10)),
    )
        box = OrthorhombicBox(lengths...)
        D = length(lengths)
        for _ in 1:5
            N = D == 2 ? 30 : 60
            x = uniform_positions(rng, box.lengths, N)
            sys = System(box, x, ones(N), pot)
            @test matches_brute(sys)
            brute = brute_neighbors(box.lengths, positions(sys), sys.diameters, pot)
            for (range, expected) in ((1, expected1), (2, expected2))
                cells = SMC.CellList(sys.frac, sys.box, sys.rc; range=range)
                @test cells.ncells == expected
                pairs = cell_pairs(sys, cells)
                @test length(pairs) == length(Set(pairs))
                @test Set(pairs) == brute
            end
        end
    end
end

@testset "incremental updates with cell_move!" begin
    rng = test_rng(3)
    for D in (2, 3), range in (1, 2)
        N = 400
        box = SMC.box_for_density(Val(D), N, 0.8)
        x = uniform_positions(rng, box.lengths, N)
        σ = 0.7 .+ 0.6 .* rand(rng, N)
        sys0 = System(box, x, σ, CLSoftSphere())
        cells = SMC.CellList(sys0.frac, sys0.box, sys0.rc; range=range)
        sys = System(sys0.frac, sys0.diameters, box, sys0.potential, cells, sys0.rc, 0.0)
        for step in 1:(10 ^ 4)
            i = rand(rng, 1:N)
            if isodd(step)
                # Small displacement, wrapped into the box.
                ds = (rand(rng, SVector{D,Float64}) .- 0.5) .* 0.05
                s = SMC.wrap_fractional(sys.frac[i] + ds)
            else
                s = rand(rng, SVector{D,Float64})
            end
            sys.frac[i] = s
            SMC.cell_move!(sys.cells, i, s)
        end
        @test check_cells(sys)
        @test matches_brute(sys)

        # Pile up particles in one cell to overflow its storage.
        capacity = size(sys.cells.members, 1)
        target = sys.frac[1]
        for i in 2:(capacity + 10)
            s = SMC.wrap_fractional(target + 1e-3 * rand(rng, SVector{D,Float64}))
            sys.frac[i] = s
            SMC.cell_move!(sys.cells, i, s)
        end
        @test size(sys.cells.members, 1) > capacity
        @test check_cells(sys)
        @test matches_brute(sys; rtol=1e-10)
    end
end

@testset "rebuild after box scaling" begin
    rng = test_rng(4)
    for D in (2, 3)
        N = 500
        box = SMC.box_for_density(Val(D), N, 0.5)
        x = uniform_positions(rng, box.lengths, N)
        σ = 0.7 .+ 0.6 .* rand(rng, N)
        sys = System(box, x, σ, CLSoftSphere())
        n0 = sys.cells.ncells
        @test SMC.cells_valid(sys.cells, sys.box, sys.rc)
        # Small changes stay within the hysteresis.
        @test SMC.cells_valid(sys.cells, SMC.scale(sys.box, 1.005), sys.rc)
        @test SMC.cells_valid(sys.cells, SMC.scale(sys.box, 0.995), sys.rc)

        # Compression: cells become too narrow.
        sys.box = SMC.scale(box, 0.8)
        @test !SMC.cells_valid(sys.cells, sys.box, sys.rc)
        SMC.rebuild_cells!(sys)
        @test all(sys.cells.ncells .< n0)
        @test SMC.cells_valid(sys.cells, sys.box, sys.rc)
        @test check_cells(sys)
        @test matches_brute(sys)

        # Expansion: more cells are worth it.
        sys.box = SMC.scale(box, 1.5)
        @test !SMC.cells_valid(sys.cells, sys.box, sys.rc)
        SMC.rebuild_cells!(sys)
        @test all(sys.cells.ncells .> n0)
        @test check_cells(sys)
        @test matches_brute(sys)
    end
end

@testset "zero cutoff (ideal gas)" begin
    rng = test_rng(5)
    for D in (2, 3), N in (1, 2, 50)
        box = SMC.CubicBox(Val(D), 10.0)
        sys = System(box, uniform_positions(rng, box.lengths, N), ones(N), Ideal())
        @test sys.rc == 0
        @test prod(sys.cells.ncells) <= max(2N, 1)
        @test total_energy(sys) == 0
        @test SMC.particle_energy(sys, 1) == 0
        @test check_cells(sys)
        @test SMC.cells_valid(sys.cells, SMC.scale(box, 0.5), sys.rc)
    end
end

@testset "positions at the upper edge" begin
    rng = test_rng(6)
    for T in (Float64, Float32), D in (2, 3)
        box = SMC.CubicBox(Val(D), T(7))
        N = 60
        x = uniform_positions(rng, box.lengths, N)
        sys = System(box, x, ones(T, N), GaussianCore(; A=1.0, rc=1.0))
        edge = SVector{D,T}(ntuple(_ -> prevfloat(one(T)), D))
        # One particle in the corner cell, two on its faces.
        for (i, s) in enumerate((
            edge, Base.setindex(edge, rand(rng, T), 1), Base.setindex(edge, rand(rng, T), D)
        ))
            sys.frac[i] = s
            SMC.cell_move!(sys.cells, i, s)
        end
        c = SMC.cell_coordinates(sys.cells, edge)
        @test c == sys.cells.ncells .- 1
        @test check_cells(sys)
        E = brute_energy(box.lengths, positions(sys), sys.diameters, sys.potential)
        @test total_energy(sys) ≈ E rtol = T == Float64 ? 1e-12 : 1e-4
    end
end
