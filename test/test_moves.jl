# Monte Carlo moves: energy bookkeeping, invariants of accepted and rejected attempts,
# hard spheres, quenches, edge cases, allocations, type stability and step adaptation.

using SwapMonteCarlo: Displacement, Swap, VolumeMove

"""
    mv_snapshot(sys)

Copy of everything a move may change, to check that rejected attempts restore it.
"""
function mv_snapshot(sys)
    c = sys.cells
    return (;
        frac=copy(sys.frac),
        diameters=copy(sys.diameters),
        box=sys.box,
        energy=sys.energy,
        ncells=c.ncells,
        members_ref=c.members,
        members=copy(c.members),
        counts=copy(c.counts),
        cell_of=copy(c.cell_of),
        slot_of=copy(c.slot_of),
    )
end

"""
    mv_same_state(sys, snap) -> Bool

Whether positions, diameters, box and energy are bitwise equal to the snapshot.
"""
function mv_same_state(sys, snap)
    return isequal(sys.frac, snap.frac) &&
           isequal(sys.diameters, snap.diameters) &&
           sys.box === snap.box &&
           sys.energy === snap.energy
end

"""
    mv_same_cells(sys, snap) -> Bool

Whether the cell list is bitwise equal to the snapshot.
"""
function mv_same_cells(sys, snap)
    c = sys.cells
    return c.ncells == snap.ncells &&
           c.members == snap.members &&
           c.counts == snap.counts &&
           c.cell_of == snap.cell_of &&
           c.slot_of == snap.slot_of
end

"""
    mv_cells_consistent(sys) -> Bool

Whether every particle is stored in the cell that contains it, exactly once.
"""
function mv_cells_consistent(sys)
    c = sys.cells
    sum(c.counts) == length(sys) || return false
    for i in eachindex(sys.frac)
        cell = Int(c.cell_of[i])
        cell == SMC.linear_cell(c, sys.frac[i]) || return false
        c.members[c.slot_of[i], cell] == i || return false
    end
    return true
end

"""
    mv_in_unit_cell(s) -> Bool

Whether the fractional position `s` lies in `[0, 1)^D`.
"""
mv_in_unit_cell(s) = all(x -> 0 <= x < 1, s)

"""
    mv_reference_energy(sys)

All-pairs energy of `sys` plus its long range correction.
"""
function mv_reference_energy(sys)
    pot = sys.potential
    U = brute_energy(sys.box.lengths, positions(sys), sys.diameters, pot)
    return U + SMC.energy_lrc(pot, sys)
end

"""
    mv_system(rng, D, N, ρ, pot, σ)

System of `N` particles on a slightly perturbed lattice at density `ρ`.
"""
function mv_system(rng, D, N, ρ, pot, σ)
    box = box_for_density(Val(D), N, ρ)
    a = (1 / ρ)^(1 / D)
    x = [
        xi + 0.1a * (rand(rng, SVector{D,Float64}) .- 0.5) for
        xi in lattice_positions(box, N)
    ]
    return System(box, x, σ, pot)
end

const MV_IPL = InversePowerLaw(0.73, 1.62)
mv_poly(rng, N) = sample_diameters(rng, MV_IPL, N; stratified=true)
mv_narrow(rng, N) = 0.9 .+ 0.2 .* rand(rng, N)

# Name, potential, diameters, density per dimension, kT, pressure, δ, δlnV.
const MV_CONFIGS = [
    (
        "PolySoftSphere NonAdditive",
        PolySoftSphere(; mixing=NonAdditive(0.2)),
        mv_poly,
        (1.0, 0.8),
        0.25,
        8.0,
        0.06,
        0.02,
    ),
    (
        "PolySoftSphere Additive",
        PolySoftSphere(; mixing=Additive()),
        mv_poly,
        (1.0, 0.8),
        0.25,
        8.0,
        0.06,
        0.02,
    ),
    ("LennardJones", LennardJones(), mv_narrow, (0.6, 0.35), 1.0, 0.5, 0.4, 0.1),
    (
        "LennardJones tail",
        LennardJones(; tail_correction=true, shift=NoShift()),
        mv_narrow,
        (0.6, 0.35),
        1.0,
        0.5,
        0.4,
        0.1,
    ),
    (
        "GaussianCore",
        GaussianCore(; A=2.0, rc=3.0),
        mv_narrow,
        (1.0, 0.25),
        0.5,
        1.0,
        0.8,
        0.1,
    ),
]

const MV_MIXES = (:displacement, :swap, :volume, :mixed)

@testset "energy bookkeeping and invariants" begin
    K = 5000
    for D in (2, 3), (cname, pot, sizes, ρs, kT, P, δ, δlnV) in MV_CONFIGS, mix in MV_MIXES
        @testset "$(D)D $(cname) $(mix)" begin
            rng = test_rng(100 * D + 7 * length(cname) + length(string(mix)))
            N = 64
            sys = mv_system(rng, D, N, ρs[D - 1], pot, sizes(rng, N))
            d = Displacement(δ)
            s = Swap()
            v = VolumeMove(δlnV)
            moves = if mix === :displacement
                (d,)
            elseif mix === :swap
                (s,)
            elseif mix === :volume
                (v,)
            else
                (d, s, v)
            end
            cond = (; kT=kT, pressure=P)
            sorted = sort(sys.diameters)
            bad_rejected = 0
            bad_cells = 0
            bad_displacement = 0
            bad_swap = 0
            bad_volume = 0
            energy_ok = true
            for k in 1:K
                move = moves[rand(rng, 1:length(moves))]
                snap = mv_snapshot(sys)
                accepted = SMC.attempt!(move, sys, cond, rng)
                if !accepted
                    bad_rejected += !mv_same_state(sys, snap)
                    # A rejected volume move may rebuild the cells (which replaces their
                    # storage); they must then still describe the same configuration.
                    if move isa VolumeMove && sys.cells.members !== snap.members_ref
                        bad_cells += !mv_cells_consistent(sys)
                    else
                        bad_cells += !mv_same_cells(sys, snap)
                    end
                elseif move isa Displacement
                    nchanged = count(i -> sys.frac[i] != snap.frac[i], 1:N)
                    bad_displacement += !(
                        nchanged <= 1 &&
                        all(mv_in_unit_cell, sys.frac) &&
                        isequal(sys.diameters, snap.diameters) &&
                        sys.box === snap.box
                    )
                elseif move isa Swap
                    nchanged = count(i -> sys.diameters[i] != snap.diameters[i], 1:N)
                    bad_swap += !(
                        isequal(sys.frac, snap.frac) &&
                        nchanged in (0, 2) &&
                        isequal(sort(sys.diameters), sorted) &&
                        sys.box === snap.box
                    )
                else
                    bad_volume += !(
                        isequal(sys.frac, snap.frac) &&
                        isequal(sys.diameters, snap.diameters) &&
                        SMC.fits_cutoff(sys.box, sys.rc)
                    )
                end
                if k % 500 == 0
                    tol = (; rtol=1e-10, atol=1e-10 * N)
                    recomputed = total_energy(sys)
                    energy_ok &= isapprox(sys.energy, recomputed; tol...)
                    energy_ok &= isapprox(recomputed, mv_reference_energy(sys); tol...)
                    energy_ok &= mv_cells_consistent(sys)
                end
            end
            @test energy_ok
            @test bad_rejected == 0
            @test bad_cells == 0
            @test bad_displacement == 0
            @test bad_swap == 0
            @test bad_volume == 0
            # The diameters are only ever permuted, bitwise.
            @test isequal(sort(sys.diameters), sorted)
            # Every move was attempted and accepted at least sometimes.
            for m in moves
                @test SMC.move_stats(m).accepts > 0
                @test SMC.move_stats(m).attempts >= K ÷ (2 * length(moves))
            end
        end
    end
end

@testset "hard spheres" begin
    for D in (2, 3)
        rng = test_rng(200 + D)
        N = 64
        σ = sample_diameters(rng, Bidisperse(1.0, 1.4, 0.5), N)
        box = box_for_packing_fraction(Val(D), σ, D == 2 ? 0.45 : 0.3)
        x = insert_hard_spheres(rng, box, σ)
        sys = System(box, x, σ, HardSphere())
        @test sys.energy === 0.0
        moves = (Displacement(0.2), Swap(), VolumeMove(0.02))
        # Reduced pressure P σ^D / kT of a moderately dense fluid.
        cond = (; kT=1.0, pressure=D == 2 ? 4.0 : 3.0)
        energy_zero = true
        no_overlaps = true
        for k in 1:5000
            SMC.attempt!(moves[rand(rng, 1:3)], sys, cond, rng)
            energy_zero &= sys.energy === 0.0
            if k % 100 == 0
                no_overlaps &= mv_reference_energy(sys) == 0
                no_overlaps &= total_energy(sys) == 0
            end
        end
        @test energy_zero
        @test no_overlaps
        for m in moves
            @test 0 < SMC.move_stats(m).accepts < SMC.move_stats(m).attempts
        end
    end
end

@testset "zero temperature quench" begin
    for D in (2, 3)
        rng = test_rng(300 + D)
        N = 64
        sys = mv_system(rng, D, N, 1.0, PolySoftSphere(), mv_poly(rng, N))
        moves = (Displacement(0.05), Swap())
        cond = (; kT=0.0)
        monotone = true
        for _ in 1:4000
            E = sys.energy
            SMC.attempt!(moves[rand(rng, 1:2)], sys, cond, rng)
            monotone &= sys.energy <= E
        end
        @test monotone
        @test all(m -> SMC.move_stats(m).accepts > 0, moves)
        @test sys.energy ≈ total_energy(sys) rtol = 1e-10

        # Swapping equal diameters changes nothing, so it is always accepted.
        mono = mv_system(rng, D, N, 1.0, PolySoftSphere(), fill(1.0, N))
        swap = Swap()
        E = mono.energy
        @test all(_ -> SMC.attempt!(swap, mono, cond, rng), 1:500)
        @test mono.energy === E
        @test SMC.move_stats(swap).accepts == 500
    end
end

@testset "swap edge cases" begin
    rng = test_rng(400)
    # Distinct diameters with max_dsigma = 0 are never swapped (but counted).
    sys = mv_system(rng, 2, 64, 0.5, PolySoftSphere(), mv_narrow(rng, 64))
    @test allunique(sys.diameters)
    swap = Swap(; max_dsigma=0.0)
    snap = mv_snapshot(sys)
    @test !any(_ -> SMC.attempt!(swap, sys, (; kT=1e6), rng), 1:1000)
    @test SMC.move_stats(swap).attempts == 1000
    @test SMC.move_stats(swap).accepts == 0
    @test mv_same_state(sys, snap)
    # A finite max_dsigma only allows swaps between similar diameters.
    swap = Swap(; max_dsigma=0.05)
    σ0 = copy(sys.diameters)
    too_far = 0
    for _ in 1:2000
        before = copy(sys.diameters)
        SMC.attempt!(swap, sys, (; kT=1.0), rng)
        changed = findall(sys.diameters .!= before)
        too_far += !(
            isempty(changed) || abs(before[changed[1]] - before[changed[2]]) <= 0.05
        )
    end
    @test too_far == 0
    @test SMC.move_stats(swap).accepts > 0
    @test sort(sys.diameters) == sort(σ0)
    @test_throws ArgumentError Swap(; max_dsigma=-1.0)
    @test SMC.name(swap) === :swap
    @test SMC.adaptive_step(swap) === nothing

    # A single particle cannot swap.
    single = System(
        CubicBox(Val(3), 5.0), [SVector(1.0, 2.0, 3.0)], [1.0], PolySoftSphere()
    )
    swap = Swap()
    @test !SMC.attempt!(swap, single, (; kT=1.0), rng)
    @test SMC.move_stats(swap).attempts == 1
end

@testset "volume move edge cases" begin
    @test_throws ArgumentError SMC.check_compatible(VolumeMove(), NVT(1.0))
    @test SMC.check_compatible(VolumeMove(), NPT(1.0, 1.0)) === nothing
    @test SMC.check_compatible(Displacement(), NVT(1.0)) === nothing
    @test SMC.check_compatible(Swap(), NPT(1.0, 1.0)) === nothing
    @test SMC.name(VolumeMove()) === :volume
    @test SMC.name(Displacement()) === :displacement

    # A box exactly twice the cutoff cannot shrink.
    for D in (2, 3)
        rng = test_rng(500 + D)
        pot = GaussianCore(; A=1.0, rc=3.0)
        box = CubicBox(Val(D), 6.0)
        sys = System(box, lattice_positions(box, 8), ones(8), pot)
        move = VolumeMove(0.2)
        cond = (; kT=1.0, pressure=0.0)
        shrunk = 0
        too_small = 0
        compressions = 0
        for _ in 1:400
            # The first random number of an attempt sets the sign of Δln V.
            compression = rand(copy(rng), Float64) < 0.5
            V = volume(sys)
            accepted = SMC.attempt!(move, sys, cond, rng)
            compressions += compression
            shrunk += accepted && volume(sys) < V
            too_small += !SMC.fits_cutoff(sys.box, sys.rc)
            # Bring the box back to the limit.
            if accepted
                sys.box = box
                SMC.rebuild_cells!(sys)
                SMC.sync_energy!(sys)
            end
        end
        @test compressions > 100
        @test shrunk == 0
        @test too_small == 0
        @test SMC.move_stats(move).accepts > 0
    end
end

@testset "allocations and type stability" begin
    for D in (2, 3), T in (Float64, Float32)
        rng = test_rng(600 + D)
        N = 216
        σ = mv_poly(rng, N)
        box = box_for_density(Val(D), N, 0.9)
        box = OrthorhombicBox(T.(box.lengths))
        x = lattice_positions(box, N)
        sys = System(box, x, σ, PolySoftSphere())
        cond = (; kT=T(0.3), pressure=T(8))
        moves = (Displacement(0.05), Swap(), VolumeMove(1e-3))
        for _ in 1:2000, m in moves
            SMC.attempt!(m, sys, cond, rng)
        end
        @test sys.energy isa SMC.energy_type(T)
        @test sys.energy ≈ total_energy(sys) rtol = (T == Float32 ? 1e-3 : 1e-10)
        for m in moves
            @test allocs(SMC.attempt!, m, sys, cond, rng) == 0
            @test (@inferred SMC.attempt!(m, sys, cond, rng)) isa Bool
        end
        # Moves that do not need the pressure also work in NVT.
        nvt = (; kT=T(0.3))
        @test allocs(SMC.attempt!, moves[1], sys, nvt, rng) == 0
        @test (@inferred SMC.attempt!(moves[2], sys, nvt, rng)) isa Bool
    end
end

@testset "step size adaptation" begin
    rng = test_rng(700)
    box = CubicBox(Val(2), 10.0)
    ideal = System(box, lattice_positions(box, 16), ones(16), Ideal())
    # Ideal gas: every displacement is accepted, so the step grows by 1.5 per window.
    m = Displacement(0.1; interval=100, bounds=(1e-4, 0.3))
    @test SMC.adaptive_step(m) === m.step
    for _ in 1:99
        SMC.attempt!(m, ideal, (; kT=1.0), rng)
    end
    SMC.adapt!(m)
    @test SMC.step_size(m) == 0.1
    SMC.attempt!(m, ideal, (; kT=1.0), rng)
    SMC.adapt!(m)
    @test SMC.step_size(m) ≈ 0.15
    for _ in 1:5
        old = SMC.step_size(m)
        for _ in 1:100
            SMC.attempt!(m, ideal, (; kT=1.0), rng)
        end
        SMC.adapt!(m)
        @test old <= SMC.step_size(m) <= 1.5 * old * (1 + 1e-12)
        @test SMC.step_size(m) <= 0.3
    end
    @test SMC.step_size(m) == 0.3

    # Dense soft spheres at a tiny temperature with a large step: almost every
    # displacement is rejected, so the step shrinks by at most 1.5 per window.
    rng = test_rng(701)
    sys = mv_system(rng, 2, 64, 1.2, PolySoftSphere(), mv_poly(rng, 64))
    m = Displacement(0.5; interval=200, bounds=(0.05, Inf))
    for _ in 1:8
        old = SMC.step_size(m)
        for _ in 1:200
            SMC.attempt!(m, sys, (; kT=1e-3), rng)
        end
        SMC.adapt!(m)
        @test old / 1.5 * (1 - 1e-12) <= SMC.step_size(m) <= old
        @test SMC.step_size(m) >= 0.05
    end
    @test SMC.step_size(m) == 0.05

    # Volume moves of an ideal gas at zero pressure are accepted with probability
    # min(1, e^{(N+1)Δln V}), above one half; the step grows.
    v = VolumeMove(0.01; interval=50, target=0.3)
    for _ in 1:50
        SMC.attempt!(v, ideal, (; kT=1.0, pressure=0.0), rng)
    end
    SMC.adapt!(v)
    @test SMC.step_size(v) ≈ 0.015
    @test_throws ArgumentError Displacement(-0.1)
    @test_throws ArgumentError VolumeMove(2.0)
end
