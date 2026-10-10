# Code quality: Aqua checks, concrete field types, JET static analysis (errors and runtime
# dispatch in the hot paths) and allocation-free sweeps.

using Aqua: Aqua
using JET: JET
using JET: @test_opt
using SwapMonteCarlo: Displacement, Swap, VolumeMove, attempt!

"""
    quality_system(D, potential; N=64) -> System

Small `D`-dimensional system of `N` polydisperse particles on a lattice at a moderate
density, valid (no overlaps) for every built-in potential.
"""
function quality_system(D::Int, potential; N::Int=64)
    rng = test_rng(900 + D)
    σ = 0.9 .+ 0.2 .* rand(rng, N)
    box = SMC.box_for_density(Val(D), N, D == 2 ? 0.5 : 0.4)
    return System(box, SMC.lattice_positions(box, N), σ, potential)
end

"""Built-in potentials with the keyword arguments used in the quality checks."""
const QUALITY_POTENTIALS = (
    PolySoftSphere(),
    PolySoftSphere(; mixing=Additive()),
    LennardJones(; xc=2.0),
    LennardJones(; xc=2.0, shift=ForceShift(), mixing=NonAdditive(0.1)),
    HardSphere(),
    PseudoHS(),
)

"""
    all_fields_concrete(x) -> Bool

Whether every field type of the type of `x` is concrete.
"""
function all_fields_concrete(x)
    T = typeof(x)
    return all(isconcretetype(fieldtype(T, f)) for f in fieldnames(T))
end

@testset "Aqua" begin
    Aqua.test_all(SwapMonteCarlo; persistent_tasks=false)
end

@testset "concrete field types" begin
    for D in (2, 3), pot in QUALITY_POTENTIALS
        sys = quality_system(D, pot)
        @test all_fields_concrete(sys)
        @test all_fields_concrete(sys.cells)
        @test all_fields_concrete(sys.box)
        @test all_fields_concrete(sys.potential)
        moves = (Displacement() => 0.8, Swap() => 0.2)
        @test all_fields_concrete(Simulation(sys, NVT(1.0), moves...; seed=1))
        npt = Simulation(sys, NPT(1.0, 1.0), moves..., VolumeMove() => 0.01; seed=1)
        @test all_fields_concrete(npt)
    end
    ramp = Simulation(
        quality_system(2, PolySoftSphere()),
        NVT(LinearRamp(1.0, 0.1, 10)),
        Displacement() => 1;
        seed=1,
    )
    @test all_fields_concrete(ramp)
end

@testset "JET" begin
    JET.test_package(SwapMonteCarlo; target_modules=(SwapMonteCarlo,))

    for D in (2, 3), pot in (PolySoftSphere(), LennardJones(; xc=2.0), HardSphere())
        sys = quality_system(D, pot)
        rng = Xoshiro(1)
        nvt = (; kT=1.0)
        npt = (; kT=1.0, pressure=1.0)
        @test_opt target_modules = (SwapMonteCarlo,) attempt!(Displacement(), sys, nvt, rng)
        @test_opt target_modules = (SwapMonteCarlo,) attempt!(Swap(), sys, nvt, rng)
        @test_opt target_modules = (SwapMonteCarlo,) attempt!(VolumeMove(), sys, npt, rng)
        @test_opt target_modules = (SwapMonteCarlo,) total_energy(sys)
        @test_opt target_modules = (SwapMonteCarlo,) SMC.particle_energy(sys, 1)
        moves = (Displacement(), Swap(), VolumeMove())
        cumweights = (0.8, 0.99, 1.0)
        @test_opt target_modules = (SwapMonteCarlo,) SMC.sweep!(
            sys, moves, cumweights, npt, rng
        )
    end
end

@testset "allocation-free sweeps" begin
    for D in (2, 3), pot in QUALITY_POTENTIALS
        sys = quality_system(D, pot; N=200)
        rng = Xoshiro(2)
        nvt = (; kT=1.0)
        moves = (Displacement(0.05), Swap())
        @test allocs(SMC.sweep!, sys, moves, (0.8, 1.0), nvt, rng) == 0
        # Volume steps far below the cell skin never rebuild the cells, the only case in
        # which a volume move allocates.
        npt = (; kT=1.0, pressure=1.0)
        moves = (Displacement(0.05), Swap(), VolumeMove(1e-5; bounds=(1e-6, 1.0)))
        @test allocs(SMC.sweep!, sys, moves, (0.79, 0.99, 1.0), npt, rng) == 0
    end
    # A dense 3D soft sphere system, whose crowded cells switch the cell list to range 2.
    box = SMC.box_for_density(Val(3), 500, 1.0)
    σ = sample_diameters(test_rng(910), InversePowerLaw(0.73, 1.62), 500)
    sys = System(box, SMC.lattice_positions(box, 500), σ, PolySoftSphere())
    @test sys.cells.range == 2
    moves = (Displacement(0.05), Swap())
    @test allocs(SMC.sweep!, sys, moves, (0.8, 1.0), (; kT=0.1), Xoshiro(3)) == 0
end
