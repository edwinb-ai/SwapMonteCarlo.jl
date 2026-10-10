# Tests of the simulation driver: construction, sweeps, determinism, ramps, adaptation.

using Random: randexp

# --------------------------------------------------------------------------------------
# File-local test moves, which exercise the MCMove interface independently of the real
# moves.
# --------------------------------------------------------------------------------------

"""
Metropolis displacement of one particle, written against the documented move contract.
"""
struct NudgeMove <: SMC.MCMove
    step::SMC.AdaptiveStep
    stats::SMC.MoveStats
end
function NudgeMove(δ=0.1)
    return NudgeMove(
        SMC.AdaptiveStep(δ; target=0.4, bounds=(1e-4, 1.0), interval=50), SMC.MoveStats()
    )
end
SMC.name(::NudgeMove) = :nudge
SMC.adaptive_step(m::NudgeMove) = m.step
function SMC.attempt!(m::NudgeMove, sys::SMC.System{D,T}, cond, rng) where {D,T}
    i = rand(rng, 1:length(sys))
    δ = T(m.step.value)
    dx = SVector{D,T}(ntuple(_ -> δ * (2 * rand(rng, T) - 1), Val(D)))
    s_new = SMC.wrap_fractional(sys.frac[i] + SMC.fractional(sys.box, dx))
    e_old = SMC.particle_energy(sys, i)
    emax = e_old + T(cond.kT) * randexp(rng, T)
    e_new = SMC.particle_energy(sys, s_new, sys.diameters[i], i, 0, emax)
    accepted = e_new <= emax
    if accepted
        sys.frac[i] = s_new
        SMC.cell_move!(sys.cells, i, s_new)
        sys.energy += e_new - e_old
    end
    return SMC.record!(m, accepted)
end

"""
Rejects every attempt and records the temperature it was offered.
"""
struct RecorderMove <: SMC.MCMove
    kTs::Vector{Float64}
    stats::SMC.MoveStats
end
RecorderMove() = RecorderMove(Float64[], SMC.MoveStats())
SMC.name(::RecorderMove) = :recorder
function SMC.attempt!(m::RecorderMove, sys, cond, rng)
    push!(m.kTs, cond.kT)
    return SMC.record!(m, false)
end

"""
A move that claims to be a volume move but does nothing.
"""
struct FakeVolume <: SMC.MCMove
    stats::SMC.MoveStats
end
FakeVolume() = FakeVolume(SMC.MoveStats())
SMC.name(::FakeVolume) = :volume
SMC.attempt!(m::FakeVolume, sys, cond, rng) = SMC.record!(m, false)

"""
    sim_soft_system(rng; N=64, D=2, ρ=0.8)

Small polydisperse soft-sphere system on a lattice.
"""
function sim_soft_system(rng; N=64, D=2, ρ=0.8)
    σ = sample_diameters(rng, UniformSizes(0.85, 1.15), N)
    box = box_for_density(Val(D), N, ρ)
    return System(box, lattice_positions(box, N), σ, PolySoftSphere())
end

"""
    sim_standard(seed; D=2)

A soft-sphere simulation with displacement and swap moves.
"""
function sim_standard(seed; D=2, kT=1.0)
    sys = sim_soft_system(StableRNG(99); D=D)
    return Simulation(sys, NVT(kT), Displacement(0.1) => 0.8, Swap() => 0.2; seed=seed)
end

"""
    same_state(a, b) -> Bool

Whether two simulations are in bitwise identical states.
"""
function same_state(a, b)
    return a.system.frac == b.system.frac &&
           a.system.diameters == b.system.diameters &&
           a.system.box == b.system.box &&
           a.system.energy === b.system.energy &&
           a.sweep == b.sweep
end

@testset "select_move and attempt_nth!" begin
    cw = (0.25, 0.75, 1.0)
    @test SMC.select_move(cw, 0.0) == 1
    @test SMC.select_move(cw, 0.2499) == 1
    @test SMC.select_move(cw, 0.25) == 2
    @test SMC.select_move(cw, 0.7499) == 2
    @test SMC.select_move(cw, 0.75) == 3
    @test SMC.select_move(cw, prevfloat(1.0)) == 3
    @test SMC.select_move((1.0,), 0.5) == 1
    @inferred SMC.select_move(cw, 0.3)

    sys = sim_soft_system(test_rng(1))
    rec = RecorderMove()
    moves = (NudgeMove(), rec)
    @test !SMC.attempt_nth!(moves, 2, sys, (; kT=0.5), Xoshiro(1))
    @test rec.kTs == [0.5]
    @test moves[1].stats.attempts == 0
    SMC.attempt_nth!(moves, 1, sys, (; kT=0.5), Xoshiro(1))
    @test moves[1].stats.attempts == 1
    @inferred SMC.attempt_nth!(moves, 1, sys, (; kT=0.5), Xoshiro(1))
end

@testset "constructor validation" begin
    sys = sim_soft_system(test_rng(2))
    # Weights.
    for bad in (0, -1.0, Inf, NaN)
        @test_throws ArgumentError Simulation(sys, NVT(1.0), NudgeMove() => bad; seed=1)
    end
    @test_throws ArgumentError Simulation(
        sys, NVT(1.0), NudgeMove() => 1, RecorderMove() => 0; seed=1
    )
    @test_throws ArgumentError Simulation(sys, NVT(1.0); seed=1)
    @test_throws ArgumentError Simulation(sys, NVT(1.0), NudgeMove() => 1; seed=-1)
    # Unique names.
    @test_throws ArgumentError Simulation(
        sys, NVT(1.0), NudgeMove() => 1, NudgeMove() => 1; seed=1
    )
    @test_throws ArgumentError Simulation(
        sys, NVT(1.0), Displacement() => 1, Displacement(0.2) => 1; seed=1
    )
    # Volume moves and NPT.
    @test_throws ArgumentError Simulation(sys, NPT(1.0, 1.0), NudgeMove() => 1; seed=1)
    @test_throws ArgumentError Simulation(sys, NPT(1.0, 1.0), Displacement() => 1; seed=1)
    @test_throws ArgumentError Simulation(
        sys, NVT(1.0), Displacement() => 1, FakeVolume() => 0.01; seed=1
    )
    @test_throws ArgumentError Simulation(
        sys, NVT(1.0), Displacement() => 1, VolumeMove() => 0.01; seed=1
    )
    sim = Simulation(sys, NPT(1.0, 1.0), Displacement() => 1, VolumeMove() => 0.01; seed=1)
    @test sim isa Simulation
    @test sim.sweep == 0
    # Swaps of identical particles are useless.
    box = box_for_density(Val(2), 16, 0.5)
    mono = System(box, lattice_positions(box, 16), fill(1.0, 16), PolySoftSphere())
    @test_logs (:warn, r"swap") Simulation(mono, NVT(1.0), Swap() => 1; seed=1)
    @test_logs Simulation(sys, NVT(1.0), Swap() => 1; seed=1)
    # Seeds and random number generators.
    sim = Simulation(sys, NVT(1.0), NudgeMove() => 1; seed=42)
    @test sim.seed === UInt64(42)
    @test sim.rng == Xoshiro(42)
    @test Simulation(sys, NVT(1.0), NudgeMove() => 1).seed isa UInt64
    rng = StableRNG(3)
    @test Simulation(sys, NVT(1.0), NudgeMove() => 1; rng=rng).rng === rng
    @test occursin("Simulation(", sprint(show, sim))
end

@testset "weights are normalized" begin
    sys = sim_soft_system(test_rng(3))
    sim = Simulation(sys, NVT(1.0), NudgeMove() => 3, RecorderMove() => 1; seed=1)
    @test sim.cumweights == (0.75, 1.0)
    sim = Simulation(
        sys, NVT(1.0), Displacement() => 0.1, Swap() => 0.2, NudgeMove() => 0.3; seed=1
    )
    @test sim.cumweights isa NTuple{3,Float64}
    @test sim.cumweights[3] === 1.0
    @test collect(sim.cumweights) ≈ [1 / 6, 1 / 2, 1]
    @test issorted(sim.cumweights)
    # The empirical move frequencies follow the weights.
    rec = RecorderMove()
    sim = Simulation(sys, NVT(1.0), NudgeMove(0.05) => 3, rec => 1; seed=7)
    run!(sim, 50)
    @test isapprox(length(rec.kTs) / (50 * length(sys)), 0.25; atol=0.03)
end

@testset "run! and equilibrate! with the test move" begin
    sys = sim_soft_system(test_rng(4))
    sim = Simulation(sys, NVT(1.0), NudgeMove() => 1; seed=5)
    @test run!(sim, 10) === sim
    @test sim.sweep == 10
    @test equilibrate!(sim, 5) === sim
    @test sim.sweep == 15
    @test run!(sim, 0).sweep == 15
    @test_throws ArgumentError run!(sim, -1)
    @test_throws ArgumentError run!(sim, 1; energy_check=-1)
    @test sim.moves[1].stats.attempts == 15 * length(sys)
    (running, recomputed) = SMC.check_energy(sys)
    @test running ≈ recomputed rtol = 1e-10
end

@testset "energy stays consistent ($(D)D)" for D in (2, 3)
    sim = sim_standard(11; D=D)
    E0 = sim.system.energy
    run!(sim, 30; energy_check=0)
    (running, recomputed) = SMC.check_energy(sim.system)
    @test running ≈ recomputed rtol = 1e-10
    @test sim.system.energy != E0
    @test sim.moves[1].stats.accepts > 0
    @test sim.moves[2].stats.accepts > 0
end

@testset "NPT runs" begin
    sys = sim_soft_system(test_rng(5); N=64, ρ=0.7)
    sim = Simulation(
        sys, NPT(1.0, 5.0), Displacement(0.1) => 1, VolumeMove(0.02) => 0.02; seed=12
    )
    V0 = volume(sys)
    run!(sim, 40)
    @test volume(sys) != V0
    @test sim.moves[2].stats.attempts > 0
    (running, recomputed) = SMC.check_energy(sys)
    @test running ≈ recomputed rtol = 1e-10
end

@testset "energy check resynchronizes" begin
    sim = sim_standard(13)
    run!(sim, 2)
    sim.system.energy += 1.0
    @test_logs (:warn, r"running energy") run!(sim, 1; energy_check=1)
    (running, recomputed) = SMC.check_energy(sim.system)
    @test running == recomputed
    # The check happens on multiples of `energy_check`, counted from sweep 0.
    sim.system.energy += 1.0
    @test_logs run!(sim, 1; energy_check=5)
    @test_logs (:warn, r"running energy") run!(sim, 2; energy_check=5)
    @test_logs run!(sim, 1; energy_check=0)
end

@testset "determinism" begin
    a = sim_standard(2024)
    b = sim_standard(2024)
    c = sim_standard(2025)
    for sim in (a, b, c)
        run!(sim, 20)
    end
    @test same_state(a, b)
    @test a.rng == b.rng
    @test acceptance_rate(a) === acceptance_rate(b)
    @test !same_state(a, c)
    # Splitting a run does not change it.
    d = sim_standard(2024)
    run!(d, 7)
    run!(d, 13)
    @test same_state(a, d)
end

@testset "temperature ramps are evaluated per sweep" begin
    sys = sim_soft_system(test_rng(6); N=16)
    ramp = LinearRamp(2.0, 0.5, 10)
    rec = RecorderMove()
    sim = Simulation(sys, NVT(ramp), rec => 1; seed=1)
    run!(sim, 12)
    @test length(rec.kTs) == 12 * 16
    for sweep in 1:12
        @test all(==(ramp(sweep)), rec.kTs[((sweep - 1) * 16 + 1):(sweep * 16)])
    end
    # Pressure ramps in NPT are evaluated the same way.
    seen = Float64[]
    sim = Simulation(
        sim_soft_system(test_rng(7); N=16, ρ=0.6),
        NPT(1.0, ExponentialRamp(1.0, 4.0, 4)),
        Displacement() => 1,
        VolumeMove() => 0.1;
        seed=2,
    )
    run!(
        sim,
        6;
        observers=Callback(
            s -> push!(seen, SMC.conditions(s.ensemble, s.sweep, Float64).pressure)
        ),
    )
    @test seen ≈ [ExponentialRamp(1.0, 4.0, 4)(k) for k in 1:6]
    # A quench at zero temperature never increases the energy.
    sim = Simulation(
        sim_soft_system(test_rng(8)), NVT(0.0), Displacement(0.05) => 1; seed=3
    )
    energies = Float64[]
    run!(sim, 20; observers=Callback(s -> push!(energies, s.system.energy)))
    @test issorted(energies; rev=true)
end

@testset "ensemble validation" begin
    # Constant temperatures and pressures are checked by the constructors.
    @test NVT(0.0) isa NVT
    for kT in (Inf, -Inf, NaN, -1.0)
        @test_throws ArgumentError NVT(kT)
        @test_throws ArgumentError NPT(kT, 1.0)
    end
    @test_throws ArgumentError NPT(0.0, 1.0)
    @test_throws ArgumentError NPT(1.0, Inf)
    @test_throws ArgumentError NPT(1.0, NaN)
    @test NPT(1.0, -1.0) isa NPT
    # Callables are checked at every sweep, after the conversion to the system's type.
    ramp = NVT(LinearRamp(1.0, -1.0, 10))
    @test SMC.conditions(ramp, 5, Float64).kT == 0
    @test_throws ArgumentError SMC.conditions(ramp, 6, Float64)
    msg = try
        SMC.conditions(ramp, 6, Float64)
    catch err
        sprint(showerror, err)
    end
    @test occursin("sweep 6", msg) && occursin("temperature", msg)
    @test_throws ArgumentError SMC.conditions(NVT(s -> 1e40), 1, Float32)
    @test_throws ArgumentError SMC.conditions(NPT(LinearRamp(1.0, 0.0, 4), 1.0), 4, Float64)
    @test_throws ArgumentError SMC.conditions(NPT(1.0, s -> s > 2 ? Inf : 1.0), 3, Float64)
    @test SMC.conditions(NPT(1.0, s -> s > 2 ? Inf : 1.0), 2, Float64) ==
        (; kT=1.0, pressure=1.0)
    # A run stops at the first invalid sweep.
    sim = Simulation(
        sim_soft_system(test_rng(11); N=16), ramp, Displacement(0.1) => 1; seed=1
    )
    @test_throws ArgumentError run!(sim, 20)
    @test sim.sweep == 6
end

@testset "acceptance rates and step sizes" begin
    sim = sim_standard(17)
    acc = acceptance_rate(sim)
    @test keys(acc) == (:displacement, :swap)
    @test all(isnan, acc)
    run!(sim, 10)
    acc = acceptance_rate(sim)
    @test 0 < acc.displacement < 1
    @test 0 < acc.swap < 1
    @test acc.displacement == SMC.acceptance_rate(sim.moves[1])
    steps = step_size(sim)
    @test keys(steps) == (:displacement, :swap)
    @test steps.displacement == 0.1
    @test isnan(steps.swap)
    @test reset_stats!(sim) === sim
    @test all(isnan, acceptance_rate(sim))
    @test sim.moves[1].stats.attempts == 0
end

@testset "adaptation only in equilibrate!" begin
    sim = Simulation(
        sim_soft_system(test_rng(9); ρ=0.3),
        NVT(1.0),
        Displacement(0.01; interval=64) => 1,
        Swap() => 0.1;
        seed=8,
    )
    run!(sim, 20)
    @test step_size(sim).displacement == 0.01
    equilibrate!(sim, 20)
    # Nearly every tiny step is accepted, so the step grows.
    @test step_size(sim).displacement > 0.01
    δ = step_size(sim).displacement
    run!(sim, 20)
    @test step_size(sim).displacement == δ
    # The test move adapts too.
    sim = Simulation(sim_soft_system(test_rng(10); ρ=0.3), NVT(1.0), NudgeMove(1e-3) => 1)
    equilibrate!(sim, 10)
    @test step_size(sim).nudge > 1e-3
end

@testset "sweep! is type stable and allocation free" begin
    sim = sim_standard(21)
    run!(sim, 2)
    sys = sim.system
    cond = SMC.conditions(sim.ensemble, sim.sweep, Float64)
    args = (sys, sim.moves, sim.cumweights, cond, sim.rng)
    @inferred SMC.sweep!(args...)
    @test allocs(SMC.sweep!, args...) == 0
    # Also with the test moves and a single move (no random number for the selection).
    sim = Simulation(sim_soft_system(test_rng(11)), NVT(1.0), NudgeMove() => 1; seed=4)
    args = (sim.system, sim.moves, sim.cumweights, (; kT=1.0), sim.rng)
    @inferred SMC.sweep!(args...)
    @test allocs(SMC.sweep!, args...) == 0
    # The whole loop without observers does not allocate per sweep either.
    sim = sim_standard(22)
    run!(sim, 2; energy_check=0)
    a1 = @allocations run!(sim, 1; energy_check=0)
    a2 = @allocations run!(sim, 21; energy_check=0)
    @test a2 == a1
end
