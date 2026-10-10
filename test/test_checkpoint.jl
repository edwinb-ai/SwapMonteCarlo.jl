# Tests of checkpoints: a restored simulation continues bitwise identically.

using Serialization: serialize

"""
    ckpt_sim(seed; npt=false)

Small polydisperse soft-sphere simulation in NVT (displacement and swap moves) or NPT
(also volume moves).
"""
function ckpt_sim(seed; npt=false)
    rng = StableRNG(seed)
    N = 48
    σ = sample_diameters(rng, UniformSizes(0.85, 1.15), N)
    box = box_for_density(Val(2), N, 0.75)
    sys = System(box, lattice_positions(box, N), σ, PolySoftSphere())
    if npt
        return Simulation(
            sys,
            NPT(LinearRamp(1.0, 0.5, 30), 5.0),
            Displacement(0.1) => 1,
            Swap() => 0.2,
            VolumeMove(0.02) => 0.05;
            seed=seed,
        )
    end
    return Simulation(sys, NVT(1.0), Displacement(0.1) => 1, Swap() => 0.2; seed=seed)
end

"""
    identical(a, b) -> Bool

Whether two simulations are in bitwise identical states, including move statistics, step
sizes and the random number generator.
"""
function identical(a, b)
    sa = a.system
    sb = b.system
    return sa.frac == sb.frac &&
           sa.diameters == sb.diameters &&
           sa.box == sb.box &&
           sa.energy === sb.energy &&
           a.sweep == b.sweep &&
           a.seed == b.seed &&
           a.rng == b.rng &&
           acceptance_rate(a) === acceptance_rate(b) &&
           step_size(a) === step_size(b)
end

@testset "checkpoint round trip ($(npt ? "NPT" : "NVT"))" for npt in (false, true)
    n = 10
    mktempdir() do dir
        path = joinpath(dir, "state.jls")
        reference = ckpt_sim(31; npt=npt)
        equilibrate!(reference, n)
        run!(reference, n)

        sim = ckpt_sim(31; npt=npt)
        equilibrate!(sim, n)
        @test save_checkpoint(path, sim) === nothing
        @test isfile(path)
        @test !isfile(path * ".tmp")
        restored = load_checkpoint(path)
        @test restored isa Simulation
        @test restored !== sim
        @test identical(restored, sim)
        # The restored cell list is valid and the running energy consistent.
        @test SMC.check_energy(restored.system)[1] == restored.system.energy
        @test restored.system.energy ≈ total_energy(restored.system) rtol = 1e-10

        run!(restored, n)
        @test identical(restored, reference)
        @test rand(copy(restored.rng), 4) == rand(copy(reference.rng), 4)
        # The original simulation is unaffected by saving.
        run!(sim, n)
        @test identical(sim, reference)

        # Saving again overwrites the file.
        save_checkpoint(path, restored)
        @test load_checkpoint(path).sweep == 2n
    end
end

@testset "invalid checkpoints" begin
    mktempdir() do dir
        path = joinpath(dir, "bad.jls")
        open(io -> serialize(io, 42), path, "w")
        @test_throws ArgumentError load_checkpoint(path)
        open(io -> serialize(io, (; format=999, simulation=1)), path, "w")
        @test_throws ArgumentError load_checkpoint(path)
        # Checkpoints from other versions load with a warning.
        sim = ckpt_sim(32)
        data = (;
            format=SMC.CHECKPOINT_FORMAT,
            julia_version=v"1.0.0",
            package_version=pkgversion(SwapMonteCarlo),
            simulation=sim,
        )
        open(io -> serialize(io, data), path, "w")
        restored = @test_logs (:warn, r"different version") load_checkpoint(path)
        @test identical(restored, sim)
    end
end
