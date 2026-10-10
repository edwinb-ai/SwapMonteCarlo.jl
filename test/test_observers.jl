# Tests of schedules, quantities and observers.

"""
    obs_soft_sim(seed; N=64, D=2, ρ=0.8)

Small soft-sphere simulation with displacement and swap moves.
"""
function obs_soft_sim(seed; N=64, D=2, ρ=0.8, kT=1.0)
    rng = StableRNG(seed)
    σ = sample_diameters(rng, UniformSizes(0.85, 1.15), N)
    box = box_for_density(Val(D), N, ρ)
    sys = System(box, lattice_positions(box, N), σ, PolySoftSphere())
    return Simulation(sys, NVT(kT), Displacement(0.1) => 0.9, Swap() => 0.1; seed=seed)
end

"""
    obs_hard_sim(seed, φ; N=100, D=3)

Hard sphere simulation at packing fraction `φ`, from random sequential insertion.
"""
function obs_hard_sim(seed, φ; N=100, D=3)
    rng = StableRNG(seed)
    σ = sample_diameters(rng, UniformSizes(0.9, 1.1), N)
    box = box_for_packing_fraction(Val(D), σ, φ)
    x = insert_hard_spheres(rng, box, σ)
    sys = System(box, x, σ, HardSphere())
    return Simulation(sys, NVT(1.0), Displacement(0.05) => 1; seed=seed)
end

"""
    brute_compression_overlaps(sys, xi) -> Int

Number of overlapping pairs after scaling all box lengths and positions by `(1 -
xi)^(1/D)`, counted over all pairs with the minimum image convention (additive hard
spheres).
"""
function brute_compression_overlaps(sys, xi)
    D = length(sys.box.lengths)
    f = (1 - xi)^(1 / D)
    L = sys.box.lengths .* f
    x = [p .* f for p in positions(sys)]
    σ = diameters(sys)
    n = 0
    for i in eachindex(x), j in (i + 1):lastindex(x)
        d = min_image_brute(x[i] - x[j], L)
        n += dot(d, d) < ((σ[i] + σ[j]) / 2)^2
    end
    return n
end

"""
User-defined observer that counts its calls.
"""
struct CountingObserver <: SMC.Observer
    count::Base.RefValue{Int}
end
function SMC.observe!(o::CountingObserver, sim)
    o.count[] += 1
    return nothing
end

@testset "Every" begin
    s = Every(10)
    @test SMC.isdue(s, 10)
    @test SMC.isdue(s, 20)
    @test !SMC.isdue(s, 5)
    @test !SMC.isdue(s, 11)
    @test all(k -> SMC.isdue(Every(1), k), 1:5)
    @test_throws ArgumentError Every(0)
    @test Every(3) isa SMC.Schedule
end

@testset "LogSchedule" begin
    s = LogSchedule()
    due = [k for k in 0:100_000 if SMC.isdue(s, k)]
    @test due[1:8] == [1, 2, 3, 5, 7, 10, 14, 19]
    @test all(diff(due) .> 0)
    ratios = due[2:end] ./ due[1:(end - 1)]
    # Roughly geometric: the ratios approach the base for large sweeps.
    @test all(r -> 1.35 - 1e-12 <= r <= 2, ratios)
    @test all(r -> r < 1.36, ratios[(end - 10):end])
    @test length(due) < 2 * log(100_000) / log(1.35)
    # Repeated and backward queries.
    @test SMC.isdue(s, 19) && SMC.isdue(s, 19)
    @test SMC.isdue(s, 5)
    @test !SMC.isdue(s, 4)
    @test SMC.isdue(s, 10)
    s = LogSchedule(; base=2, first=10)
    @test [k for k in 1:200 if SMC.isdue(s, k)] == [10, 20, 40, 80, 160]
    s = LogSchedule(; base=1.01, first=1)
    @test [k for k in 1:6 if SMC.isdue(s, k)] == 1:6
    @test_throws ArgumentError LogSchedule(; base=1.0)
    @test_throws ArgumentError LogSchedule(; first=-1)
end

@testset "TimeSeries and Callback" begin
    sim = obs_soft_sim(1)
    ts = TimeSeries(PotentialEnergy(), Density(), Volume(); schedule=Every(5))
    seen = Tuple{Int,Float64}[]
    cb = Callback(s -> push!(seen, (s.sweep, s.system.energy / length(s.system))))
    run!(sim, 20; observers=[ts, cb])
    @test ts.sweeps == [5, 10, 15, 20]
    @test ts.labels == ["energy", "density", "volume"]
    @test length(ts.values) == 3
    @test ts.values[1] == [e for (k, e) in seen if k % 5 == 0]
    @test ts["energy"] === ts.values[1]
    @test all(==(density(sim.system)), ts["density"])
    @test all(==(volume(sim.system)), ts["volume"])
    @test_throws KeyError ts["nope"]
    @test first.(seen) == 1:20
    # A single observer works as well, and the series continues.
    run!(sim, 5; observers=ts)
    @test ts.sweeps == [5, 10, 15, 20, 25]
    @test_throws ArgumentError TimeSeries()
    @test_throws ArgumentError TimeSeries(PotentialEnergy(), 3)
    @test_throws ArgumentError run!(sim, 1; observers=(ts, 1))
end

@testset "quantities" begin
    sim = obs_soft_sim(2)
    run!(sim, 5)
    sys = sim.system
    @test SMC.measure(PotentialEnergy(), sim) == sys.energy / length(sys)
    @test SMC.measure(Pressure(), sim) == pressure(sys, 1.0)
    @test SMC.measure(Density(), sim) == density(sys)
    @test SMC.measure(PackingFraction(), sim) == packing_fraction(sys)
    @test SMC.measure(Volume(), sim) == volume(sys)
    @test SMC.measure(Acceptance(:displacement), sim) == acceptance_rate(sim).displacement
    @test SMC.measure(Acceptance(:swap), sim) == acceptance_rate(sim).swap
    @test SMC.measure(StepSize(:displacement), sim) == 0.1
    @test isnan(SMC.measure(StepSize(:swap), sim))
    @test_throws ArgumentError SMC.measure(Acceptance(:volume), sim)
    @test SMC.label(Acceptance(:swap)) == "acceptance_swap"
    @test SMC.label(StepSize(:displacement)) == "step_displacement"
    @test SMC.label(Pressure()) == "pressure"
    for q in (PotentialEnergy(), Pressure(), Density(), Volume(), Acceptance(:swap))
        @test SMC.measure(q, sim) isa Float64
    end
    # The pressure uses the temperature of the current sweep.
    sys2 = obs_soft_sim(2).system
    sim2 = Simulation(sys2, NVT(LinearRamp(1.0, 3.0, 10)), Displacement() => 1; seed=1)
    run!(sim2, 5)
    @test SMC.measure(Pressure(), sim2) == pressure(sys2, 2.0)
    # Hard potentials have no virial pressure, soft ones no virtual pressure.
    @test_throws ArgumentError SMC.measure(VirtualPressure(), sim)
    @test_throws ArgumentError SMC.measure(Pressure(), obs_hard_sim(3, 0.3))
    @test_throws ArgumentError VirtualPressure(; xi=0)
    @test_throws ArgumentError VirtualPressure(; xi=1)
end

@testset "VirtualPressure ($(D)D)" for D in (2, 3)
    φ = D == 3 ? 0.35 : 0.5
    sim = obs_hard_sim(4 + D, φ; N=D == 3 ? 150 : 100, D=D)
    run!(sim, 10)
    sys = sim.system
    N = length(sys)
    V = volume(sys)
    # The compression factor is below the cell skin for the smallest xi and above it for
    # the largest one, which uses a temporary cell list.
    for xi in (1e-3, 0.01, 0.03, 0.1, 0.35)
        n = brute_compression_overlaps(sys, xi)
        @test SMC.compression_overlaps(sys, xi) == n
        @test SMC.measure(VirtualPressure(; xi=xi), sim) ≈ N / V + n / (xi * V)
    end
    @test brute_compression_overlaps(sys, 0.1) > 0
    @test !SMC.cells_cover(sys.cells, sys.box, sys.rc / (1 - 0.35)^(1 / D))
    @test SMC.cells_cover(sys.cells, sys.box, sys.rc)
    # Temperature enters linearly.
    sim2 = Simulation(sys, NVT(2.5), Displacement() => 1; seed=1)
    q = VirtualPressure(; xi=0.05)
    @test SMC.measure(q, sim2) ≈ 2.5 * SMC.measure(q, sim)
    # At very low density the pressure is ideal.
    dilute = obs_hard_sim(9, 0.005; N=50, D=D)
    @test SMC.compression_overlaps(dilute.system, 1e-3) == 0
    @test SMC.measure(VirtualPressure(), dilute) ≈ density(dilute.system)
    # Compressions that break the minimum image convention are refused.
    tiny = obs_hard_sim(10, 0.1; N=8, D=D)
    @test_throws ArgumentError SMC.measure(VirtualPressure(; xi=0.99), tiny)
end

@testset "ThermoWriter" begin
    mktempdir() do dir
        path = joinpath(dir, "thermo.txt")
        sim = obs_soft_sim(5)
        w = ThermoWriter(
            path, PotentialEnergy(), Pressure(), Acceptance(:swap); schedule=Every(5)
        )
        ts = TimeSeries(PotentialEnergy(), Pressure(), Acceptance(:swap); schedule=Every(5))
        @test !isfile(path)
        run!(sim, 12; observers=(w, ts))
        @test isfile(path)
        io = w.io
        @test isopen(io)
        run!(sim, 8; observers=(ts, w))
        @test w.io === io
        close(w)
        @test !isopen(io)
        @test w.io === nothing
        close(w)

        lines = readlines(path)
        header = filter(startswith('#'), lines)
        data = filter(!startswith('#'), lines)
        @test all(startswith('#'), lines[1:length(header)])
        @test any(
            l -> occursin("SwapMonteCarlo.jl v$(pkgversion(SwapMonteCarlo))", l), header
        )
        @test any(l -> occursin("seed = $(sim.seed)", l), header)
        @test any(l -> occursin("ensemble", l), header)
        @test split(header[end])[2:end] ==
            ["sweep", "energy", "pressure", "acceptance_swap"]
        @test length(data) == 4
        rows = [split(l) for l in data]
        @test [parse(Int, r[1]) for r in rows] == [5, 10, 15, 20]
        for k in 1:3
            # Values are written with full precision and parse back exactly.
            @test [parse(Float64, r[k + 1]) for r in rows] == ts.values[k]
        end

        # After closing, the writer appends to the same file without a new header.
        run!(sim, 5; observers=w)
        close(w)
        lines = readlines(path)
        @test count(startswith('#'), lines) == length(header)
        @test parse(Int, split(lines[end])[1]) == 25

        # A new writer overwrites the file.
        w2 = ThermoWriter(path, Density(); schedule=Every(1))
        run!(sim, 2; observers=w2)
        close(w2)
        lines = readlines(path)
        @test length(filter(!startswith('#'), lines)) == 2
        @test_throws ArgumentError ThermoWriter(path)
    end
end

@testset "TrajectoryWriter" begin
    mktempdir() do dir
        path = joinpath(dir, "traj.xyz")
        sim = obs_soft_sim(6)
        w = TrajectoryWriter(path; schedule=Every(10))
        run!(sim, 25; observers=w)
        snapshot = positions(sim.system)
        run!(sim, 5; observers=w)
        close(w)
        @test w.io === nothing
        first_frame = read_extxyz(path; frame=1)
        @test first_frame.sweep == 10
        @test read_extxyz(path; frame=2).sweep == 20
        last_frame = read_extxyz(path, Val(2))
        @test last_frame.sweep == 30
        @test last_frame.positions == positions(sim.system)
        @test last_frame.diameters == diameters(sim.system)
        @test last_frame.box.lengths == sim.system.box.lengths
        @test read_extxyz(path; frame=2).positions != snapshot
    end
end

@testset "ProgressPrinter" begin
    sim = obs_soft_sim(7)
    buf = IOBuffer()
    p = ProgressPrinter(; schedule=Every(4), io=buf)
    run!(sim, 8; observers=p)
    out = String(take!(buf))
    lines = split(chomp(out), '\n')
    @test length(lines) == 2
    @test occursin("sweep", lines[1]) && occursin("4", lines[1])
    @test occursin("displacement", lines[2]) && occursin("swap", lines[2])
    close(p)
end

@testset "generic observers" begin
    o = CountingObserver(Ref(0))
    sim = obs_soft_sim(8)
    run!(sim, 3; observers=(o,))
    @test o.count[] == 3
    @test flush(o) === nothing
    @test close(o) === nothing
end
