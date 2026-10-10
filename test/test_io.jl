# Tests of the extended XYZ writer and reader.

"""
    md_read_file(filepath; dimension=3)

Transcription of the parsing logic of `MolecularDynamics.read_file` (MolecularDynamics.jl
0.8.1), used to check that our files can be read by MolecularDynamics.jl.
"""
function md_read_file(filepath; dimension=3)
    return open(filepath, "r") do io
        n_particles = parse(Int64, readline(io))
        header = readline(io)
        m = match(r"Lattice=\"([^\"]+)\"", header)
        m === nothing && error("Could not parse Lattice property in file header")
        box_entries = parse.(Float64, split(m.captures[1]))
        unitcell = reshape(box_entries, 3, 3)[1:dimension, 1:dimension]
        positions = SVector{dimension,Float64}[]
        radii = Float64[]
        for _ in 1:n_particles
            line = split(readline(io))
            push!(radii, parse(Float64, line[3]))
            push!(
                positions,
                SVector{dimension,Float64}(parse.(Float64, line[4:(3 + dimension)])),
            )
        end
        return unitcell, positions, radii .* 2.0
    end
end

"""
    random_configuration(rng, Val(D), N)

Random orthorhombic box, Cartesian positions and diameters with awkward decimal
expansions, to check exact round trips.
"""
function random_configuration(rng, ::Val{D}, N) where {D}
    L = SVector(ntuple(_ -> 5 + 10 * rand(rng), Val(D)))
    box = OrthorhombicBox(L)
    x = uniform_positions(rng, L, N)
    σ = 0.7 .+ 0.6 .* rand(rng, N)
    # Extreme magnitudes and signed zeros must also survive.
    σ[1] = 1e-300
    x[1] = SVector(ntuple(d -> d == 1 ? -0.0 : 1e-17, Val(D)))
    return box, x, σ
end

@testset "header format" begin
    for D in (2, 3)
        box = CubicBox(Val(D), 3.5)
        x = [SVector(ntuple(d -> 0.25d, Val(D)))]
        io = IOBuffer()
        write_extxyz(io, box, x, [1.5]; sweep=42)
        lines = split(String(take!(io)), '\n')
        @test lines[1] == "1"
        header = lines[2]
        lattice = match(r"Lattice=\"([^\"]+)\"", header).captures[1]
        L = parse.(Float64, split(lattice))
        @test length(L) == 9
        M = reshape(L, 3, 3)
        @test M[1:D, 1:D] == 3.5 * [i == j for i in 1:D, j in 1:D]
        D == 2 && @test M[:, 3] == [0, 0, 1]
        @test occursin("Properties=type:I:1:id:I:1:radius:R:1:pos:R:3", header)
        @test occursin(D == 2 ? "pbc=\"T T F\"" : "pbc=\"T T T\"", header)
        @test occursin(r"(^|\s)Sweep=42($|\s)", header)
        cols = split(lines[3])
        @test length(cols) == 6
        @test cols[1:2] == ["1", "1"]
        @test parse.(Float64, cols[3:6]) == [0.75, 0.25, 0.5, D == 2 ? 0.0 : 0.75]
        @test lines[4] == ""
    end
end

@testset "exact round trip, D = $D" for D in (2, 3)
    rng = test_rng(100 + D)
    (box, x, σ) = random_configuration(rng, Val(D), 50)
    mktempdir() do dir
        path = joinpath(dir, "conf.xyz")
        write_extxyz(path, box, x, σ; sweep=123456789)
        conf = read_extxyz(path)
        @test conf.box isa OrthorhombicBox{D,Float64}
        @test conf.box.lengths == box.lengths
        @test conf.positions isa Vector{SVector{D,Float64}}
        @test conf.positions == x
        @test signbit(conf.positions[1][1])
        @test conf.diameters == σ
        @test conf.sweep === 123456789

        typed = @inferred read_extxyz(path, Val(D))
        @test typed == conf
        @test_throws ArgumentError read_extxyz(path, Val(5 - D))

        # MolecularDynamics.jl reads the same configuration.
        (unitcell, xmd, σmd) = md_read_file(path; dimension=D)
        @test unitcell == [i == j ? box.lengths[i] : 0.0 for i in 1:D, j in 1:D]
        @test xmd == x
        @test σmd == σ
    end
end

@testset "Float32 boxes" begin
    box = OrthorhombicBox(SVector(3.1f0, 2.7f0))
    x = [SVector(0.1f0, 2.3f0), SVector(1.7f0, 0.3f0)]
    σ = [1.1f0, 0.9f0]
    io = IOBuffer()
    write_extxyz(io, box, x, σ)
    mktempdir() do dir
        path = joinpath(dir, "f32.xyz")
        write(path, take!(io))
        conf = read_extxyz(path, Val(2))
        @test Float32.(conf.box.lengths) == box.lengths
        @test map(p -> Float32.(p), conf.positions) == x
        @test Float32.(conf.diameters) == σ
    end
end

@testset "multiple frames" begin
    rng = test_rng(110)
    configurations = [random_configuration(rng, Val(3), n) for n in (5, 1, 8, 3)]
    mktempdir() do dir
        path = joinpath(dir, "traj.xyz")
        for (k, (box, x, σ)) in enumerate(configurations)
            write_extxyz(path, box, x, σ; sweep=10k, mode=k == 1 ? "w" : "a")
        end
        for (k, (box, x, σ)) in enumerate(configurations)
            conf = read_extxyz(path; frame=k)
            @test (conf.box.lengths, conf.positions, conf.diameters, conf.sweep) ==
                (box.lengths, x, σ, 10k)
        end
        @test read_extxyz(path; frame=:first) == read_extxyz(path; frame=1)
        @test read_extxyz(path; frame=:last) == read_extxyz(path; frame=4)
        @test read_extxyz(path) == read_extxyz(path, Val(3); frame=4)
        @test_throws ArgumentError read_extxyz(path; frame=5)
        @test_throws ArgumentError read_extxyz(path; frame=0)
        @test_throws ArgumentError read_extxyz(path; frame=:middle)

        # Writing with mode "w" starts a new file.
        (box, x, σ) = configurations[2]
        write_extxyz(path, box, x, σ; sweep=7)
        @test read_extxyz(path; frame=:first).sweep == 7
        @test_throws ArgumentError read_extxyz(path; frame=2)

        # An incomplete last frame (an interrupted write) is ignored by `:last`.
        (box, x, σ) = configurations[3]
        io = IOBuffer()
        write_extxyz(io, box, x, σ; sweep=8)
        text = String(take!(io))
        open(io -> write(io, text[1:(end - 30)]), path, "a")
        @test read_extxyz(path).sweep == 7
        @test_throws ArgumentError read_extxyz(path; frame=2)
        @test_throws ArgumentError write_extxyz(path, box, x, σ; mode="r")
    end
end

@testset "frames of a long trajectory" begin
    rng = test_rng(120)
    box = CubicBox(Val(2), 4.0)
    mktempdir() do dir
        path = joinpath(dir, "long.xyz")
        open(path, "w") do io
            for k in 1:2000
                n = 1 + k % 7
                write_extxyz(
                    io, box, uniform_positions(rng, box.lengths, n), ones(n); sweep=k
                )
            end
        end
        @test read_extxyz(path).sweep == 2000
        @test length(read_extxyz(path).positions) == 1 + 2000 % 7
        @test read_extxyz(path; frame=1234).sweep == 1234
    end
end

@testset "files written by MolecularDynamics.jl" begin
    # Output of `MolecularDynamics.write_to_file` (2D, padded 3×3 lattice, 6 decimals).
    md2d = """
    2
    Lattice="10.0 0.0 0.0 0.0 8.0 0.0 0.0 0.0 1.0" Properties=type:I:1:id:I:1:radius:R:1:pos:R:3 pbc="T T F" Time=500
    1 1 0.500000 1.250000 3.500000 0.000000
    1 2 0.700000 4.000000 0.125000 0.000000
    """
    # Files written before MolecularDynamics.jl 0.8.1 used a 2×2 lattice in 2D.
    old2d = """
    1
    Lattice="3.0 0.0 0.0 2.0" Properties=type:I:1:id:I:1:radius:R:1:pos:R:2 Time=0.5
    1 1 0.5 1.0 1.5
    """
    md3d = """
    1
    Lattice="2.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 4.0" Properties=type:I:1:id:I:1:radius:R:1:pos:R:3 pbc="T T T" Time=1.5e+06
    1 1 0.600000 0.100000 0.200000 0.300000
    """
    mktempdir() do dir
        path = joinpath(dir, "md.xyz")
        write(path, md2d)
        conf = read_extxyz(path)
        @test conf.box.lengths == SVector(10.0, 8.0)
        @test conf.positions == [SVector(1.25, 3.5), SVector(4.0, 0.125)]
        @test conf.diameters == [1.0, 1.4]
        @test conf.sweep == 500

        write(path, old2d)
        conf = read_extxyz(path)
        @test conf.box.lengths == SVector(3.0, 2.0)
        @test conf.positions == [SVector(1.0, 1.5)]
        @test conf.diameters == [1.0]
        @test conf.sweep == 0

        write(path, md3d)
        conf = read_extxyz(path, Val(3))
        @test conf.box.lengths == SVector(2.0, 3.0, 4.0)
        @test conf.positions == [SVector(0.1, 0.2, 0.3)]
        @test conf.diameters == [1.2]
        @test conf.sweep == 1_500_000

        # Only orthorhombic boxes are supported.
        write(path, replace(md3d, "2.0 0.0 0.0 0.0 3.0" => "2.0 0.0 0.0 0.5 3.0"))
        @test_throws ArgumentError read_extxyz(path)
    end
end

@testset "invalid arguments" begin
    box = CubicBox(Val(2), 2.0)
    @test_throws DimensionMismatch write_extxyz(
        IOBuffer(), box, [SVector(0.0, 0.0)], Float64[]
    )
    @test_throws MethodError write_extxyz(IOBuffer(), box, [SVector(0.0, 0.0, 0.0)], [1.0])
    mktempdir() do dir
        path = joinpath(dir, "empty.xyz")
        touch(path)
        @test_throws ArgumentError read_extxyz(path)
    end
end

@testset "write a System" begin
    box = CubicBox(Val(2), 5.0)
    x = [SVector(0.5, 1.0), SVector(4.25, 2.5), SVector(2.0, 4.75)]
    σ = [1.0, 1.2, 0.8]
    if applicable(System, box, x, σ, Ideal())
        sys = System(box, x, σ, Ideal())
        mktempdir() do dir
            path = joinpath(dir, "sys.xyz")
            write_extxyz(path, sys; sweep=3)
            conf = read_extxyz(path, Val(2))
            @test conf.box.lengths == box.lengths
            @test conf.positions ≈ x
            @test conf.positions == [SMC.cartesian(sys.box, s) for s in sys.frac]
            @test conf.diameters == σ
            @test conf.sweep == 3
        end
    else
        # The `System` constructor is not available yet.
        @test_skip false
    end
end

@testset "float formatting" begin
    # The formatting round-trips and gives the same text as `print`.
    rng = test_rng(130)
    values = [0.0, -0.0, 1.0, 0.1, 1e-5, 1e20, -2.2250738585072014e-308, Inf, NaN]
    append!(values, randn(rng, 100) .* 10.0 .^ rand(rng, -30:30, 100))
    for x in values
        io = IOBuffer()
        SMC.print_float(io, x)
        @test String(take!(io)) == string(x)
    end
end
