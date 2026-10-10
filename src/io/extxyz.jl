# Extended XYZ trajectories, compatible with MolecularDynamics.jl, OVITO and ASE.
#
# A frame looks like (2D example, one line per particle after the two header lines):
#
#     2
#     Lattice="10.0 0.0 0.0 0.0 8.0 0.0 0.0 0.0 1.0" Properties=type:I:1:id:I:1:radius:R:1:pos:R:3 pbc="T T F" Sweep=100
#     1 1 0.5 1.25 3.5 0.0
#     1 2 0.7 4.0 0.125 0.0
#
# `Lattice` lists the three box vectors one after another, `radius` is half the diameter
# and positions are Cartesian with three components (z = 0 in 2D, where the box is padded
# with a unit z vector and `pbc="T T F"`).

"""
    EXTXYZ_PROPERTIES

Per particle columns written by [`write_extxyz`](@ref), the same as MolecularDynamics.jl.
"""
const EXTXYZ_PROPERTIES = "type:I:1:id:I:1:radius:R:1:pos:R:3"

"""
    print_float(io, x)

Print `x` as a `Float64` with the shortest representation that parses back to the same
number, so that files round-trip exactly (also for `Float32`, whose conversion to
`Float64` is exact).
"""
print_float(io::IO, x::Real) = print(io, Float64(x))

"""
    pbc_string(D)

Periodic boundary flags of a `D`-dimensional box embedded in 3D, e.g. `"T T F"` in 2D.
"""
pbc_string(D::Int) = join((d <= D ? "T" : "F" for d in 1:3), " ")

"""
    write_frame(io, box, positions, diameters, sweep)

Write one extended XYZ frame. `positions` is any iterable of Cartesian `SVector{D}` with
the same length as `diameters`.

The frame is formatted in memory and written to `io` at once, which is much faster than
many small writes to a file (each one takes a lock).
"""
function write_frame(io::IO, box::AbstractBox{D}, positions, diameters, sweep) where {D}
    D <= 3 || throw(ArgumentError("extended XYZ supports at most 3 dimensions, got $(D)"))
    out = IOBuffer(; sizehint=96 * (length(diameters) + 2))
    format_frame(out, box, positions, diameters, sweep)
    write(io, take!(out))
    return nothing
end

"""
    format_frame(io, box, positions, diameters, sweep)

Format one extended XYZ frame into `io`, see [`write_frame`](@ref).
"""
function format_frame(
    io::IO, box::AbstractBox{D,T}, positions, diameters, sweep
) where {D,T}
    println(io, length(diameters))

    # The box vectors are the images of the fractional unit vectors, padded to 3D with
    # unit vectors along the missing axes.
    print(io, "Lattice=\"")
    for k in 1:3
        for d in 1:3
            k + d > 2 && print(io, ' ')
            if k <= D
                ek = SVector(ntuple(i -> i == k ? one(T) : zero(T), Val(D)))
                print_float(io, d <= D ? cartesian(box, ek)[d] : zero(T))
            else
                print_float(io, d == k ? one(T) : zero(T))
            end
        end
    end
    print(io, "\" Properties=", EXTXYZ_PROPERTIES)
    print(io, " pbc=\"", pbc_string(D), "\" Sweep=", sweep, '\n')

    # One line per particle: type, id, radius and three Cartesian coordinates.
    for (i, (x, σ)) in enumerate(zip(positions, diameters))
        print(io, "1 ", i, ' ')
        print_float(io, σ / 2)
        for d in 1:3
            print(io, ' ')
            print_float(io, d <= D ? x[d] : zero(T))
        end
        print(io, '\n')
    end
    return nothing
end

"""
    write_extxyz(io::IO, box, positions, diameters; sweep=0)
    write_extxyz(io::IO, sys::System; sweep=0)
    write_extxyz(path::AbstractString, args...; mode="w", sweep=0)

Write a configuration as one extended XYZ frame, readable by [`read_extxyz`](@ref),
MolecularDynamics.jl (`read_file`), OVITO and ASE.

`positions` are Cartesian `SVector{D}` and `diameters` the particle diameters; the file
stores radii (`diameters / 2`) as MolecularDynamics.jl does. The header holds a
9-component `Lattice` (2D boxes are padded with a unit z vector and `pbc="T T F"`), the
`Properties` of the columns, `pbc` and `Sweep=sweep`. Floats are written with full
precision, so reading the file back gives bitwise identical `Float64` values.

With a `path`, the file is overwritten (`mode="w"`) or a frame is appended (`mode="a"`).
"""
function write_extxyz(
    io::IO,
    box::AbstractBox{D},
    positions::AbstractVector{<:SVector{D}},
    diameters::AbstractVector{<:Real};
    sweep::Integer=0,
) where {D}
    length(positions) == length(diameters) || throw(
        DimensionMismatch(
            "got $(length(positions)) positions and $(length(diameters)) diameters"
        ),
    )
    write_frame(io, box, positions, diameters, sweep)
    return nothing
end

function write_extxyz(io::IO, sys::System; sweep::Integer=0)
    box = sys.box
    positions = Iterators.map(s -> cartesian(box, s), sys.frac)
    write_frame(io, box, positions, sys.diameters, sweep)
    return nothing
end

function write_extxyz(path::AbstractString, args...; mode::AbstractString="w", kwargs...)
    mode in ("w", "a") || throw(ArgumentError("mode must be \"w\" or \"a\", got $(mode)"))
    open(path, mode) do io
        return write_extxyz(io, args...; kwargs...)
    end
    return nothing
end

# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------

"""
    LineSkipper(io; bufsize=2^16)

Buffered forward reader that skips lines with `memchr`, much faster than `readline` for
long trajectories. It tracks the file offset of the next unread byte.
"""
mutable struct LineSkipper{I<:IO}
    const io::I
    const buf::Vector{UInt8}
    i::Int
    n::Int
    offset::Int
end

function LineSkipper(io::IO; bufsize::Int=2^16)
    return LineSkipper(io, Vector{UInt8}(undef, bufsize), 1, 0, position(io))
end

"""
    fill_buffer!(sk::LineSkipper) -> Bool

Read the next chunk of the file into the buffer. Returns `false` at the end of the file.
"""
function fill_buffer!(sk::LineSkipper)
    sk.offset += sk.n
    sk.n = readbytes!(sk.io, sk.buf)
    sk.i = 1
    return sk.n > 0
end

"""
    position(sk::LineSkipper)

File offset of the next unread byte.
"""
Base.position(sk::LineSkipper) = sk.offset + sk.i - 1

"""
    find_newline(sk::LineSkipper) -> Union{Int,Nothing}

Index of the next newline in the valid part of the buffer, or `nothing`.
"""
@inline function find_newline(sk::LineSkipper)
    # Searching the whole `Vector` (not a view) uses `memchr`; matches past the valid
    # bytes are stale data from a previous chunk.
    j = findnext(==(UInt8('\n')), sk.buf, sk.i)
    return (j === nothing || j > sk.n) ? nothing : j
end

"""
    skip_lines!(sk::LineSkipper, k) -> Bool

Skip `k` lines. Returns `false` if the file ends before `k` complete lines. A line is
complete only if it ends with a newline, so that a frame cut by an interrupted write is
detected.
"""
function skip_lines!(sk::LineSkipper, k::Int)
    while k > 0
        if sk.i > sk.n
            fill_buffer!(sk) || return false
        end
        j = find_newline(sk)
        if j === nothing
            sk.i = sk.n + 1
        else
            sk.i = j + 1
            k -= 1
        end
    end
    return true
end

"""
    read_line!(sk::LineSkipper) -> Union{String,Nothing}

Read the next line (without the newline), or `nothing` at the end of the file.
"""
function read_line!(sk::LineSkipper)
    out = UInt8[]
    while true
        if sk.i > sk.n
            fill_buffer!(sk) || return isempty(out) ? nothing : String(out)
        end
        j = find_newline(sk)
        if j === nothing
            append!(out, view(sk.buf, (sk.i):(sk.n)))
            sk.i = sk.n + 1
        else
            append!(out, view(sk.buf, (sk.i):(j - 1)))
            sk.i = j + 1
            return String(out)
        end
    end
end

"""
    parse_count(line) -> Union{Int,Nothing}

Number of particles from the first line of a frame, or `nothing` if the line is blank.
"""
function parse_count(line::AbstractString)
    s = strip(line)
    isempty(s) && return nothing
    n = tryparse(Int, s)
    (n === nothing || n < 0) &&
        throw(ArgumentError("invalid particle count line in extended XYZ file: $(line)"))
    return n
end

"""
    frame_offset(io, frame) -> Int

File offset of the start of `frame` (`:first`, `:last` or a 1-based `Int`). Incomplete
frames at the end of the file (e.g. from an interrupted run) are ignored.
"""
function frame_offset(io::IO, frame)
    target = if frame isa Integer
        frame >= 1 || throw(ArgumentError("frame must be ≥ 1, got $(frame)"))
        Int(frame)
    elseif frame === :first
        1
    elseif frame === :last
        typemax(Int)
    else
        throw(ArgumentError("frame must be :first, :last or an integer, got $(frame)"))
    end
    sk = LineSkipper(io)
    count = 0
    last_offset = -1
    while count < target
        start = position(sk)
        line = read_line!(sk)
        line === nothing && break
        n = parse_count(line)
        n === nothing && break
        # The comment line and the particle lines.
        skip_lines!(sk, n + 1) || break
        count += 1
        last_offset = start
    end
    count == 0 && throw(ArgumentError("no complete frame in extended XYZ file"))
    if frame !== :last && count < target
        throw(ArgumentError("frame $(frame) requested but the file has $(count) frames"))
    end
    return last_offset
end

"""
    header_value(header, key) -> Union{String,Nothing}

Value of `key=value` or `key="value"` in an extended XYZ comment line, or `nothing`.
Keys are matched case-insensitively, as in the extended XYZ specification.
"""
function header_value(header::AbstractString, key::AbstractString)
    m = match(Regex("(?:^|\\s)\\Q$(key)\\E=(?:\"([^\"]*)\"|(\\S*))", "i"), header)
    m === nothing && return nothing
    return String(something(m.captures[1], m.captures[2]))
end

"""
    header_dimension(header) -> Int

Dimension of a frame: the number of periodic directions in `pbc` if present, otherwise 2
for a 4-component `Lattice` (old MolecularDynamics.jl 2D files) and 3 for 9 components.
"""
function header_dimension(header::AbstractString)
    pbc = header_value(header, "pbc")
    if pbc !== nothing
        flags = map(f -> uppercase(f) in ("T", "TRUE", "1"), split(pbc))
        D = count(flags)
        # Periodic directions must come first, as in a box padded to 3D.
        (length(flags) in (2, 3) && all(d -> flags[d] == (d <= D), eachindex(flags))) ||
            throw(ArgumentError("unsupported pbc in extended XYZ: $(pbc)"))
        D >= 1 || throw(ArgumentError("no periodic direction in extended XYZ: $(pbc)"))
        return D
    end
    lattice = header_value(header, "Lattice")
    lattice === nothing && throw(ArgumentError("no Lattice in extended XYZ header"))
    nl = length(split(lattice))
    nl == 4 && return 2
    nl == 9 && return 3
    return throw(ArgumentError("Lattice has $(nl) entries, expected 9"))
end

"""
    header_box(header, Val(D)) -> OrthorhombicBox{D,Float64}

Box of a frame from its `Lattice`. Only orthorhombic boxes (diagonal lattices) are
supported.
"""
function header_box(header::AbstractString, ::Val{D}) where {D}
    lattice = header_value(header, "Lattice")
    lattice === nothing && throw(ArgumentError("no Lattice in extended XYZ header"))
    entries = parse.(Float64, split(lattice))
    # Files written by MolecularDynamics.jl before 0.8.1 store a D×D lattice in 2D.
    m = if length(entries) == 9
        3
    elseif length(entries) == D^2
        D
    else
        0
    end
    m == 0 && throw(ArgumentError("Lattice has $(length(entries)) entries, expected 9"))
    M = reshape(entries, m, m)
    for k in 1:D, d in 1:D
        d == k ||
            iszero(M[d, k]) ||
            throw(ArgumentError("only orthorhombic boxes are supported, got $(lattice)"))
    end
    return OrthorhombicBox(SVector(ntuple(d -> M[d, d], Val(D))))
end

"""
    header_columns(header, D) -> (radius_column, position_column)

Columns of the radius and of the first coordinate, from the `Properties` key (defaults to
the MolecularDynamics.jl layout `type id radius x y z` when it is missing).
"""
function header_columns(header::AbstractString, D::Int)
    props = header_value(header, "Properties")
    props === nothing && return (3, 4)
    fields = split(props, ':')
    length(fields) % 3 == 0 || throw(ArgumentError("invalid Properties: $(props)"))
    (rcol, pcol, col) = (0, 0, 1)
    for k in 1:3:length(fields)
        (name, n) = (lowercase(fields[k]), parse(Int, fields[k + 2]))
        if name == "radius"
            rcol = col
        elseif name == "pos"
            n >= D || throw(ArgumentError("pos has $(n) components, expected ≥ $(D)"))
            pcol = col
        end
        col += n
    end
    (rcol == 0 || pcol == 0) &&
        throw(ArgumentError("Properties must contain radius and pos, got $(props)"))
    return (rcol, pcol)
end

"""
    header_sweep(header) -> Int

Sweep of a frame: the `Sweep` key, else an integer valued `Time` key (as written by
MolecularDynamics.jl), else 0.
"""
function header_sweep(header::AbstractString)
    s = header_value(header, "Sweep")
    s !== nothing && return parse(Int, s)
    t = header_value(header, "Time")
    t === nothing && return 0
    tf = tryparse(Float64, t)
    return (tf !== nothing && isinteger(tf)) ? Int(tf) : 0
end

"""
    read_frame(io, Val(D)) -> NamedTuple

Read the frame that starts at the current position of `io`.
"""
function read_frame(io::IO, ::Val{D}) where {D}
    n = parse_count(readline(io))
    n === nothing && throw(ArgumentError("missing particle count in extended XYZ frame"))
    header = readline(io)
    Dfile = header_dimension(header)
    Dfile == D || throw(ArgumentError("the file is $(Dfile)-dimensional, not $(D)"))
    box = header_box(header, Val(D))
    (rcol, pcol) = header_columns(header, D)
    sweep = header_sweep(header)

    positions = Vector{SVector{D,Float64}}(undef, n)
    diameters = Vector{Float64}(undef, n)
    xs = zeros(Float64, D)
    for i in 1:n
        line = readline(io)
        radius = NaN
        nfound = 0
        for (col, word) in enumerate(eachsplit(line))
            if col == rcol
                radius = parse(Float64, word)
                nfound += 1
            elseif pcol <= col < pcol + D
                xs[col - pcol + 1] = parse(Float64, word)
                nfound += 1
            end
        end
        nfound == D + 1 ||
            throw(ArgumentError("invalid particle line $(i) in extended XYZ: $(line)"))
        positions[i] = SVector{D,Float64}(xs)
        diameters[i] = 2 * radius
    end
    return (; box, positions, diameters, sweep)
end

"""
    read_extxyz(path; frame=:last) -> (; box, positions, diameters, sweep)
    read_extxyz(path, Val(D); frame=:last)

Read one frame of an extended XYZ file written by [`write_extxyz`](@ref) or by
MolecularDynamics.jl.

`frame` is `:first`, `:last` (the last complete frame) or a 1-based frame number.
Returns the `OrthorhombicBox{D,Float64}`, the Cartesian positions as
`Vector{SVector{D,Float64}}`, the diameters (twice the `radius` column) and the sweep
(the `Sweep` key; an integer `Time` key or 0 if absent).

The dimension is inferred from `pbc` (or the size of `Lattice`), so the first form is
type unstable; pass `Val(D)` for a type stable result (an error is thrown if the file has
a different dimension).
"""
function read_extxyz(path::AbstractString; frame=:last)
    return open(path, "r") do io
        seek(io, frame_offset(io, frame))
        start = position(io)
        readline(io)
        D = header_dimension(readline(io))
        seek(io, start)
        # Explicit branches keep the result a small union of concrete types.
        if D == 3
            return read_frame(io, Val(3))
        elseif D == 2
            return read_frame(io, Val(2))
        else
            return read_frame(io, Val(1))
        end
    end
end

function read_extxyz(path::AbstractString, ::Val{D}; frame=:last) where {D}
    return open(path, "r") do io
        seek(io, frame_offset(io, frame))
        return read_frame(io, Val(D))
    end
end
