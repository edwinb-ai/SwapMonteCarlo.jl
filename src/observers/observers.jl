"""
    Observer

Supertype of the objects that record data during [`run!`](@ref).

# Interface

  - `observe!(obs, sim)`: called after every sweep of the [`Simulation`](@ref) `sim`
    (after `sim.sweep` was incremented); the observer checks its own schedule, see
    [`Schedule`](@ref).
  - `flush(obs)`: called at the end of every `run!`. Defaults to nothing.
  - `close(obs)`: release resources such as open files. Defaults to nothing.

Observers are not part of the simulation (nor of its checkpoints) and can be passed to
several consecutive calls of `run!`; files stay open between them.
"""
abstract type Observer end

"""
    observe!(obs::Observer, sim)

Let the observer `obs` record the current state of the simulation `sim`, if its schedule
is due at `sim.sweep`.
"""
function observe! end

Base.flush(::Observer) = nothing
Base.close(::Observer) = nothing

"""
    format_value(io, x)

Write the number `x` with the shortest representation that parses back to the same
`Float64` (integers are written as integers).
"""
format_value(io::IO, x::Integer) = print(io, x)
format_value(io::IO, x::Real) = print(io, Float64(x))

"""
    check_quantities(quantities)

Throw an `ArgumentError` unless all `quantities` are [`Quantity`](@ref) objects.
"""
function check_quantities(quantities::Tuple)
    for q in quantities
        q isa Quantity || throw(
            ArgumentError("expected Quantity objects (e.g. PotentialEnergy()), got $(q)"),
        )
    end
    return nothing
end

"""
    ThermoWriter(path, quantities...; schedule=Every(100))

Write the `quantities` (see [`Quantity`](@ref)) as columns of a text file, one line per
due sweep, preceded by the sweep number.

The file is created (overwriting an existing one) at the first due sweep and kept open
across calls of [`run!`](@ref) until [`close`](@ref) is called (it is reopened in
append mode if the observer is used after closing it). Header lines start with `#` and
give the package version, the seed of the simulation, the ensemble, the system and
finally the column labels. Numbers are written with full precision, so the file can be
read with e.g. `readdlm(path; comments=true)`.
"""
mutable struct ThermoWriter{Q<:Tuple,S<:Schedule} <: Observer
    const path::String
    const quantities::Q
    const schedule::S
    io::Union{IOStream,Nothing}
    started::Bool
end

function ThermoWriter(path::AbstractString, quantities...; schedule::Schedule=Every(100))
    isempty(quantities) && throw(ArgumentError("ThermoWriter needs at least one quantity"))
    check_quantities(quantities)
    return ThermoWriter(String(path), quantities, schedule, nothing, false)
end

"""
    write_thermo_header(io, w::ThermoWriter, sim)

Write the comment lines that start a thermo file.
"""
function write_thermo_header(io::IO, w::ThermoWriter, sim)
    println(io, "# SwapMonteCarlo.jl v", pkgversion(@__MODULE__))
    println(io, "# seed = ", sim.seed)
    println(io, "# ensemble = ", sim.ensemble)
    println(io, "# system = ", sim.system)
    println(io, "# moves = ", join(map(name, sim.moves), " "))
    println(io, "# ", join(("sweep", map(label, w.quantities)...), " "))
    return nothing
end

"""
    output_stream!(w) -> (io::IOStream, created::Bool)

The open file of the writer `w` (a [`ThermoWriter`](@ref) or [`TrajectoryWriter`](@ref)),
which is created on first use and reopened in append mode after a `close`. Also returns
whether the file was just created.
"""
function output_stream!(w)
    created = false
    if w.io === nothing
        w.io = open(w.path, w.started ? "a" : "w")
        created = !w.started
        w.started = true
    end
    return (w.io::IOStream, created)
end

function observe!(w::ThermoWriter, sim)
    isdue(w.schedule, sim.sweep) || return nothing
    (io, created) = output_stream!(w)
    created && write_thermo_header(io, w, sim)
    # Format the line in memory and write it at once.
    line = IOBuffer()
    print(line, sim.sweep)
    for q in w.quantities
        print(line, ' ')
        format_value(line, measure(q, sim))
    end
    print(line, '\n')
    write(io, take!(line))
    return nothing
end

function Base.flush(w::ThermoWriter)
    w.io === nothing || flush(w.io)
    return nothing
end

function Base.close(w::ThermoWriter)
    if w.io !== nothing
        close(w.io)
        w.io = nothing
    end
    return nothing
end

"""
    TrajectoryWriter(path; schedule=Every(1000))

Append the configuration as an extended XYZ frame (see [`write_extxyz`](@ref)) to the
file `path` at every due sweep, with `Sweep` set to the sweep number. Frames can be read
back with [`read_extxyz`](@ref)`(path; frame=k)`.

The file is created (overwriting an existing one) at the first due sweep and kept open
across calls of [`run!`](@ref) until [`close`](@ref) is called.
"""
mutable struct TrajectoryWriter{S<:Schedule} <: Observer
    const path::String
    const schedule::S
    io::Union{IOStream,Nothing}
    started::Bool
end

function TrajectoryWriter(path::AbstractString; schedule::Schedule=Every(1000))
    return TrajectoryWriter(String(path), schedule, nothing, false)
end

function observe!(w::TrajectoryWriter, sim)
    isdue(w.schedule, sim.sweep) || return nothing
    (io, _) = output_stream!(w)
    write_extxyz(io, sim.system; sweep=sim.sweep)
    return nothing
end

function Base.flush(w::TrajectoryWriter)
    w.io === nothing || flush(w.io)
    return nothing
end

function Base.close(w::TrajectoryWriter)
    if w.io !== nothing
        close(w.io)
        w.io = nothing
    end
    return nothing
end

"""
    TimeSeries(quantities...; schedule=Every(1))

Record the `quantities` (see [`Quantity`](@ref)) in memory at every due sweep.

# Fields

  - `sweeps::Vector{Int}`: the sweeps at which data was recorded.
  - `values::Vector{Vector{Float64}}`: one time series per quantity, in the order given.
  - `labels::Vector{String}`: the labels of the quantities.
"""
struct TimeSeries{Q<:Tuple,S<:Schedule} <: Observer
    quantities::Q
    schedule::S
    labels::Vector{String}
    sweeps::Vector{Int}
    values::Vector{Vector{Float64}}
end

function TimeSeries(quantities...; schedule::Schedule=Every(1))
    isempty(quantities) && throw(ArgumentError("TimeSeries needs at least one quantity"))
    check_quantities(quantities)
    labels = [label(q) for q in quantities]
    values = [Float64[] for _ in quantities]
    return TimeSeries(quantities, schedule, labels, Int[], values)
end

function observe!(ts::TimeSeries, sim)
    isdue(ts.schedule, sim.sweep) || return nothing
    push!(ts.sweeps, sim.sweep)
    for (k, q) in enumerate(ts.quantities)
        push!(ts.values[k], measure(q, sim))
    end
    return nothing
end

"""
    getindex(ts::TimeSeries, label::AbstractString) -> Vector{Float64}

Time series of the quantity with the given label, e.g. `ts["energy"]`.
"""
function Base.getindex(ts::TimeSeries, l::AbstractString)
    k = findfirst(==(l), ts.labels)
    k === nothing && throw(KeyError(l))
    return ts.values[k]
end

"""
    Callback(f; schedule=Every(1))

Call `f(sim)` at every due sweep, e.g. to compute custom observables or to stop early by
throwing an exception.
"""
struct Callback{F,S<:Schedule} <: Observer
    f::F
    schedule::S
end

Callback(f; schedule::Schedule=Every(1)) = Callback(f, schedule)

function observe!(c::Callback, sim)
    isdue(c.schedule, sim.sweep) && c.f(sim)
    return nothing
end

"""
    ProgressPrinter(; schedule=Every(1000), io=stdout)

Print the sweep, the energy per particle, the density and the acceptance rates of the
moves (since the last [`reset_stats!`](@ref)) at every due sweep.
"""
struct ProgressPrinter{S<:Schedule,I<:IO} <: Observer
    schedule::S
    io::I
end

function ProgressPrinter(; schedule::Schedule=Every(1000), io::IO=stdout)
    return ProgressPrinter(schedule, io)
end

function observe!(p::ProgressPrinter, sim)
    isdue(p.schedule, sim.sweep) || return nothing
    sys = sim.system
    io = p.io
    @printf(
        io,
        "sweep %10d  U/N = %12.6f  ρ = %.6f",
        sim.sweep,
        measure(PotentialEnergy(), sim),
        density(sys)
    )
    for move in sim.moves
        @printf(io, "  %s = %.3f", name(move), acceptance_rate(move))
    end
    println(io)
    return nothing
end

function Base.flush(p::ProgressPrinter)
    flush(p.io)
    return nothing
end
