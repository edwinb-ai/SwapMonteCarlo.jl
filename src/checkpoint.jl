"""
Version of the checkpoint file layout, increased when it changes.
"""
const CHECKPOINT_FORMAT = 1

"""
    save_checkpoint(path, sim)

Save the complete state of the simulation `sim` to `path`: the system (positions, box,
running energy, cell list), the ensemble, the moves with their acceptance counters and
step sizes, the state of the random number generator, the seed and the sweep counter.
[`load_checkpoint`](@ref) restores it, and continuing the restored simulation gives
bitwise the same results as continuing the original one. Observers are not saved.

The file is written with the `Serialization` standard library, so it can only be read
reliably by the same Julia version and the same version of SwapMonteCarlo.jl (and of the
packages that define custom potentials or moves). It is a restart file, not an archive
format; use [`write_extxyz`](@ref) for configurations meant to be kept. The file is
written to a temporary name first and then renamed, so an interrupted save does not
destroy an existing checkpoint.
"""
function save_checkpoint(path::AbstractString, sim::Simulation)
    data = (;
        format=CHECKPOINT_FORMAT,
        julia_version=VERSION,
        package_version=pkgversion(@__MODULE__),
        simulation=sim,
    )
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        return serialize(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

"""
    load_checkpoint(path) -> Simulation

Restore a simulation saved with [`save_checkpoint`](@ref). Warns if the checkpoint was
written by a different Julia or package version, in which case it may fail to load or be
restored incorrectly.
"""
function load_checkpoint(path::AbstractString)
    data = open(deserialize, path)
    (data isa NamedTuple && haskey(data, :format) && haskey(data, :simulation)) ||
        throw(ArgumentError("$(path) is not a SwapMonteCarlo.jl checkpoint"))
    data.format == CHECKPOINT_FORMAT ||
        throw(ArgumentError("unsupported checkpoint format $(data.format) in $(path)"))
    package_version = pkgversion(@__MODULE__)
    if data.julia_version != VERSION || data.package_version != package_version
        @warn(
            "checkpoint written by a different version; it may not be restored correctly",
            path,
            saved_julia = data.julia_version,
            saved_package = data.package_version,
            julia = VERSION,
            package = package_version,
        )
    end
    sim = data.simulation
    sim isa Simulation || throw(ArgumentError("$(path) does not contain a Simulation"))
    return sim
end
