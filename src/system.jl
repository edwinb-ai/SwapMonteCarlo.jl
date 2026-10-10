"""
    System(box, positions, diameters, potential)

Physical state of a periodic particle system: positions, diameters, simulation box and
pair potential, plus a cell list and the running total energy.

The dimension `D` and floating point type `T` are taken from `box`. Positions are given in
Cartesian coordinates and stored internally as fractional coordinates in `[0, 1)^D`.

# Fields

  - `frac::Vector{SVector{D,T}}`: fractional coordinates.
  - `diameters::Vector{T}`: particle diameters.
  - `box::B`: simulation box (replaced by volume moves).
  - `potential::P`: pair potential.
  - `cells::C`: cell list used for neighbor searches.
  - `rc::T`: largest pair cutoff, see [`max_cutoff`](@ref).
  - `energy::T`: running total energy, including long range corrections.
"""
mutable struct System{D,T<:AbstractFloat,B<:AbstractBox{D,T},P<:Potential,C}
    const frac::Vector{SVector{D,T}}
    const diameters::Vector{T}
    box::B
    const potential::P
    const cells::C
    const rc::T
    energy::T
end

"""
    System(box::AbstractBox{D,T}, positions, diameters, potential)

Build a system from Cartesian `positions` (wrapped into the box) and `diameters`,
converted to the box's float type `T`. Throws an `ArgumentError` if the lengths do not
match, the largest cutoff does not fit in the box (minimum image convention), the
potential is not symmetric in the diameters, or the initial energy is not finite.
"""
function System(
    box::AbstractBox{D,T},
    positions::AbstractVector{<:AbstractVector},
    diameters::AbstractVector{<:Real},
    potential::Potential,
) where {D,T}
    n = length(positions)
    n >= 1 || throw(ArgumentError("a system needs at least one particle"))
    length(diameters) == n || throw(
        ArgumentError(
            "got $(n) positions but $(length(diameters)) diameters; they must match"
        ),
    )
    for x in positions
        length(x) == D || throw(
            ArgumentError("positions must have $(D) components (as the box), got $(x)")
        )
    end
    σ = T[d for d in diameters]
    all(d -> isfinite(d) && d > 0, σ) ||
        throw(ArgumentError("diameters must be positive and finite"))
    frac = [wrap_fractional(fractional(box, SVector{D,T}(x))) for x in positions]
    rc = T(max_cutoff(potential, σ))
    (isfinite(rc) && rc >= 0) ||
        throw(ArgumentError("the largest pair cutoff must be finite, got $(rc)"))
    fits_cutoff(box, rc) || throw(
        ArgumentError(
            "the largest pair cutoff rc = $(rc) exceeds half the smallest box width " *
            "$(minimum(perpendicular_widths(box)) / 2); use a larger box (more particles)",
        ),
    )
    check_symmetry(potential, σ, Xoshiro(0x5eed))
    cells = CellList(frac, box, rc)
    sys = System(frac, σ, box, potential, cells, rc, zero(T))
    energy = total_energy(sys)
    isfinite(energy) || throw(
        ArgumentError(
            "the initial energy is $(energy): overlapping hard cores; use " *
            "insert_hard_spheres or start at a lower density",
        ),
    )
    sys.energy = energy
    return sys
end

"""
    length(sys::System)

Number of particles.
"""
Base.length(sys::System) = length(sys.frac)

"""
    dimension(sys::System)

Spatial dimension `D` of the system.
"""
dimension(::System{D}) where {D} = D

"""
    positions(sys) -> Vector{SVector{D,T}}

Cartesian positions of the particles (a copy), inside the box.
"""
positions(sys::System) = [cartesian(sys.box, s) for s in sys.frac]

"""
    diameters(sys) -> Vector{T}

Particle diameters (a copy).
"""
diameters(sys::System) = copy(sys.diameters)

"""
    volume(sys)

Volume (area in 2D) of the simulation box.
"""
volume(sys::System) = volume(sys.box)

"""
    density(sys)

Number density `N / V`.
"""
density(sys::System) = length(sys) / volume(sys)

"""
    packing_fraction(sys)

Fraction of the volume covered by the particles, `π/4 Σσ² / V` in 2D and `π/6 Σσ³ / V`
in 3D (overlaps are not subtracted).
"""
function packing_fraction(sys::System{D,T}) where {D,T}
    return particle_volume(Val(D), sys.diameters) / volume(sys)
end

"""
    particle_volume(Val(D), diameters)

Total volume of `D`-dimensional spheres with the given diameters (`D` = 2 or 3).
"""
function particle_volume(::Val{2}, diameters::AbstractVector{T}) where {T}
    return T(π) / 4 * sum(abs2, diameters)
end
function particle_volume(::Val{3}, diameters::AbstractVector{T}) where {T}
    return T(π) / 6 * sum(d -> d^3, diameters)
end
function particle_volume(::Val{D}, diameters::AbstractVector) where {D}
    return throw(
        ArgumentError("packing fractions are only defined in 2 and 3 dimensions, got $(D)")
    )
end

function Base.show(io::IO, sys::System{D,T}) where {D,T}
    print(
        io,
        "System{$(D),$(T)}: N=$(length(sys)), V=$(round(volume(sys); sigdigits=6)), ",
        "ρ=$(round(density(sys); sigdigits=6)), $(nameof(typeof(sys.potential)))",
    )
    return nothing
end
