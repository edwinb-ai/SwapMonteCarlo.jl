"""
    Quantity

Supertype of the scalar observables recorded by [`ThermoWriter`](@ref) and
[`TimeSeries`](@ref).

# Interface

  - `measure(q, sim) -> Float64`: value of the quantity for the current state of the
    [`Simulation`](@ref) `sim`.
  - `label(q) -> String`: column label, without spaces.
"""
abstract type Quantity end

"""
    measure(q::Quantity, sim) -> Float64

Value of the quantity `q` in the current state of the simulation `sim`.
"""
function measure end

"""
    label(q::Quantity) -> String

Short label of the quantity `q`, used as a column name.
"""
function label end

"""
    current_kT(sim)

Temperature of the simulation at its current sweep, as seen by the moves.
"""
function current_kT(sim)
    T = eltype(sim.system.diameters)
    return conditions(sim.ensemble, sim.sweep, T).kT
end

"""
    find_move(sim, name::Symbol) -> MCMove

The move of `sim` called `name`; throws an `ArgumentError` if there is none.
"""
function find_move(sim, move_name::Symbol)
    for move in sim.moves
        name(move) === move_name && return move
    end
    names = map(name, sim.moves)
    return throw(
        ArgumentError("no move named :$(move_name) in the simulation, got $(names)")
    )
end

"""
    PotentialEnergy()

Potential energy per particle, `U / N` (including long range corrections).
"""
struct PotentialEnergy <: Quantity end
measure(::PotentialEnergy, sim) = Float64(sim.system.energy / length(sim.system))
label(::PotentialEnergy) = "energy"

"""
    Pressure()

Virial pressure, see [`pressure`](@ref), at the temperature of the current sweep. Not
defined for hard potentials; use [`VirtualPressure`](@ref) for those.
"""
struct Pressure <: Quantity end
function measure(::Pressure, sim)
    sys = sim.system
    # The virial of hard particles is a contact term that the pair sum never sees (no
    # pair is inside its cutoff), so the result would silently be the ideal gas value.
    is_hard(sys.potential) && throw(
        ArgumentError(
            "the virial pressure is not defined for the hard potential " *
            "$(nameof(typeof(sys.potential))); use VirtualPressure() instead",
        ),
    )
    return Float64(pressure(sys, current_kT(sim)))
end
label(::Pressure) = "pressure"

"""
    VirtualPressure(; xi=1e-3)

Pressure of hard particles (see [`is_hard`](@ref)) from virtual compressions
(Eppenga and Frenkel, Mol. Phys. 52, 1303 (1984)).

All box lengths are scaled by `(1 - xi)^(1/D)`, which shrinks the volume `V` by the
fraction `xi`, and the number `n` of pairs that would then overlap is counted. The
instantaneous estimator is

    P = kT (N / V + n / (xi V)),

whose average is the pressure up to corrections of order `xi`: the exact relation is
`βP = ρ - ln P_no-overlap / (xi V)` in the limit `xi → 0`, and the mean of `n` differs
from `-ln P_no-overlap` by the probability of several overlaps and by the curvature of
the contact value, both `O(xi)` relative. Smaller `xi` reduces this bias but increases
the statistical noise (fewer configurations with an overlap), so `xi` should be chosen
such that `xi N` is of order `1e-2`–`1`, and the bias checked by comparing two values.
"""
struct VirtualPressure <: Quantity
    xi::Float64
    function VirtualPressure(; xi::Real=1e-3)
        0 < xi < 1 || throw(ArgumentError("xi must be in (0, 1), got $(xi)"))
        return new(Float64(xi))
    end
end

function measure(q::VirtualPressure, sim)
    sys = sim.system
    is_hard(sys.potential) || throw(
        ArgumentError(
            "VirtualPressure needs a hard potential (is_hard), got " *
            "$(nameof(typeof(sys.potential))); use Pressure() instead",
        ),
    )
    n = compression_overlaps(sys, q.xi)
    V = Float64(volume(sys))
    return Float64(current_kT(sim)) * (length(sys) / V + n / (q.xi * V))
end
label(::VirtualPressure) = "virtual_pressure"

"""
    compression_overlaps(sys, xi) -> Int

Number of pairs that overlap (infinite pair energy) after the box of `sys` is compressed
isotropically by the volume fraction `xi`, i.e. all lengths are scaled by `f = (1 -
xi)^(1/D)`.

Pairs up to the distance `rc / f` (in the uncompressed box) are visited, which can exceed
the cutoff `sys.rc`. The cell list of `sys` is used when its cells are wide enough (cells
are built with a skin of [`CELL_SKIN`](@ref)); otherwise a temporary cell list is built.
"""
function compression_overlaps(sys::System{D,T}, xi::Real) where {D,T}
    f = T((1 - xi)^(1 / D))
    reach = sys.rc / f
    fits_cutoff(sys.box, reach) || throw(
        ArgumentError(
            "the compressed box violates the minimum image convention; use a smaller xi " *
            "or a larger box",
        ),
    )
    cells = sys.cells
    if !cells_cover(cells, sys.box, reach)
        cells = CellList(sys.frac, sys.box, reach)
    end
    return overlap_count(sys, cells, f, reach)
end

"""
    cells_cover(cells, box, reach) -> Bool

Whether the neighbor stencil of `cells` contains every pair closer than `reach`.
"""
function cells_cover(cells::CellList{D}, box::AbstractBox{D}, reach) where {D}
    widths = perpendicular_widths(box)
    for d in 1:D
        n = cells.ncells[d]
        # With one or two cells along `d` the stencil spans the whole dimension.
        (n <= 2 || widths[d] >= n * reach) || return false
    end
    return true
end

"""
    overlap_count(sys, cells, f, reach) -> Int

Kernel of [`compression_overlaps`](@ref): count the pairs `i < j` closer than `reach`
whose pair energy at the distance scaled by `f` is infinite.
"""
function overlap_count(sys::System{D,T}, cells::CellList{D,T}, f::T, reach::T) where {D,T}
    box = sys.box
    pot = sys.potential
    frac = sys.frac
    diameters = sys.diameters
    members = cells.members
    counts = cells.counts
    wrap = cells.wrap
    reach2 = reach * reach
    f2 = f * f
    n = 0
    for i in eachindex(frac)
        s = frac[i]
        σi = diameters[i]
        c = cell_coordinates(cells, s)
        @inbounds for I in CartesianIndices(cells.stencil)
            cell = neighbor_cell(wrap, c, I)
            for k in 1:Int(counts[cell])
                j = Int(members[k, cell])
                j > i || continue
                dx = min_image(box, frac[j] - s)
                r2 = dot(dx, dx)
                r2 < reach2 || continue
                u = convert(T, pair_energy(pot, f2 * r2, σi, diameters[j]))
                n += isinf(u)
            end
        end
    end
    return n
end

"""
    Density()

Number density `N / V`.
"""
struct Density <: Quantity end
measure(::Density, sim) = Float64(density(sim.system))
label(::Density) = "density"

"""
    PackingFraction()

Packing fraction, see [`packing_fraction`](@ref) (2D and 3D).
"""
struct PackingFraction <: Quantity end
measure(::PackingFraction, sim) = Float64(packing_fraction(sim.system))
label(::PackingFraction) = "packing_fraction"

"""
    Volume()

Volume (area in 2D) of the simulation box.
"""
struct Volume <: Quantity end
measure(::Volume, sim) = Float64(volume(sim.system))
label(::Volume) = "volume"

"""
    Acceptance(name::Symbol)

Acceptance rate of the move called `name` since the last [`reset_stats!`](@ref) (`NaN`
before the first attempt).
"""
struct Acceptance <: Quantity
    name::Symbol
end
measure(q::Acceptance, sim) = Float64(acceptance_rate(find_move(sim, q.name)))
label(q::Acceptance) = "acceptance_$(q.name)"

"""
    StepSize(name::Symbol)

Current step size of the move called `name` (`NaN` if it has none).
"""
struct StepSize <: Quantity
    name::Symbol
end
measure(q::StepSize, sim) = Float64(step_size(find_move(sim, q.name)))
label(q::StepSize) = "step_$(q.name)"
