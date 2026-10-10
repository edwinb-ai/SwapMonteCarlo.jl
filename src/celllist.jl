"""
    CELL_SKIN

Relative hysteresis of the cell list. A rebuild chooses cells at least `rc * (1 +
CELL_SKIN) / range` wide (see [`cell_range`](@ref)), so the box can shrink by about this
factor before the cells become too narrow and expand by about this factor before more
cells are worth it.
"""
const CELL_SKIN = 0.02

"""
    CROWDED_CELL

Average number of particles per cell of range 1 from which a cell list uses range 2, see
[`cell_range`](@ref).
"""
const CROWDED_CELL = 10

"""
    cell_range(box, rc, nparticles) -> Int

Range `r` of a new cell list: cells are at least `rc / r` wide and the neighbors of a
particle are searched in the `(2r + 1)^D` block of cells around its own. Returns 2 if
cells of range 1 (at least `rc` wide) would hold on average at least
[`CROWDED_CELL`](@ref) particles, and 1 otherwise.

# Performance

Narrower cells fit the cutoff sphere more tightly: the block of cells searched covers
`((2r + 1) / r)^D rc^D`, which is `27 rc³` for `r = 1` and `15.6 rc³` for `r = 2` in 3D
(`9 rc²` and `6.25 rc²` in 2D), so there are fewer candidates. In exchange, `(2r + 1)^D`
cells have to be visited, and each costs a (mostly mispredicted) loop exit, which only
pays off if the cells are crowded. Measured on displacement and swap moves (see
`benchmark/README.md`): with 11 to 16 particles per cell of range 1, range 2 is 10–30%
faster in 3D and up to 10% in 2D; with 8 or fewer it is 15% to 3 times slower in 3D.
"""
function cell_range(box::AbstractBox, rc::Real, nparticles::Integer)
    n = cell_counts(perpendicular_widths(box), rc, nparticles)
    return nparticles >= CROWDED_CELL * prod(n) ? 2 : 1
end

"""
    CellList{D,T}

Cell list in fractional coordinates for `O(1)` neighbor searches in Monte Carlo moves.

The unit hypercube `[0, 1)^D` is split into `n_d` cells along each dimension `d`, so cell
membership does not change when the box is scaled (only the number of cells may need to
change, see [`cells_valid`](@ref)). Every cell is at least `rc / range` wide, where `rc`
is the largest cutoff, so all neighbors of a particle are found in the `(2 range + 1)^D`
block of cells around its own (see [`cell_range`](@ref)).

Each cell stores its particles in a column of a dense matrix and removes particles by
swapping in the last one, which makes [`cell_move!`](@ref) `O(1)`.

# Fields

  - `range::Int`: number of neighboring cells searched on each side of a cell.
  - `ncells::NTuple{D,Int}`: number of cells `n_d` along each dimension.
  - `scale::SVector{D,T}`: the same numbers as floats, to compute cell coordinates.
  - `strides::NTuple{D,Int}`: linear index strides of the cell coordinates.
  - `stencil::NTuple{D,Int}`: number of distinct neighboring cell coordinates along each
    dimension, `min(2 range + 1, n_d)`.
  - `wrap::NTuple{D,Matrix{Int32}}`: `wrap[d][k, c + 1]` is the contribution to the
    (0-based) linear cell index of the `k`-th neighboring coordinate of the (0-based)
    coordinate `c` along dimension `d`, periodic images included and duplicates removed.
  - `members::Matrix{Int32}`: `members[k, cell]` is the `k`-th particle in `cell`.
  - `counts::Vector{Int32}`: number of particles in each cell.
  - `cell_of::Vector{Int32}`: linear cell of each particle (`0` if not in the list).
  - `slot_of::Vector{Int32}`: position of each particle in the column of its cell.
"""
mutable struct CellList{D,T<:AbstractFloat}
    const range::Int
    ncells::NTuple{D,Int}
    scale::SVector{D,T}
    strides::NTuple{D,Int}
    stencil::NTuple{D,Int}
    wrap::NTuple{D,Matrix{Int32}}
    members::Matrix{Int32}
    counts::Vector{Int32}
    cell_of::Vector{Int32}
    slot_of::Vector{Int32}
end

"""
    CellList(frac, box, rc; range=cell_range(box, rc, length(frac)))

Cell list of the particles at fractional positions `frac` in `box`, for interactions of
range at most `rc`.
"""
function CellList(
    frac::AbstractVector{SVector{D,T}},
    box::AbstractBox{D,T},
    rc;
    range::Int=cell_range(box, rc, length(frac)),
) where {D,T}
    cells = empty_cell_list(box, rc, length(frac); range=range)
    for i in eachindex(frac)
        cell_insert!(cells, i, frac[i])
    end
    return cells
end

"""
    empty_cell_list(box, rc, nparticles; range=cell_range(box, rc, nparticles))

Cell list with room for particles `1:nparticles`, none of which is in the list yet. Add
them with [`cell_insert!`](@ref).
"""
function empty_cell_list(
    box::AbstractBox{D,T},
    rc,
    nparticles::Integer;
    range::Int=cell_range(box, rc, nparticles),
) where {D,T}
    range >= 1 || throw(ArgumentError("the cell range must be positive, got $(range)"))
    cells = CellList{D,T}(
        range,
        ntuple(_ -> 1, Val(D)),
        ones(SVector{D,T}),
        ntuple(_ -> 1, Val(D)),
        ntuple(_ -> 1, Val(D)),
        ntuple(_ -> zeros(Int32, 2 * range + 1, 1), Val(D)),
        Matrix{Int32}(undef, 0, 1),
        zeros(Int32, 1),
        zeros(Int32, nparticles),
        zeros(Int32, nparticles),
    )
    set_geometry!(cells, cell_counts(perpendicular_widths(box), rc / range, nparticles))
    return cells
end

"""
    cell_counts(widths, width, nparticles) -> NTuple{D,Int}

Number of cells along each dimension for a box with perpendicular `widths` and cells at
least `width` wide (the cutoff divided by the range of the cell list).

Cells are at least `width * (1 + CELL_SKIN)` wide. The total number of cells is capped at
`max(2 nparticles, 1)` (see [`cap_cell_counts`](@ref)), which matters for very short (or
zero) cutoffs and dilute systems; capping only makes cells wider, so neighbor searches
stay correct.
"""
function cell_counts(widths::SVector{D,T}, width::Real, nparticles::Integer) where {D,T}
    ncap = max(2 * Int(nparticles), 1)
    target = width * (1 + CELL_SKIN)
    n = ntuple(Val(D)) do d
        x = target > 0 ? widths[d] / target : Inf
        return isfinite(x) ? clamp(floor(Int, min(x, ncap)), 1, ncap) : ncap
    end
    return cap_cell_counts(n, ncap)
end

"""
    cap_cell_counts(n::NTuple{D,Int}, ncap) -> NTuple{D,Int}

Reduce the cell counts `n` until their product is at most `ncap`: scale all counts down
by the same factor (rounding up), which keeps the cells as isotropic as the box, then
decrement the largest count until the total fits. After the scaling only a few
decrements are needed.
"""
function cap_cell_counts(n::NTuple{D,Int}, ncap::Int) where {D}
    total = prod(Float64, n)
    total <= ncap && return n
    f = (ncap / total)^(1 / D)
    capped = ntuple(d -> clamp(ceil(Int, n[d] * f), 1, n[d]), Val(D))
    while prod(Float64, capped) > ncap
        d = argmax(capped)
        capped = Base.setindex(capped, capped[d] - 1, d)
    end
    return capped
end

"""
    wrap_table(n, stride, range) -> Matrix{Int32}

Periodic neighbor table of one dimension with `n` cells. Entry `[k, c + 1]` is `stride`
times the `k`-th distinct coordinate among `c - range, …, c + range` (modulo `n`), for
`k ≤ min(2 range + 1, n)`; with fewer cells than that, every coordinate appears once,
starting from `c`.
"""
function wrap_table(n::Int, stride::Int, range::Int)
    width = 2 * range + 1
    table = zeros(Int32, width, n)
    for c in 0:(n - 1)
        if n >= width
            # The coordinates c - range, …, c + range are all distinct.
            for (k, offset) in enumerate((-range):range)
                table[k, c + 1] = mod(c + offset, n) * stride
            end
        else
            # Fewer cells than the stencil is wide: every coordinate once, from c on.
            for k in 1:n
                table[k, c + 1] = mod(c + k - 1, n) * stride
            end
        end
    end
    return table
end

"""
    set_geometry!(cells, ncells)

Reset `cells` to `ncells` empty cells along each dimension. Allocates.
"""
function set_geometry!(cells::CellList{D,T}, ncells::NTuple{D,Int}) where {D,T}
    # Column-major strides; the generator keeps the product type stable (a tuple of
    # length `d - 1` would not be).
    strides = ntuple(d -> prod((ncells[k] for k in 1:(d - 1)); init=1), Val(D))
    total = prod(ncells)
    nparticles = length(cells.cell_of)
    capacity = 2 * cld(nparticles, total) + 8
    cells.ncells = ncells
    cells.scale = SVector{D,T}(ncells)
    cells.strides = strides
    width = 2 * cells.range + 1
    cells.stencil = ntuple(d -> min(width, ncells[d]), Val(D))
    cells.wrap = ntuple(d -> wrap_table(ncells[d], strides[d], cells.range), Val(D))
    cells.members = zeros(Int32, capacity, total)
    cells.counts = zeros(Int32, total)
    fill!(cells.cell_of, 0)
    fill!(cells.slot_of, 0)
    return cells
end

"""
    cell_coordinates(cells, s) -> NTuple{D,Int}

0-based cell coordinates of the fractional position `s ∈ [0, 1)^D`. Products `s * n` that
round up to `n` (and invalid positions) are clamped into range.
"""
@inline function cell_coordinates(cells::CellList{D,T}, s::SVector{D,T}) where {D,T}
    n = cells.ncells
    x = s .* cells.scale
    return ntuple(d -> clamp(unsafe_trunc(Int, x[d]), 0, n[d] - 1), Val(D))
end

"""
    neighbor_cell(wrap, c, I) -> Int

1-based linear index of the neighbor `I` (a `CartesianIndex` into the stencil) of the
cell with 0-based coordinates `c`.
"""
@inline function neighbor_cell(
    wrap::NTuple{D,Matrix{Int32}}, c::NTuple{D,Int}, I::CartesianIndex{D}
) where {D}
    return 1 + sum(ntuple(d -> Int(@inbounds wrap[d][I[d], c[d] + 1]), Val(D)))
end

"""
    linear_cell(cells, s) -> Int

1-based linear index of the cell containing the fractional position `s`.
"""
@inline function linear_cell(cells::CellList{D}, s::SVector{D}) where {D}
    c = cell_coordinates(cells, s)
    return 1 + sum(c .* cells.strides)
end

"""
    cell_insert!(cells, i, s)

Add particle `i`, at fractional position `s`, to its cell. The particle must not be in the
list already.
"""
@inline function cell_insert!(cells::CellList{D}, i::Integer, s::SVector{D}) where {D}
    cell_push!(cells, i, linear_cell(cells, s))
    return cells
end

"""
    cell_push!(cells, i, cell)

Append particle `i` to the linear cell `cell`, growing the storage if it is full.
"""
@inline function cell_push!(cells::CellList, i::Integer, cell::Int)
    k = Int(cells.counts[cell]) + 1
    k > size(cells.members, 1) && grow_cells!(cells)
    @inbounds begin
        cells.members[k, cell] = i
        cells.counts[cell] = k
        cells.cell_of[i] = cell
        cells.slot_of[i] = k
    end
    return cells
end

"""
    grow_cells!(cells)

Double the number of particles each cell can hold.
"""
@noinline function grow_cells!(cells::CellList)
    old = cells.members
    members = zeros(Int32, 2 * size(old, 1), size(old, 2))
    members[axes(old, 1), :] .= old
    cells.members = members
    return cells
end

"""
    cell_move!(cells, i::Int, s_new)

Update the cell of particle `i` after it moved to the fractional position `s_new`. `O(1)`
and allocation free (unless a cell overflows its storage).
"""
@inline function cell_move!(cells::CellList{D}, i::Integer, s_new::SVector{D}) where {D}
    new = linear_cell(cells, s_new)
    old = Int(cells.cell_of[i])
    new == old && return cells
    @inbounds begin
        # Swap-remove `i` from its old cell.
        last = cells.counts[old]
        slot = cells.slot_of[i]
        moved = cells.members[last, old]
        cells.members[slot, old] = moved
        cells.slot_of[moved] = slot
        cells.counts[old] = last - one(last)
    end
    cell_push!(cells, i, new)
    return cells
end

"""
    rebuild_cells!(cells, frac, box, rc)

Rebuild `cells` from scratch for the fractional positions `frac` in `box`, choosing the
number of cells again. Allocates.
"""
function rebuild_cells!(
    cells::CellList{D,T}, frac::AbstractVector{SVector{D,T}}, box::AbstractBox{D,T}, rc
) where {D,T}
    width = rc / cells.range
    set_geometry!(cells, cell_counts(perpendicular_widths(box), width, length(frac)))
    for i in eachindex(frac)
        cell_insert!(cells, i, frac[i])
    end
    return cells
end

"""
    cells_valid(cells, box, rc) -> Bool

Whether the cells can still be used in `box` with cutoff `rc`: every cell must be at least
`rc / range` wide (along dimensions with more than `2 range` cells; with fewer, the
stencil spans the whole dimension) and the box must not have grown so much that a rebuild
would choose more cells. Allocation free; call [`rebuild_cells!`](@ref) when it returns
`false`.
"""
function cells_valid(cells::CellList{D}, box::AbstractBox{D}, rc) where {D}
    widths = perpendicular_widths(box)
    n = cells.ncells
    r = cells.range
    wanted = cell_counts(widths, rc / r, length(cells.cell_of))
    # A tiny safety margin keeps rounding in the cell coordinates harmless.
    margin = 1 + 8 * eps(eltype(widths))
    for d in 1:D
        (n[d] <= 2 * r || r * widths[d] >= n[d] * rc * margin) || return false
        wanted[d] > n[d] && return false
    end
    return true
end
