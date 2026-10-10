"""
    box_for_density(Val(D), N, ρ) -> CubicBox

Hypercubic box holding `N` particles at number density `ρ`.
"""
function box_for_density(::Val{D}, N::Integer, ρ::Real) where {D}
    N >= 1 || throw(ArgumentError("the number of particles must be positive, got $(N)"))
    ρ > 0 || throw(ArgumentError("the density must be positive, got $(ρ)"))
    return CubicBox(Val(D), (N / ρ)^(1 / D))
end

"""
    box_for_packing_fraction(Val(D), diameters, φ) -> CubicBox

Hypercubic box in which spheres with the given `diameters` cover a fraction `φ` of the
volume (`D` = 2 or 3), see [`packing_fraction`](@ref).
"""
function box_for_packing_fraction(
    ::Val{D}, diameters::AbstractVector{<:Real}, φ::Real
) where {D}
    isempty(diameters) && throw(ArgumentError("at least one diameter is needed"))
    φ > 0 || throw(ArgumentError("the packing fraction must be positive, got $(φ)"))
    σ = float.(diameters)
    return CubicBox(Val(D), (particle_volume(Val(D), σ) / φ)^(1 / D))
end

"""
    lattice_positions(box::AbstractBox{D,T}, N) -> Vector{SVector{D,T}}

Cartesian positions of the first `N` sites of a hypercubic lattice with
`m = ceil(N^(1/D))` sites along each box vector, offset by half a lattice spacing from the
box faces. In a rectangular box the spacing differs between directions.
"""
function lattice_positions(box::AbstractBox{D,T}, N::Integer) where {D,T}
    N >= 0 || throw(ArgumentError("the number of particles must be non-negative, got $(N)"))
    # Integer search, since N^(1/D) can be off by one ulp (27^(1/3) > 3).
    m = max(1, round(Int, N^(1 / D)))
    while m^D < N
        m += 1
    end
    while m > 1 && (m - 1)^D >= N
        m -= 1
    end
    positions = Vector{SVector{D,T}}(undef, N)
    for (k, I) in enumerate(CartesianIndices(ntuple(_ -> m, Val(D))))
        k > N && break
        s = SVector{D,T}(ntuple(d -> (T(I[d]) - T(1) / 2) / m, Val(D)))
        positions[k] = cartesian(box, s)
    end
    return positions
end

"""
    random_positions(rng, box::AbstractBox{D,T}, N) -> Vector{SVector{D,T}}

`N` Cartesian positions drawn uniformly in the box (overlaps are not avoided).
"""
function random_positions(rng::AbstractRNG, box::AbstractBox{D,T}, N::Integer) where {D,T}
    return [cartesian(box, random_fractional(rng, box)) for _ in 1:N]
end

"""
    random_fractional(rng, box::AbstractBox{D,T}) -> SVector{D,T}

Uniformly random fractional position in `[0, 1)^D`.
"""
@inline function random_fractional(rng::AbstractRNG, ::AbstractBox{D,T}) where {D,T}
    return SVector{D,T}(ntuple(_ -> rand(rng, T), Val(D)))
end

"""
    insert_hard_spheres(rng, box, diameters; max_tries=10^6) -> Vector{SVector{D,T}}

Cartesian positions of non-overlapping hard spheres with the given `diameters`, placed by
random sequential insertion, largest first. Two particles overlap when they are closer
than `(σi + σj) / 2`, which also excludes overlaps for non-additive mixing with `ε ≥ 0`.
The positions are returned in the order of `diameters`.

Throws an `ErrorException` if a particle cannot be placed in `max_tries` attempts.
Random sequential insertion jams around `φ ≈ 0.55` in 2D and `φ ≈ 0.38` in 3D; for
denser states, insert at a lower packing fraction and compress (NPT).
"""
function insert_hard_spheres(
    rng::AbstractRNG,
    box::AbstractBox{D,T},
    diameters::AbstractVector{<:Real};
    max_tries::Integer=10^6,
) where {D,T}
    N = length(diameters)
    σ = T[d for d in diameters]
    frac = zeros(SVector{D,T}, N)
    N == 0 && return frac
    σmax = maximum(σ)
    fits_cutoff(box, σmax) || throw(
        ArgumentError("the largest diameter $(σmax) exceeds half the smallest box width"),
    )
    cells = empty_cell_list(box, σmax, N)
    for i in sortperm(σ; rev=true)
        placed = false
        for _ in 1:max_tries
            s = random_fractional(rng, box)
            if !overlaps(cells, box, frac, σ, s, σ[i])
                frac[i] = s
                cell_insert!(cells, i, s)
                placed = true
                break
            end
        end
        placed || error(
            "insert_hard_spheres: could not place particle $(i) (σ = $(σ[i])) in " *
            "$(max_tries) attempts; the packing fraction is too high for random " *
            "sequential insertion",
        )
    end
    return [cartesian(box, s) for s in frac]
end

"""
    overlaps(cells, box, frac, σ, s, σi) -> Bool

Whether a sphere of diameter `σi` at fractional position `s` overlaps any sphere already
in `cells` (additive diameters).
"""
function overlaps(
    cells::CellList{D,T},
    box::AbstractBox{D,T},
    frac::Vector{SVector{D,T}},
    σ::Vector{T},
    s::SVector{D,T},
    σi::T,
) where {D,T}
    c = cell_coordinates(cells, s)
    for I in CartesianIndices(cells.stencil)
        cell = neighbor_cell(cells.wrap, c, I)
        for k in 1:Int(cells.counts[cell])
            j = Int(cells.members[k, cell])
            dx = min_image(box, frac[j] - s)
            σij = (σi + σ[j]) / 2
            dot(dx, dx) < σij * σij && return true
        end
    end
    return false
end
