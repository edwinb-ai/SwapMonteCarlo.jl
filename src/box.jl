"""
    AbstractBox{D,T}

Supertype of periodic simulation boxes in `D` dimensions with floating point type `T`.

Particle positions are stored in *fractional* (reduced) coordinates `s ∈ [0, 1)^D`, so a
box only needs to know how to map between fractional and Cartesian coordinates. A new box
type must implement [`volume`](@ref), `scale`, `cartesian`, `fractional`, `min_image` and
`perpendicular_widths`.
"""
abstract type AbstractBox{D,T<:AbstractFloat} end

"""
    OrthorhombicBox(lengths::SVector{D,T})
    OrthorhombicBox(lengths...)

Periodic box with orthogonal sides of lengths `lengths`.
"""
struct OrthorhombicBox{D,T} <: AbstractBox{D,T}
    lengths::SVector{D,T}
    inv_lengths::SVector{D,T}
    function OrthorhombicBox(lengths::SVector{D,T}) where {D,T<:AbstractFloat}
        all(>(zero(T)), lengths) ||
            throw(ArgumentError("box lengths must be positive, got $(lengths)"))
        return new{D,T}(lengths, inv.(lengths))
    end
end

OrthorhombicBox(lengths::SVector{D,<:Real}) where {D} = OrthorhombicBox(float.(lengths))
OrthorhombicBox(lengths::Real...) = OrthorhombicBox(SVector(float.(promote(lengths...))))

"""
    CubicBox(Val(D), L)

Periodic hypercubic box of side `L` in `D` dimensions.
"""
function CubicBox(::Val{D}, L::Real) where {D}
    return OrthorhombicBox(SVector(ntuple(_ -> float(L), Val(D))))
end

Base.eltype(::AbstractBox{D,T}) where {D,T} = T
dimension(::AbstractBox{D}) where {D} = D

"""
    volume(box)

Volume (area in 2D) of the box.
"""
@inline volume(b::OrthorhombicBox) = prod(b.lengths)

"""
    scale(box, f)

Return a new box whose sides are multiplied by `f` (a scalar or an `SVector`).
"""
@inline scale(b::OrthorhombicBox, f) = OrthorhombicBox(b.lengths .* f)

"""
    perpendicular_widths(box) -> SVector{D}

Distances between opposite faces of the box.
"""
@inline perpendicular_widths(b::OrthorhombicBox) = b.lengths

"""
    cartesian(box, s)

Convert fractional coordinates `s` into Cartesian coordinates.
"""
@inline cartesian(b::OrthorhombicBox, s::SVector) = b.lengths .* s

"""
    fractional(box, x)

Convert Cartesian coordinates (or displacements) `x` into fractional coordinates.
"""
@inline fractional(b::OrthorhombicBox, x::SVector) = x .* b.inv_lengths

"""
    min_image(box, Δs) -> SVector

Cartesian minimum image vector of the fractional separation `Δs`.
"""
@inline function min_image(b::OrthorhombicBox, Δs::SVector)
    return b.lengths .* (Δs .- round.(Δs))
end

"""
    wrap_fractional(s)

Wrap fractional coordinates into `[0, 1)`.

`s - floor(s)` can round up to exactly `1` for tiny negative `s` (e.g. `-1e-17`), which is
mapped back to `0` so that cell indices always stay in range.
"""
@inline function wrap_fractional(s::SVector{D,T}) where {D,T}
    w = s .- floor.(s)
    return ifelse.(w .>= one(T), zero(T), w)
end

"""
    fits_cutoff(box, rc) -> Bool

Whether a cutoff `rc` is compatible with the minimum image convention, `rc ≤ L_min / 2`.
"""
fits_cutoff(b::AbstractBox, rc) = 2 * rc <= minimum(perpendicular_widths(b))
