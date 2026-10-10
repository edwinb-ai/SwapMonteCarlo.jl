"""
    MixingRule

Supertype of rules that combine two particle diameters `σi` and `σj` into an effective
pair diameter `σij`. A new rule must implement [`mix`](@ref) and must be symmetric,
`mix(rule, a, b) == mix(rule, b, a)`.
"""
abstract type MixingRule end

"""
    Additive()

Additive mixing rule, `σij = (σi + σj) / 2`.
"""
struct Additive <: MixingRule end

"""
    NonAdditive(ε)

Non-additive mixing rule of Ninarello, Berthier and Coslovich (PRX 7, 021039, 2017),
`σij = (σi + σj) / 2 * (1 - ε |σi - σj|)`, with `0 ≤ ε < 1`.

Restricting to `ε ≥ 0` guarantees that `σij ≤ max(σi, σj)`, so the largest pair cutoff is
always the one between the two largest particles.
"""
struct NonAdditive{T<:AbstractFloat} <: MixingRule
    epsilon::T
    function NonAdditive(epsilon::Real)
        0 <= epsilon < 1 ||
            throw(ArgumentError("non-additivity must satisfy 0 ≤ ε < 1, got $(epsilon)"))
        e = float(epsilon)
        return new{typeof(e)}(e)
    end
end

"""
    mix(rule, σi, σj)

Effective pair diameter of two particles with diameters `σi` and `σj`.
"""
@inline mix(::Additive, σi, σj) = (σi + σj) / 2
@inline mix(m::NonAdditive, σi, σj) = (σi + σj) / 2 * (1 - m.epsilon * abs(σi - σj))
