"""
    select_move(cumweights::NTuple{K,Float64}, u) -> Int

Index of the move selected by the uniform random number `u ∈ [0, 1)`: the first `k` with
`u < cumweights[k]`. The cumulative weights are normalized, so the last entry is exactly
`1.0` and a move is always found.
"""
@inline function select_move(cumweights::NTuple{K,Float64}, u::Real) where {K}
    for k in 1:(K - 1)
        @inbounds u < cumweights[k] && return k
    end
    return K
end

"""
    choose_move(cumweights, rng) -> Int

Draw the index of the next move to attempt. With a single move no random number is drawn.
"""
@inline choose_move(::NTuple{1,Float64}, rng::AbstractRNG) = 1
@inline function choose_move(cumweights::NTuple{K,Float64}, rng::AbstractRNG) where {K}
    return select_move(cumweights, rand(rng))
end

"""
    attempt_nth!(moves::Tuple, k, sys, cond, rng) -> Bool

Attempt the `k`-th move of `moves`. The tuple is unrolled recursively, so every call of
[`attempt!`](@ref) is statically dispatched even if the moves have different types.
"""
@inline function attempt_nth!(moves::Tuple, k::Int, sys, cond, rng::AbstractRNG)
    k == 1 && return attempt!(first(moves), sys, cond, rng)::Bool
    return attempt_nth!(Base.tail(moves), k - 1, sys, cond, rng)
end

# Unreachable for valid indices; it terminates the recursion.
@inline attempt_nth!(::Tuple{}, k::Int, sys, cond, rng::AbstractRNG) = false

"""
    foreach_unrolled(f, t::Tuple)

Call `f(x)` for every element `x` of the tuple `t`, unrolled at compile time so that the
calls are type stable for heterogeneous tuples. Returns `nothing`.
"""
@inline function foreach_unrolled(f::F, t::Tuple) where {F}
    f(first(t))
    return foreach_unrolled(f, Base.tail(t))
end
@inline foreach_unrolled(::F, ::Tuple{}) where {F} = nothing
