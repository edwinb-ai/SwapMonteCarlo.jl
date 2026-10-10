"""
    pair_float_type(xs...)

Floating point type in which a pair kernel is evaluated: the promotion of the argument
types. Kernels convert the parameters of the potential to this type, so a `Float32` system
keeps computing in `Float32` even if the potential was built from `Float64` parameters.
"""
@inline pair_float_type(xs...) = float(promote_type(map(typeof, xs)...))

"""
    PolySoftSphere(; epsilon=1.0, xc=1.25, mixing=NonAdditive(0.2))

Polydisperse soft sphere (SS12) potential of Ninarello, Berthier and Coslovich (PRX 7,
021039, 2017),

    u(r) = ε [(σij/r)^12 + c0 + c2 (r/σij)^2 + c4 (r/σij)^4]    for r < xc σij,

and zero beyond. The constants `c0 = -28/xc^12`, `c2 = 48/xc^14` and `c4 = -21/xc^16`
make the energy and its first two derivatives vanish at the cutoff. The pair diameter
`σij` is given by the mixing rule.

# Keyword arguments

  - `epsilon`: energy scale.
  - `xc`: cutoff in units of `σij`.
  - `mixing`: mixing rule, see [`Additive`](@ref) and [`NonAdditive`](@ref).
"""
struct PolySoftSphere{T<:AbstractFloat,M<:MixingRule} <: Potential
    epsilon::T
    xc::T
    c0::T
    c2::T
    c4::T
    mixing::M
end

function PolySoftSphere(;
    epsilon::Real=1.0, xc::Real=1.25, mixing::MixingRule=NonAdditive(0.2)
)
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $(epsilon)"))
    xc > 0 || throw(ArgumentError("xc must be positive, got $(xc)"))
    (e, x) = float.(promote(epsilon, xc))
    T = typeof(e)
    x2 = x * x
    ix12 = inv(x2 * x2 * x2)^2
    c0 = -28 * ix12
    c2 = 48 * ix12 / x2
    c4 = -21 * ix12 / (x2 * x2)
    return PolySoftSphere{T,typeof(mixing)}(e, x, c0, c2, c4, mixing)
end

"""
    cutoff(pot::PolySoftSphere, σi, σj)

Pair cutoff `xc σij`.
"""
@inline function cutoff(p::PolySoftSphere, σi, σj)
    F = pair_float_type(σi, σj)
    return F(p.xc) * F(mix(p.mixing, σi, σj))
end

"""
    pair_energy(pot::PolySoftSphere, r2, σi, σj)

SS12 pair energy from the squared distance, without `sqrt` or `^`. Exactly zero for
`r2 ≥ cutoff(pot, σi, σj)^2` and never negative.
"""
@inline function pair_energy(p::PolySoftSphere, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    rc = F(p.xc) * σij
    r2 < rc * rc || return zero(F)
    x2 = F(r2) / (σij * σij)
    ix6 = inv(x2 * x2 * x2)
    u = F(p.epsilon) * (ix6 * ix6 + F(p.c0) + x2 * (F(p.c2) + F(p.c4) * x2))
    # Near the cutoff cancellations can give tiny negative values; clamping them keeps
    # the energy non-negative, as `is_repulsive` promises.
    return max(u, zero(F))
end

"""
    pair_virial(pot::PolySoftSphere, r2, σi, σj)

SS12 pair virial `-r du/dr = ε (12 x^-12 - 2 c2 x^2 - 4 c4 x^4)` with `x = r/σij`.
"""
@inline function pair_virial(p::PolySoftSphere, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    rc = F(p.xc) * σij
    r2 < rc * rc || return zero(F)
    x2 = F(r2) / (σij * σij)
    ix6 = inv(x2 * x2 * x2)
    return F(p.epsilon) * (12 * ix6 * ix6 - x2 * (2 * F(p.c2) + 4 * F(p.c4) * x2))
end

"""
    is_repulsive(pot::PolySoftSphere)

The SS12 pair energy is non-negative, so energy sums may stop early.
"""
is_repulsive(::PolySoftSphere) = true
