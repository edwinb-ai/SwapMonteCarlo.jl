"""
    PseudoHS(; epsilon=1.0, mixing=Additive())

Pseudo hard sphere potential of Jover et al. (J. Chem. Phys. 137, 144505, 2012): a
(50, 49) Mie potential cut and shifted at its minimum,

    u(r) = A ε [(σij/r)^50 - (σij/r)^49] + ε    for r < (50/49) σij,

with `A = 50 (50/49)^49`, and zero beyond, as in MolecularDynamics.jl. The pair diameter
`σij` is given by the mixing rule.
"""
struct PseudoHS{T<:AbstractFloat,M<:MixingRule} <: Potential
    epsilon::T
    # Prefactor A = 50 (50/49)^49 and cutoff xc = 50/49 in units of σij.
    A::T
    xc::T
    mixing::M
end

function PseudoHS(; epsilon::Real=1.0, mixing::MixingRule=Additive())
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $(epsilon)"))
    e = float(epsilon)
    T = typeof(e)
    # Compute the constants in extended precision so that they are correctly rounded.
    A = T(50 * (big(50) / 49)^49)
    xc = T(big(50) / 49)
    return PseudoHS{T,typeof(mixing)}(e, A, xc, mixing)
end

"""
    cutoff(pot::PseudoHS, σi, σj)

Pair cutoff `(50/49) σij`, at the minimum of the Mie potential.
"""
@inline function cutoff(p::PseudoHS, σi, σj)
    F = pair_float_type(σi, σj)
    return F(p.xc) * F(mix(p.mixing, σi, σj))
end

"""
    pseudohs_powers(s2) -> (s^49, s^50)

Powers 49 and 50 of `s = sqrt(s2)` by repeated squaring, avoiding the generic `^`.
"""
@inline function pseudohs_powers(s2)
    s4 = s2 * s2
    s8 = s4 * s4
    s16 = s8 * s8
    s48 = (s16 * s8) * (s16 * s8)
    return s48 * sqrt(s2), s48 * s2
end

"""
    pair_energy(pot::PseudoHS, r2, σi, σj)

Pseudo hard sphere pair energy from the squared distance. Exactly zero for
`r2 ≥ cutoff(pot, σi, σj)^2` and never negative.
"""
@inline function pair_energy(p::PseudoHS, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    rc = F(p.xc) * σij
    r2 < rc * rc || return zero(F)
    (s49, s50) = pseudohs_powers(σij * σij / F(r2))
    u = F(p.epsilon) * (F(p.A) * (s50 - s49) + one(F))
    # Near the cutoff cancellations can give tiny negative values; clamping them keeps
    # the energy non-negative, as `is_repulsive` promises.
    return max(u, zero(F))
end

"""
    pair_virial(pot::PseudoHS, r2, σi, σj)

Pseudo hard sphere pair virial `-r du/dr = A ε [50 (σij/r)^50 - 49 (σij/r)^49]`.
"""
@inline function pair_virial(p::PseudoHS, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    rc = F(p.xc) * σij
    r2 < rc * rc || return zero(F)
    (s49, s50) = pseudohs_powers(σij * σij / F(r2))
    return F(p.epsilon) * F(p.A) * (50 * s50 - 49 * s49)
end

"""
    is_repulsive(pot::PseudoHS)

The pseudo hard sphere energy is non-negative.
"""
is_repulsive(::PseudoHS) = true
