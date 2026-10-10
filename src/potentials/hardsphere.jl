"""
    HardSphere(; mixing=Additive())

Hard sphere potential: infinite energy if two particles overlap, `r < σij`, and zero
otherwise (touching particles do not overlap). The pair diameter `σij` is given by the
mixing rule.

Hard spheres have no virial pressure; measure the pressure with
[`VirtualPressure`](@ref) instead.
"""
struct HardSphere{M<:MixingRule} <: Potential
    mixing::M
end

HardSphere(; mixing::MixingRule=Additive()) = HardSphere(mixing)

"""
    cutoff(pot::HardSphere, σi, σj)

Pair cutoff, equal to the contact distance `σij`.
"""
@inline function cutoff(p::HardSphere, σi, σj)
    F = pair_float_type(σi, σj)
    return F(mix(p.mixing, σi, σj))
end

"""
    pair_energy(pot::HardSphere, r2, σi, σj)

`Inf` if `r2 < σij^2` (overlap) and zero otherwise.
"""
@inline function pair_energy(p::HardSphere, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    return ifelse(F(r2) < σij * σij, F(Inf), zero(F))
end

"""
    pair_virial(pot::HardSphere, r2, σi, σj)

Not defined: the hard sphere force is a delta function at contact. Throws an error that
points to [`VirtualPressure`](@ref).
"""
function pair_virial(::HardSphere, r2, σi, σj)
    return throw(
        ArgumentError(
            "the pair virial of HardSphere is not defined (the force is a delta function " *
            "at contact); measure the pressure with VirtualPressure() instead",
        ),
    )
end

"""
    is_repulsive(pot::HardSphere)

Hard sphere energies are non-negative.
"""
is_repulsive(::HardSphere) = true

"""
    is_hard(pot::HardSphere)

Hard sphere energies are either zero or infinite.
"""
is_hard(::HardSphere) = true
