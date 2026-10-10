"""
    ShiftMode

Supertype of the ways a truncated [`LennardJones`](@ref) potential is shifted at its
cutoff: [`NoShift`](@ref), [`EnergyShift`](@ref) or [`ForceShift`](@ref).
"""
abstract type ShiftMode end

"""
    NoShift()

Plain truncation: the energy jumps to zero at the cutoff.
"""
struct NoShift <: ShiftMode end

"""
    EnergyShift()

Truncated and shifted potential, `u(r) - u(rc)`, continuous at the cutoff.
"""
struct EnergyShift <: ShiftMode end

"""
    ForceShift()

Shifted-force potential, `u(r) - u(rc) - (r - rc) u'(rc)`, whose energy and force both
vanish at the cutoff. It needs a `sqrt` inside the cutoff.
"""
struct ForceShift <: ShiftMode end

"""
    LennardJones(; epsilon=1.0, xc=2.5, shift=EnergyShift(), mixing=Additive(),
                 tail_correction=false)

Lennard-Jones potential `u(r) = 4ε [(σij/r)^12 - (σij/r)^6]` truncated at `rc = xc σij`,
with the pair diameter `σij` given by the mixing rule.

Since the cutoff is relative to `σij`, the shift constants do not depend on the diameters
and are precomputed.

# Keyword arguments

  - `epsilon`: well depth.
  - `xc`: cutoff in units of `σij`.
  - `shift`: [`NoShift`](@ref), [`EnergyShift`](@ref) or [`ForceShift`](@ref).
  - `mixing`: mixing rule, see [`Additive`](@ref) and [`NonAdditive`](@ref).
  - `tail_correction`: add the standard long range corrections of the full (untruncated)
    potential beyond the cutoff to the energy and pressure, see [`energy_lrc`](@ref).
    Only available with `Additive` mixing. With a shifted potential these corrections
    account only for the missing tail, not for the shift inside the cutoff.
"""
struct LennardJones{T<:AbstractFloat,M<:MixingRule,S<:ShiftMode} <: Potential
    epsilon::T
    xc::T
    # Reduced energy φ(xc) = 4 (xc^-12 - xc^-6) and derivative φ'(xc) at the cutoff.
    phic::T
    dphic::T
    shift::S
    mixing::M
    tail_correction::Bool
end

function LennardJones(;
    epsilon::Real=1.0,
    xc::Real=2.5,
    shift::ShiftMode=EnergyShift(),
    mixing::MixingRule=Additive(),
    tail_correction::Bool=false,
)
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $(epsilon)"))
    xc > 0 || throw(ArgumentError("xc must be positive, got $(xc)"))
    if tail_correction && !(mixing isa Additive)
        throw(
            ArgumentError(
                "tail corrections of LennardJones are only implemented for Additive " *
                "mixing, got $(mixing)",
            ),
        )
    end
    (e, x) = float.(promote(epsilon, xc))
    T = typeof(e)
    ix2 = inv(x * x)
    ix6 = ix2 * ix2 * ix2
    phic = 4 * ix6 * (ix6 - 1)
    dphic = -24 * ix6 * (2 * ix6 - 1) / x
    return LennardJones{T,typeof(mixing),typeof(shift)}(
        e, x, phic, dphic, shift, mixing, tail_correction
    )
end

"""
    cutoff(pot::LennardJones, σi, σj)

Pair cutoff `xc σij`.
"""
@inline function cutoff(p::LennardJones, σi, σj)
    F = pair_float_type(σi, σj)
    return F(p.xc) * F(mix(p.mixing, σi, σj))
end

"""
    lj_shift(shift, pot, x2, F)

Reduced energy subtracted from the plain Lennard-Jones energy at reduced squared distance
`x2 = (r/σij)^2`, in the floating point type `F`.
"""
@inline lj_shift(::NoShift, p::LennardJones, x2, ::Type{F}) where {F} = zero(F)
@inline lj_shift(::EnergyShift, p::LennardJones, x2, ::Type{F}) where {F} = F(p.phic)
@inline function lj_shift(::ForceShift, p::LennardJones, x2, ::Type{F}) where {F}
    return F(p.phic) + (sqrt(x2) - F(p.xc)) * F(p.dphic)
end

"""
    lj_virial_shift(shift, pot, x2, F)

Reduced virial subtracted from the plain Lennard-Jones virial, `-x d(shift)/dx`.
"""
@inline lj_virial_shift(::ShiftMode, p::LennardJones, x2, ::Type{F}) where {F} = zero(F)
@inline function lj_virial_shift(::ForceShift, p::LennardJones, x2, ::Type{F}) where {F}
    return -sqrt(x2) * F(p.dphic)
end

"""
    pair_energy(pot::LennardJones, r2, σi, σj)

Truncated (and possibly shifted) Lennard-Jones pair energy from the squared distance.
Exactly zero for `r2 ≥ cutoff(pot, σi, σj)^2`.
"""
@inline function pair_energy(p::LennardJones, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    rc = F(p.xc) * σij
    r2 < rc * rc || return zero(F)
    x2 = F(r2) / (σij * σij)
    ix6 = inv(x2 * x2 * x2)
    return F(p.epsilon) * (4 * ix6 * (ix6 - 1) - lj_shift(p.shift, p, x2, F))
end

"""
    pair_virial(pot::LennardJones, r2, σi, σj)

Lennard-Jones pair virial `-r du/dr`, consistent with the shift mode of `pot`.
"""
@inline function pair_virial(p::LennardJones, r2, σi, σj)
    F = pair_float_type(r2, σi, σj)
    σij = F(mix(p.mixing, σi, σj))
    rc = F(p.xc) * σij
    r2 < rc * rc || return zero(F)
    x2 = F(r2) / (σij * σij)
    ix6 = inv(x2 * x2 * x2)
    return F(p.epsilon) * (24 * ix6 * (2 * ix6 - 1) - lj_virial_shift(p.shift, p, x2, F))
end

"""
    surface_constant(Val(D), T)

Surface area of the unit sphere in `D = 2` or `3` dimensions, in the float type `T`.
"""
surface_constant(::Val{2}, ::Type{T}) where {T} = 2 * T(π)
surface_constant(::Val{3}, ::Type{T}) where {T} = 4 * T(π)
function surface_constant(::Val{D}, ::Type) where {D}
    return throw(
        ArgumentError("tail corrections are only implemented for D = 2 and 3, got $(D)")
    )
end

"""
    additive_pair_moment(diameters, Val(D))

Sum over all ordered pairs `(i, j)`, including `i = j`, of `σij^D` with the additive
mixing rule `σij = (σi + σj) / 2`.

The binomial expansion `(σi + σj)^D = Σ_k binomial(D, k) σi^k σj^(D-k)` turns the double
sum into products of the single-particle moments `M_k = Σ_i σi^k`, which costs `O(N)`.
Including the `i = j` terms (`N²` pairs instead of `N(N - 1)`) is the usual convention of
the monodisperse formulas and a relative error of order `1/N`.
"""
function additive_pair_moment(diameters::AbstractVector{T}, ::Val{D}) where {T,D}
    M = ntuple(_ -> zero(T), Val(D + 1))
    for σ in diameters
        # Powers σ^0, …, σ^D; the closure must not capture the reassigned `M`.
        powers = ntuple(k -> σ^(k - 1), Val(D + 1))
        M = M .+ powers
    end
    s = zero(T)
    for k in 0:D
        s += binomial(D, k) * M[k + 1] * M[D - k + 1]
    end
    return s / 2^D
end

"""
    lj_tail_integrals(pot::LennardJones, Val(D)) -> (KU, KP)

Reduced tail integrals of the full Lennard-Jones potential beyond `xc`:
`KU = 4 [xc^(D-12)/(12-D) - xc^(D-6)/(6-D)]` for the energy, so that
`∫_rc^∞ u(r) r^(D-1) dr = ε σij^D KU`, and `KP = 4 [12 xc^(D-12)/(12-D) - 6 xc^(D-6)/(6-D)]`
for the virial.
"""
function lj_tail_integrals(p::LennardJones{T}, ::Val{D}) where {T,D}
    a12 = p.xc^(D - 12) / (12 - D)
    a6 = p.xc^(D - 6) / (6 - D)
    return 4 * (a12 - a6), 4 * (12 * a12 - 6 * a6)
end

"""
    energy_lrc(pot::LennardJones, sys)

Standard tail correction to the total energy, `U_lrc = S_D/(2V) Σ_ij ∫_rc^∞ u_ij r^(D-1)
dr`, where `S_D` is the surface of the unit sphere, for the full Lennard-Jones potential
beyond the cutoff. The pair sum runs over all ordered pairs including `i = j`, see
`additive_pair_moment`. Zero unless `pot.tail_correction`.
"""
function energy_lrc(p::LennardJones, sys)
    T = eltype(sys.diameters)
    p.tail_correction || return zero(T)
    vD = Val(dimension(sys.box))
    (KU, _) = lj_tail_integrals(p, vD)
    m = additive_pair_moment(sys.diameters, vD)
    return T(surface_constant(vD, T) * p.epsilon * KU * m / (2 * volume(sys.box)))
end

"""
    pressure_lrc(pot::LennardJones, sys)

Standard tail correction to the pressure, `P_lrc = S_D/(2 D V²) Σ_ij ∫_rc^∞ w_ij r^(D-1)
dr` with the pair virial `w = -r du/dr` of the full Lennard-Jones potential. Zero unless
`pot.tail_correction`.
"""
function pressure_lrc(p::LennardJones, sys)
    T = eltype(sys.diameters)
    p.tail_correction || return zero(T)
    D = dimension(sys.box)
    vD = Val(D)
    (_, KP) = lj_tail_integrals(p, vD)
    m = additive_pair_moment(sys.diameters, vD)
    V = volume(sys.box)
    return T(surface_constant(vD, T) * p.epsilon * KP * m / (2 * D * V * V))
end
