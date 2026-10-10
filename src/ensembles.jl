"""
    Ensemble

Supertype of the thermodynamic ensembles. An ensemble provides the external conditions
(temperature, pressure) through [`conditions`](@ref).
"""
abstract type Ensemble end

"""
    NVT(kT)

Canonical ensemble at temperature `kT`, either a number or a callable of the sweep number
such as [`LinearRamp`](@ref). `kT = 0` gives a zero temperature quench.
"""
struct NVT{K} <: Ensemble
    kT::K
    function NVT(kT::K) where {K}
        if kT isa Real && !(kT >= 0)
            throw(ArgumentError("the temperature must be non-negative, got $(kT)"))
        end
        return new{K}(kT)
    end
end

"""
    NPT(kT, pressure)

Isothermal-isobaric ensemble at temperature `kT > 0` and pressure `pressure`. Each can be
a number or a callable of the sweep number. Requires a [`VolumeMove`](@ref).
"""
struct NPT{K,P} <: Ensemble
    kT::K
    pressure::P
    function NPT(kT::K, pressure::P) where {K,P}
        if kT isa Real && !(kT > 0)
            throw(ArgumentError("the temperature must be positive in NPT, got $(kT)"))
        end
        if pressure isa Real && !isfinite(pressure)
            throw(ArgumentError("the pressure must be finite, got $(pressure)"))
        end
        return new{K,P}(kT, pressure)
    end
end

"""
    conditions(ensemble, sweep, T) -> NamedTuple

External conditions at a given sweep, converted to the float type `T`: `(; kT)` for
[`NVT`](@ref) and `(; kT, pressure)` for [`NPT`](@ref). Evaluated once per sweep, so the
moves only see plain numbers.
"""
conditions(e::NVT, sweep::Integer, ::Type{T}) where {T} = (; kT=T(value_at(e.kT, sweep)))
function conditions(e::NPT, sweep::Integer, ::Type{T}) where {T}
    return (; kT=T(value_at(e.kT, sweep)), pressure=T(value_at(e.pressure, sweep)))
end
