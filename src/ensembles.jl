"""
    Ensemble

Supertype of the thermodynamic ensembles. An ensemble provides the external conditions
(temperature, pressure) through [`conditions`](@ref).
"""
abstract type Ensemble end

"""
    NVT(kT)

Canonical ensemble at temperature `kT`, either a number or a callable of the sweep number
such as [`LinearRamp`](@ref). `kT = 0` gives a zero temperature quench. The temperature
must be finite and non-negative: a number is checked here, a callable at every sweep by
[`conditions`](@ref).
"""
struct NVT{K} <: Ensemble
    kT::K
    function NVT(kT::K) where {K}
        if kT isa Real && !(isfinite(kT) && kT >= 0)
            throw(
                ArgumentError("the temperature must be finite and non-negative, got $(kT)")
            )
        end
        return new{K}(kT)
    end
end

"""
    NPT(kT, pressure)

Isothermal-isobaric ensemble at temperature `kT > 0` and pressure `pressure`. Each can be
a number or a callable of the sweep number. Requires a [`VolumeMove`](@ref). The
temperature must be positive and finite and the pressure finite: numbers are checked here,
callables at every sweep by [`conditions`](@ref).
"""
struct NPT{K,P} <: Ensemble
    kT::K
    pressure::P
    function NPT(kT::K, pressure::P) where {K,P}
        if kT isa Real && !(isfinite(kT) && kT > 0)
            throw(
                ArgumentError(
                    "the temperature must be positive and finite in NPT, got $(kT)"
                ),
            )
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

Throws an `ArgumentError` naming the sweep if the temperature is negative (zero in
[`NPT`](@ref)) or not finite, or if the pressure is not finite, for example a ramp that
goes below zero.
"""
function conditions(e::NVT, sweep::Integer, ::Type{T}) where {T}
    kT = T(value_at(e.kT, sweep))
    (isfinite(kT) && kT >= 0) || invalid_condition("temperature", kT, sweep)
    return (; kT)
end
function conditions(e::NPT, sweep::Integer, ::Type{T}) where {T}
    kT = T(value_at(e.kT, sweep))
    pressure = T(value_at(e.pressure, sweep))
    (isfinite(kT) && kT > 0) || invalid_condition("temperature", kT, sweep)
    isfinite(pressure) || invalid_condition("pressure", pressure, sweep)
    return (; kT, pressure)
end

"""
    invalid_condition(what, value, sweep)

Throw the `ArgumentError` of an invalid temperature or pressure at a sweep. Kept out of
line, so that [`conditions`](@ref) stays small.
"""
@noinline function invalid_condition(what::String, value, sweep::Integer)
    return throw(
        ArgumentError(
            "invalid $(what) $(value) at sweep $(sweep): the temperature must be finite " *
            "and non-negative (positive in NPT) and the pressure finite",
        ),
    )
end
