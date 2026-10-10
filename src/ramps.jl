"""
    check_ramp_arguments(initial, final, nsweeps)

Throw an `ArgumentError` unless the ramp end points are finite and `nsweeps ≥ 1`.
"""
function check_ramp_arguments(initial, final, nsweeps)
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be at least 1, got $(nsweeps)"))
    (isfinite(initial) && isfinite(final)) ||
        throw(ArgumentError("ramp end points must be finite, got $(initial) and $(final)"))
    return nothing
end

"""
    clamp_between(x, a, b)

Clamp `x` to the interval between `a` and `b` (in any order). Rounding errors could
otherwise make a ramp overshoot its final value by an ulp just before the end.
"""
@inline clamp_between(x, a, b) = clamp(x, min(a, b), max(a, b))

"""
    LinearRamp(initial, final, nsweeps)

Linear interpolation from `initial` (at sweep 0) to `final` (at sweep `nsweeps`), constant
and exactly equal to `final` afterwards.

A ramp is callable, `ramp(sweep)` returns its value at `sweep`. It can be used for any
control parameter, e.g. the temperature `kT` of [`NVT`](@ref) or the pressure of
[`NPT`](@ref). Sweeps before 0 return `initial`.

# Examples

```julia
ramp = LinearRamp(2.0, 0.5, 1000)
ramp(0)    # 2.0
ramp(500)  # 1.25
ramp(5000) # 0.5
```
"""
struct LinearRamp{T<:AbstractFloat}
    initial::T
    final::T
    nsweeps::Int
    function LinearRamp(initial::Real, final::Real, nsweeps::Integer)
        check_ramp_arguments(initial, final, nsweeps)
        (a, b) = float.(promote(initial, final))
        return new{typeof(a)}(a, b, Int(nsweeps))
    end
end

"""
    (ramp::LinearRamp)(sweep::Integer)

Value of the linear ramp at `sweep`.
"""
function (ramp::LinearRamp)(sweep::Integer)
    # The end points are returned exactly, without rounding errors.
    sweep <= 0 && return ramp.initial
    sweep >= ramp.nsweeps && return ramp.final
    # Fraction of the ramp completed, strictly between 0 and 1 here.
    p = oftype(ramp.initial, sweep / ramp.nsweeps)
    value = ramp.initial + (ramp.final - ramp.initial) * p
    return clamp_between(value, ramp.initial, ramp.final)
end

"""
    ExponentialRamp(initial, final, nsweeps)

Geometric (exponential) interpolation from `initial` (at sweep 0) to `final` (at sweep
`nsweeps`), constant and exactly equal to `final` afterwards.

The value at sweep `t ≤ nsweeps` is `initial * (final / initial)^(t / nsweeps)`, so equal
numbers of sweeps change the value by equal factors, and the midpoint is the geometric
mean of the end points. Both end points must be nonzero and have the same sign. Like
[`LinearRamp`](@ref), it can be used for the temperature or the pressure.
"""
struct ExponentialRamp{T<:AbstractFloat}
    initial::T
    final::T
    nsweeps::Int
    rate::T
    function ExponentialRamp(initial::Real, final::Real, nsweeps::Integer)
        check_ramp_arguments(initial, final, nsweeps)
        (a, b) = float.(promote(initial, final))
        (iszero(a) || iszero(b) || signbit(a) != signbit(b)) && throw(
            ArgumentError(
                "exponential ramp end points must be nonzero and have the same sign, " *
                "got $(initial) and $(final)",
            ),
        )
        return new{typeof(a)}(a, b, Int(nsweeps), log(b / a))
    end
end

"""
    (ramp::ExponentialRamp)(sweep::Integer)

Value of the exponential ramp at `sweep`.
"""
function (ramp::ExponentialRamp)(sweep::Integer)
    # The end points are returned exactly, without rounding errors.
    sweep <= 0 && return ramp.initial
    sweep >= ramp.nsweeps && return ramp.final
    # Fraction of the ramp completed, strictly between 0 and 1 here.
    p = oftype(ramp.initial, sweep / ramp.nsweeps)
    return clamp_between(ramp.initial * exp(ramp.rate * p), ramp.initial, ramp.final)
end

"""
    value_at(x, sweep::Integer)

Value of a control parameter at `sweep`: `x` itself for a number, `x(sweep)` for a
callable such as a [`LinearRamp`](@ref).
"""
value_at(x::Real, sweep::Integer) = x
value_at(f, sweep::Integer) = f(sweep)
