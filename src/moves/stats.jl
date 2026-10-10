"""
    MCMove

Supertype of Monte Carlo moves.

# Interface

Required:

  - `attempt!(move, sys, cond, rng) -> Bool`: propose a change of `sys`, accept or reject
    it with the Metropolis rule at the conditions `cond` (see [`conditions`](@ref)),
    record the outcome with [`record!`](@ref) and return whether it was accepted.
  - `name(move) -> Symbol`: short name used in acceptance reports.
  - A field `stats::MoveStats` (or a method of [`move_stats`](@ref)).

Optional:

  - [`adaptive_step`](@ref)`(move)`: the tunable [`AdaptiveStep`](@ref), if any.
  - [`check_compatible`](@ref)`(move, ensemble)`: throw if the move cannot be used with
    the ensemble.
"""
abstract type MCMove end

"""
    attempt!(move, sys, cond, rng) -> Bool

Propose a change of `sys` with `move`, accept or reject it with the Metropolis rule at
the external conditions `cond` (a `NamedTuple` such as `(; kT)` or `(; kT, pressure)`, see
[`conditions`](@ref)), record the outcome and return whether it was accepted. A rejected
attempt leaves `sys` unchanged.

All moves use the Metropolis rule with the threshold drawn first: with `ξ = randexp(rng)`
and `ln A = -ΔU/kT + bias`, a move is accepted iff `U_new ≤ U_max := U_old + kT (bias +
ξ)`. Since `P(ξ ≥ x) = e^{-x}`, this happens with probability `min(1, A)`. A zero energy
change is always accepted, also at `kT = 0` (a quench), and `U_max` is known before the
new energy is computed, so energy sums of repulsive potentials stop early.
"""
function attempt! end

"""
    MoveStats()

Number of attempted and accepted moves since the last [`reset_stats!`](@ref).
"""
mutable struct MoveStats
    attempts::Int
    accepts::Int
end
MoveStats() = MoveStats(0, 0)

"""
    AdaptiveStep(value; target, bounds, interval)

Step size that is tuned toward a target acceptance ratio during equilibration.

Every `interval` attempts the step is multiplied by `ratio / target`, where `ratio` is the
acceptance ratio in that window. The factor is limited to `[1/1.5, 1.5]` and the step is
clamped to `bounds = (lower, upper)`. Tuning is only applied when [`adapt!`](@ref) is
called, which [`equilibrate!`](@ref) does and [`run!`](@ref) does not, so that production
runs keep a fixed proposal and satisfy detailed balance.
"""
mutable struct AdaptiveStep
    value::Float64
    const target::Float64
    const lower::Float64
    const upper::Float64
    const interval::Int
    window_attempts::Int
    window_accepts::Int
end

"""Largest factor by which a step size can change in one adaptation."""
const MAX_STEP_FACTOR = 1.5

function AdaptiveStep(
    value::Real; target::Real, bounds::Tuple{Real,Real}, interval::Integer
)
    (lower, upper) = bounds
    0 < lower <= upper || throw(ArgumentError("invalid step bounds $(bounds)"))
    lower <= value <= upper ||
        throw(ArgumentError("step $(value) outside of the bounds $(bounds)"))
    0 < target < 1 || throw(ArgumentError("target acceptance must be in (0, 1)"))
    interval >= 1 || throw(ArgumentError("adaptation interval must be positive"))
    return AdaptiveStep(value, target, lower, upper, interval, 0, 0)
end

"""
    adapt!(step::AdaptiveStep)

Update the step size if a full adaptation window has been collected.
"""
function adapt!(a::AdaptiveStep)
    a.window_attempts < a.interval && return nothing
    ratio = a.window_accepts / a.window_attempts
    factor = clamp(ratio / a.target, inv(MAX_STEP_FACTOR), MAX_STEP_FACTOR)
    a.value = clamp(a.value * factor, a.lower, a.upper)
    a.window_attempts = 0
    a.window_accepts = 0
    return nothing
end

"""
    move_stats(move) -> MoveStats

Acceptance counters of a move. Defaults to the field `move.stats`.
"""
move_stats(move::MCMove) = move.stats

"""
    adaptive_step(move) -> Union{AdaptiveStep,Nothing}

The tunable step of a move, or `nothing` if the move has no step size.
"""
adaptive_step(::MCMove) = nothing

"""
    name(move) -> Symbol

Short name of a move, used as a key in [`acceptance_rate`](@ref).
"""
function name end

"""
    check_compatible(move, ensemble)

Throw an `ArgumentError` if `move` cannot be used in `ensemble`. Defaults to no check.
"""
check_compatible(::MCMove, ::Ensemble) = nothing

"""
    record!(move, accepted::Bool) -> Bool

Count an attempt of `move` (and its acceptance) and return `accepted`.
"""
@inline function record!(move::MCMove, accepted::Bool)
    s = move_stats(move)
    s.attempts += 1
    s.accepts += accepted
    a = adaptive_step(move)
    if a !== nothing
        a.window_attempts += 1
        a.window_accepts += accepted
    end
    return accepted
end

"""
    adapt!(move)

Tune the step size of `move` if it has one, see [`AdaptiveStep`](@ref).
"""
function adapt!(move::MCMove)
    a = adaptive_step(move)
    a === nothing || adapt!(a)
    return nothing
end

"""
    reset_stats!(move)

Reset the acceptance counters (and the adaptation window) of `move`.
"""
function reset_stats!(move::MCMove)
    s = move_stats(move)
    s.attempts = 0
    s.accepts = 0
    a = adaptive_step(move)
    if a !== nothing
        a.window_attempts = 0
        a.window_accepts = 0
    end
    return nothing
end

"""
    acceptance_rate(move) -> Float64

Fraction of accepted attempts since the last reset (`NaN` if there were no attempts).
"""
function acceptance_rate(move::MCMove)
    s = move_stats(move)
    return s.attempts == 0 ? NaN : s.accepts / s.attempts
end

"""
    step_size(move) -> Float64

Current step size of `move`, or `NaN` if it has none.
"""
function step_size(move::MCMove)
    a = adaptive_step(move)
    return a === nothing ? NaN : a.value
end
