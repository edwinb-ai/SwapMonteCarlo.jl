"""
    Schedule

Supertype of observation schedules. A schedule decides at which sweeps an
[`Observer`](@ref) records data through `isdue(schedule, sweep) -> Bool`.
"""
abstract type Schedule end

"""
    isdue(schedule, sweep::Integer) -> Bool

Whether an observer with `schedule` records data at `sweep`.
"""
function isdue end

"""
    Every(n)

Due every `n` sweeps, at the sweeps that are multiples of `n`. Since sweeps are counted
for the whole lifetime of a [`Simulation`](@ref), the output stays regular across
several calls of [`run!`](@ref).
"""
struct Every <: Schedule
    n::Int
    function Every(n::Integer)
        n >= 1 || throw(ArgumentError("the interval must be positive, got $(n)"))
        return new(Int(n))
    end
end

isdue(s::Every, sweep::Integer) = sweep % s.n == 0

"""
    LogSchedule(; base=1.35, first=1)

Due at roughly logarithmically spaced sweeps, for time correlation functions and aging
studies. The due sweeps are `t₁ = first` and `t_{k+1} = max(t_k + 1, ⌈base t_k⌉)`, a
strictly increasing and asymptotically geometric sequence (`1, 2, 3, 5, 7, 10, 14, 19,
26, 36, …` with the defaults).

The schedule remembers the next due sweep, so a check costs `O(1)` when sweeps are
queried in increasing order (as [`run!`](@ref) does). Querying an earlier sweep restarts
the sequence from `first`.
"""
mutable struct LogSchedule <: Schedule
    const base::Float64
    const first::Int
    previous::Int
    next::Int
    function LogSchedule(; base::Real=1.35, first::Integer=1)
        (isfinite(base) && base > 1) ||
            throw(ArgumentError("the base must be larger than 1, got $(base)"))
        first >= 0 || throw(ArgumentError("first must be non-negative, got $(first)"))
        return new(Float64(base), Int(first), -1, Int(first))
    end
end

"""
    next_log_sweep(t, base) -> Int

The due sweep after `t` in a [`LogSchedule`](@ref) with the given `base`.
"""
next_log_sweep(t::Int, base::Float64) = max(t + 1, ceil(Int, t * base))

function isdue(s::LogSchedule, sweep::Integer)
    if sweep <= s.previous
        # Restart the sequence for a query in the past.
        s.previous = -1
        s.next = s.first
    end
    while s.next < sweep
        s.previous = s.next
        s.next = next_log_sweep(s.next, s.base)
    end
    return s.next == sweep
end
