"""
    Simulation(system, ensemble, moves::Pair{<:MCMove,<:Real}...; seed, rng=Xoshiro(seed))

A Monte Carlo simulation of `system` in `ensemble` with the given moves, each paired with
its relative weight, e.g. `Displacement(0.1) => 0.9, Swap() => 0.1`.

One sweep is `N` move attempts, each move chosen at random with probability proportional
to its weight. The system and the moves are used (and modified) in place; use a fresh
set of moves for every simulation, since they hold acceptance counters and step sizes.

The random number generator is `Xoshiro(seed)` by default, with a random `seed` that is
stored (and written to output headers) so that a run can be reproduced. If `rng` is
given, `seed` is only informative.

The constructor checks that the weights are positive and finite, that the move names are
unique, that each move is compatible with the ensemble and that [`NPT`](@ref) is used
with a [`VolumeMove`](@ref) (and only then). It warns if swap moves are used on a
system whose particles all have the same diameter.

# Fields

  - `system`, `ensemble`, `moves::Tuple`, `rng`.
  - `cumweights::NTuple{K,Float64}`: normalized cumulative move weights.
  - `seed::UInt64`: seed of the default random number generator.
  - `sweep::Int`: number of sweeps done so far.
"""
mutable struct Simulation{S<:System,E<:Ensemble,M<:Tuple,W<:Tuple,R<:AbstractRNG}
    const system::S
    const ensemble::E
    const moves::M
    const cumweights::W
    const rng::R
    const seed::UInt64
    sweep::Int
end

function Simulation(
    system::System,
    ensemble::Ensemble,
    moves::Pair{<:MCMove,<:Real}...;
    seed::Integer=rand(RandomDevice(), UInt64),
    rng::AbstractRNG=Xoshiro(seed),
)
    isempty(moves) && throw(ArgumentError("a simulation needs at least one move"))
    seed >= 0 || throw(ArgumentError("the seed must be non-negative, got $(seed)"))
    movetuple = map(first, moves)
    cumweights = cumulative_weights(map(last, moves))
    check_moves(system, ensemble, movetuple)
    return Simulation(system, ensemble, movetuple, cumweights, rng, UInt64(seed), 0)
end

"""
    cumulative_weights(weights::Tuple) -> NTuple{K,Float64}

Normalized cumulative sums of the move weights, with the last entry exactly `1.0`.
Throws an `ArgumentError` unless all weights are positive and finite.
"""
function cumulative_weights(weights::NTuple{K,Real}) where {K}
    for w in weights
        (isfinite(w) && w > 0) ||
            throw(ArgumentError("move weights must be positive and finite, got $(weights)"))
    end
    total = sum(Float64, weights)
    partial = cumsum(collect(Float64, weights))
    return ntuple(k -> k == K ? 1.0 : min(partial[k] / total, 1.0), Val(K))
end

"""
    check_moves(system, ensemble, moves::Tuple)

Validate a set of moves for a simulation, see [`Simulation`](@ref).
"""
function check_moves(system::System, ensemble::Ensemble, moves::Tuple)
    names = map(name, moves)
    allunique(names) || throw(
        ArgumentError("move names must be unique, got $(names); use one move of each kind"),
    )
    for move in moves
        check_compatible(move, ensemble)
    end
    has_volume = :volume in names
    if ensemble isa NPT && !has_volume
        throw(ArgumentError("the NPT ensemble needs a VolumeMove"))
    end
    if has_volume && !(ensemble isa NPT)
        throw(ArgumentError("a VolumeMove can only be used in the NPT ensemble"))
    end
    if :swap in names && allequal(system.diameters)
        @warn "swap moves have no effect: all particles have the same diameter"
    end
    return nothing
end

"""
    sweep!(sys, moves::Tuple, cumweights, cond, rng)

Attempt `N = length(sys)` moves chosen at random according to `cumweights`, at the
conditions `cond` (see [`conditions`](@ref)). Allocation free and type stable; this is the
function barrier of the simulation loop.
"""
function sweep!(sys::System, moves::Tuple, cumweights::Tuple, cond, rng::AbstractRNG)
    for _ in 1:length(sys)
        k = choose_move(cumweights, rng)
        attempt_nth!(moves, k, sys, cond, rng)
    end
    return nothing
end

"""
Relative tolerance of the periodic comparison of the running and recomputed energies.
"""
const ENERGY_RTOL = 1e-8

"""
    run!(sim, nsweeps; observers=(), adapt=false, energy_check=1000) -> sim

Do `nsweeps` sweeps of the simulation `sim`.

Before each sweep, the sweep counter is incremented and the conditions of the ensemble
(e.g. a temperature ramp) are evaluated at the new sweep. After each sweep the
`observers` (an [`Observer`](@ref) or a collection of them) are called; they are flushed
at the end.

# Keyword arguments

  - `observers=()`: observers, e.g. `(ThermoWriter("thermo.txt", PotentialEnergy()),)`.
  - `adapt=false`: tune the step sizes toward their target acceptance (breaks detailed
    balance, use [`equilibrate!`](@ref) instead).
  - `energy_check=1000`: every `energy_check` sweeps (counted from the start of the
    simulation) the running energy is compared to the energy recomputed from scratch; a
    relative mismatch larger than `1e-8` gives a warning and the running energy is
    resynchronized. `0` disables the check.
"""
function run!(
    sim::Simulation,
    nsweeps::Integer;
    observers=(),
    adapt::Bool=false,
    energy_check::Integer=1000,
)
    nsweeps >= 0 || throw(ArgumentError("nsweeps must be non-negative, got $(nsweeps)"))
    energy_check >= 0 ||
        throw(ArgumentError("energy_check must be non-negative, got $(energy_check)"))
    obs = observers isa Observer ? (observers,) : Tuple(observers)
    for o in obs
        o isa Observer || throw(ArgumentError("expected Observer objects, got $(o)"))
    end
    run_loop!(sim, Int(nsweeps), obs, adapt, Int(energy_check))
    return sim
end

"""
    equilibrate!(sim, nsweeps; kwargs...) -> sim

[`run!`](@ref) with step size adaptation (`adapt=true`).
"""
function equilibrate!(sim::Simulation, nsweeps::Integer; kwargs...)
    return run!(sim, nsweeps; adapt=true, kwargs...)
end

"""
    run_loop!(sim, nsweeps, observers::Tuple, adapt, energy_check)

Simulation loop of [`run!`](@ref), specialized on the types of the observers.
"""
function run_loop!(
    sim::Simulation, nsweeps::Int, observers::Tuple, adapt::Bool, energy_check::Int
)
    sys = sim.system
    T = eltype(sys.diameters)
    for _ in 1:nsweeps
        sim.sweep += 1
        cond = conditions(sim.ensemble, sim.sweep, T)
        sweep!(sys, sim.moves, sim.cumweights, cond, sim.rng)
        adapt && foreach_unrolled(adapt!, sim.moves)
        if energy_check > 0 && sim.sweep % energy_check == 0
            verify_energy!(sim)
        end
        foreach_unrolled(o -> observe!(o, sim), observers)
    end
    foreach_unrolled(flush, observers)
    return sim
end

"""
    verify_energy!(sim) -> Bool

Compare the running energy of the system with the energy recomputed from scratch. If
they differ by more than a relative `1e-8` (or `100 eps`), warn and set the running
energy to the recomputed one. Returns whether the energies agreed.
"""
function verify_energy!(sim::Simulation)
    sys = sim.system
    T = eltype(sys.diameters)
    (running, recomputed) = check_energy(sys)
    rtol = max(ENERGY_RTOL, 100 * eps(T))
    if abs(running - recomputed) <= rtol * max(one(T), abs(recomputed))
        return true
    end
    @warn(
        "running energy drifted from the recomputed one; resynchronizing",
        sim.sweep,
        running,
        recomputed,
    )
    sync_energy!(sys)
    return false
end

"""
    reset_stats!(sim)

Reset the acceptance counters of all moves of `sim`.
"""
function reset_stats!(sim::Simulation)
    foreach_unrolled(reset_stats!, sim.moves)
    return sim
end

"""
    acceptance_rate(sim) -> NamedTuple

Acceptance rates of the moves since the last [`reset_stats!`](@ref), keyed by move name,
e.g. `(displacement = 0.41, swap = 0.08)`.
"""
function acceptance_rate(sim::Simulation)
    return NamedTuple{map(name, sim.moves)}(map(acceptance_rate, sim.moves))
end

"""
    step_size(sim) -> NamedTuple

Current step sizes of the moves keyed by move name (`NaN` for moves without a step).
"""
function step_size(sim::Simulation)
    return NamedTuple{map(name, sim.moves)}(map(step_size, sim.moves))
end

function Base.show(io::IO, sim::Simulation)
    print(
        io,
        "Simulation(",
        sim.system,
        ", ",
        sim.ensemble,
        ", moves=",
        join(map(name, sim.moves), "/"),
        ", sweep=",
        sim.sweep,
        ", seed=",
        sim.seed,
        ")",
    )
    return nothing
end
