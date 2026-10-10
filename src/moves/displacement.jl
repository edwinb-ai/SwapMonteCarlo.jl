"""
    Displacement(δ=0.1; target=0.4, interval=1000, bounds=(1e-4, Inf))

Single particle displacement move: a random particle is displaced uniformly in the cube
`[-δ, δ]^D`. The step `δ` is capped at half the smallest box width, so that the proposal
stays symmetric, and is tuned toward the acceptance ratio `target` during equilibration
(see [`AdaptiveStep`](@ref)).

# Arguments

  - `δ`: initial maximum displacement along each axis.

# Keyword arguments

  - `target`: target acceptance ratio of the step size tuning.
  - `interval`: number of attempts between step size updates.
  - `bounds`: lower and upper bounds of the step size.
"""
struct Displacement <: MCMove
    step::AdaptiveStep
    stats::MoveStats
end

function Displacement(
    δ::Real=0.1;
    target::Real=0.4,
    interval::Integer=1000,
    bounds::Tuple{Real,Real}=(1e-4, Inf),
)
    step = AdaptiveStep(δ; target=target, bounds=bounds, interval=interval)
    return Displacement(step, MoveStats())
end

"""
    name(::Displacement)

The displacement move is reported as `:displacement`.
"""
name(::Displacement) = :displacement

"""
    adaptive_step(move::Displacement)

The tunable maximum displacement of the move.
"""
adaptive_step(m::Displacement) = m.step

"""
    attempt!(move::Displacement, sys, cond, rng) -> Bool

Displace a uniformly chosen particle by a uniform random vector in `[-δ, δ]^D` and accept
the move with the Metropolis rule at temperature `cond.kT`. The proposal is symmetric, so
there is no bias. Allocation free.
"""
function attempt!(
    m::Displacement, sys::System{D,T}, cond::NamedTuple, rng::AbstractRNG
) where {D,T}
    N = length(sys)
    i = rand(rng, 1:N)
    box = sys.box
    pot = sys.potential
    # A step larger than half the box would make the proposal asymmetric through the
    # periodic images, so it is capped.
    δ = T(min(m.step.value, minimum(perpendicular_widths(box)) / 2))
    u = SVector{D,T}(ntuple(_ -> 2 * rand(rng, T) - one(T), Val(D)))
    s_old = sys.frac[i]
    σ = sys.diameters[i]
    s_new = wrap_fractional(s_old + fractional(box, δ .* u))
    kT = T(cond.kT)
    if !is_hard(pot) && linear_cell(sys.cells, s_new) == sys.cells.cell_of[i]
        # Usual case: the particle stays in its cell, so the old and new energies have the
        # same neighbor candidates and are summed in one pass (bitwise equal to separate
        # sums, see `two_pair_sums`).
        (U_old, U_new) = two_pair_sums(sys, s_old, σ, s_new, σ, Int(i), 0)
        U_max = U_old + kT * randexp(rng, T)
    else
        # Hard particles in a valid configuration never interact.
        U_old = is_hard(pot) ? zero(T) : particle_energy(sys, i)
        U_max = U_old + kT * randexp(rng, T)
        # The old position of `i` is excluded, so the cell list need not be updated
        # first. For repulsive potentials the sum stops early once it exceeds `U_max`.
        U_new = particle_energy(sys, s_new, σ, i, 0, U_max)
    end
    accepted = U_new <= U_max
    if accepted
        sys.frac[i] = s_new
        cell_move!(sys.cells, i, s_new)
        sys.energy += U_new - U_old
    end
    return record!(m, accepted)
end
