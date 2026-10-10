"""
    Potential

Supertype of all pair potentials.

# Interface

Required:

  - `pair_energy(pot, r2, σi, σj) -> u`: pair energy at squared distance `r2` between
    particles of diameters `σi` and `σj`. It must return exactly zero beyond the cutoff and
    it must be symmetric in `(σi, σj)`. Alternatively, define the
    MolecularDynamics.jl-style `evaluate(pot, r, σi, σj) -> (u, f)`.
  - `cutoff(pot, σi, σj)`: distance beyond which the pair energy is zero.

Optional:

  - `pair_virial(pot, r2, σi, σj) -> -r du/dr`, needed for the virial pressure.
  - `is_repulsive(pot)`: `true` if every pair energy is non-negative (allows early
    rejection of moves).
  - `is_hard(pot)`: `true` if pair energies are either zero or infinite.
  - `energy_lrc(pot, sys)`, `pressure_lrc(pot, sys)`: long range corrections.
  - `max_cutoff(pot, diameters)`: largest pair cutoff in a set of diameters.

# Example

```julia
import SwapMonteCarlo: Potential, pair_energy, cutoff

struct GaussianCore <: Potential end
pair_energy(::GaussianCore, r2, σi, σj) = r2 < 9 ? exp(-r2) : zero(r2)
cutoff(::GaussianCore, σi, σj) = 3.0
```
"""
abstract type Potential end

"""
    evaluate(pot, r, σi, σj) -> (u, f)

MolecularDynamics.jl-style pair energy and force `f = -du/dr`. Defining this method is an
alternative to defining [`pair_energy`](@ref) and [`pair_virial`](@ref).
"""
function evaluate(pot::Potential, r, σi, σj)
    return error(
        "$(typeof(pot)) does not implement the pair potential interface. Define " *
        "`pair_energy(pot, r2, σi, σj)` and `cutoff(pot, σi, σj)` (and optionally " *
        "`pair_virial`), or the MolecularDynamics.jl-style `evaluate(pot, r, σi, σj)`.",
    )
end

"""
    evaluate_r2(pot, r2, σi, σj) -> (u, f/r)

Pair energy and force divided by distance, from the squared distance.
"""
@inline function evaluate_r2(pot::Potential, r2, σi, σj)
    r = sqrt(r2)
    (u, f) = evaluate(pot, r, σi, σj)
    return u, f / r
end

"""
    pair_energy(pot, r2, σi, σj) -> u

Pair energy at squared distance `r2`. Must be zero beyond the cutoff and symmetric in
`(σi, σj)`.
"""
@inline pair_energy(pot::Potential, r2, σi, σj) = first(evaluate_r2(pot, r2, σi, σj))

"""
    pair_virial(pot, r2, σi, σj) -> w

Pair virial `w = -r du/dr = r f` at squared distance `r2`.
"""
@inline pair_virial(pot::Potential, r2, σi, σj) = r2 * last(evaluate_r2(pot, r2, σi, σj))

"""
    cutoff(pot, σi, σj)

Distance beyond which the pair energy between diameters `σi` and `σj` vanishes.
"""
function cutoff(pot::Potential, σi, σj)
    return error("$(typeof(pot)) must implement `cutoff(pot, σi, σj)`.")
end

"""
    max_cutoff(pot, diameters)

Largest pair cutoff among particles with the given diameters. The default checks the
pairs formed by the smallest and largest diameters, which is exact for the mixing rules
provided by this package.
"""
function max_cutoff(pot::Potential, diameters::AbstractVector)
    (a, b) = extrema(diameters)
    return max(cutoff(pot, a, a), cutoff(pot, a, b), cutoff(pot, b, b))
end

"""
    is_repulsive(pot) -> Bool

Whether all pair energies are non-negative. Enables early termination of energy sums.
"""
is_repulsive(::Potential) = false

"""
    is_hard(pot) -> Bool

Whether pair energies are either zero or infinite (hard-core potentials).
"""
is_hard(::Potential) = false

"""
    energy_lrc(pot, sys)

Long range (tail) correction to the total energy. Defaults to zero.
"""
energy_lrc(::Potential, sys) = zero(eltype(sys.diameters))

"""
    pressure_lrc(pot, sys)

Long range (tail) correction to the pressure. Defaults to zero.
"""
pressure_lrc(::Potential, sys) = zero(eltype(sys.diameters))

"""
    check_symmetry(pot, diameters, rng; npairs=256)

Throw an `ArgumentError` if `pair_energy` is not symmetric in the diameters for randomly
chosen pairs of `diameters`. Swap moves rely on this symmetry.
"""
function check_symmetry(pot::Potential, diameters::AbstractVector, rng; npairs::Int=256)
    n = length(diameters)
    n < 2 && return nothing
    for _ in 1:npairs
        σi = diameters[rand(rng, 1:n)]
        σj = diameters[rand(rng, 1:n)]
        rc = cutoff(pot, σi, σj)
        r2 = (rand(rng) * rc)^2
        uij = pair_energy(pot, r2, σi, σj)
        uji = pair_energy(pot, r2, σj, σi)
        isequal(uij, uji) || throw(
            ArgumentError(
                "pair_energy of $(typeof(pot)) is not symmetric in the diameters: " *
                "u($(σi), $(σj)) = $(uij) but u($(σj), $(σi)) = $(uji)",
            ),
        )
    end
    return nothing
end

"""
    check_cutoff(pot, diameters, rng; npairs=256)

Throw an `ArgumentError` if `pair_energy` is not exactly zero beyond the pair cutoff for
randomly chosen pairs of `diameters`. Energy sums visit every particle of the neighboring
cells, also those beyond the cutoff, so a potential that is not truncated by its own
`pair_energy` would give results that depend on the cell geometry.
"""
function check_cutoff(pot::Potential, diameters::AbstractVector, rng; npairs::Int=256)
    n = length(diameters)
    for _ in 1:npairs
        σi = diameters[rand(rng, 1:n)]
        σj = diameters[rand(rng, 1:n)]
        rc = cutoff(pot, σi, σj)
        # Just beyond the cutoff and somewhat further out.
        for r in (nextfloat(float(rc)), (1 + rand(rng)) * rc)
            u = pair_energy(pot, r^2, σi, σj)
            iszero(u) || throw(
                ArgumentError(
                    "pair_energy of $(typeof(pot)) must be exactly zero beyond the " *
                    "cutoff, but u(r = $(r)) = $(u) for diameters $(σi) and $(σj) " *
                    "with cutoff $(rc)",
                ),
            )
        end
    end
    return nothing
end
