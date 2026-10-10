"""
    Swap(; max_dsigma=Inf)

Swap move: the diameters of two distinct, uniformly chosen particles are exchanged
(Grigera and Parisi, PRE 63, 045102, 2001; Ninarello, Berthier and Coslovich, PRX 7,
021039, 2017). Positions do not change.

Pairs whose diameters differ by more than `max_dsigma` are rejected without computing
any energy (they still count as attempts). Since this restriction is symmetric, the
proposal stays symmetric and detailed balance holds.
"""
struct Swap <: MCMove
    max_dsigma::Float64
    stats::MoveStats
    function Swap(max_dsigma::Real, stats::MoveStats)
        max_dsigma >= 0 ||
            throw(ArgumentError("max_dsigma must be non-negative, got $(max_dsigma)"))
        return new(Float64(max_dsigma), stats)
    end
end

Swap(; max_dsigma::Real=Inf) = Swap(max_dsigma, MoveStats())

"""
    name(::Swap)

The swap move is reported as `:swap`.
"""
name(::Swap) = :swap

"""
    attempt!(move::Swap, sys, cond, rng) -> Bool

Exchange the diameters of a uniformly chosen ordered pair `i ≠ j` and accept the swap
with the Metropolis rule at temperature `cond.kT`.

The energy change is computed from the energies of `i` and `j` with all particles except
`i` and `j`: the `i`-`j` pair energy does not change because pair energies are symmetric
in the diameters (checked when the [`System`](@ref) is built). Swapping equal diameters
changes nothing and is accepted without computing energies. Allocation free.
"""
function attempt!(m::Swap, sys::System{D,T}, cond::NamedTuple, rng::AbstractRNG) where {D,T}
    N = length(sys)
    N < 2 && return record!(m, false)
    # Uniform ordered pair of distinct particles without a rejection loop.
    i = rand(rng, 1:N)
    j = rand(rng, 1:(N - 1))
    j += j >= i
    σi = sys.diameters[i]
    σj = sys.diameters[j]
    abs(σi - σj) > m.max_dsigma && return record!(m, false)
    # The energy change of swapping equal diameters is exactly zero, which the
    # Metropolis rule always accepts.
    σi == σj && return record!(m, true)
    pot = sys.potential
    si = sys.frac[i]
    sj = sys.frac[j]
    kT = T(cond.kT)
    if is_hard(pot)
        # Hard particles in a valid configuration never interact, and the new energies
        # are non-negative, so the swap is rejected as soon as either partial sum exceeds
        # what is left of the threshold (the first overlap). A sum that stops early is
        # only known to exceed its limit, so the limits are checked as well as the total.
        U_old = zero(T)
        U_max = U_old + kT * randexp(rng, T)
        Ui = particle_energy(sys, si, σj, i, j, U_max)
        Ui <= U_max || return record!(m, false)
        Uj = particle_energy(sys, sj, σi, j, i, U_max - Ui)
        accepted = (Uj <= U_max - Ui) & (Ui + Uj <= U_max)
    else
        # The old and new energies of each particle are at the same position, so both
        # are summed in one pass over its neighbors (see `two_pair_sums`).
        (Ui_old, Ui) = two_pair_sums(sys, si, σi, si, σj, i, j)
        (Uj_old, Uj) = two_pair_sums(sys, sj, σj, sj, σi, j, i)
        U_old = Ui_old + Uj_old
        U_max = U_old + kT * randexp(rng, T)
        accepted = Ui + Uj <= U_max
    end
    if accepted
        sys.diameters[i] = σj
        sys.diameters[j] = σi
        sys.energy += (Ui + Uj) - U_old
    end
    return record!(m, accepted)
end
