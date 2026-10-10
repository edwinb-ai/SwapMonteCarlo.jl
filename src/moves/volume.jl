"""
    VolumeMove(δlnV=0.01; target=0.3, interval=100, bounds=(1e-5, 1.0))

Isotropic volume move for the NPT ensemble: `ln V` is changed by a uniform random amount
in `[-δlnV, δlnV]` and all positions are scaled with the box (fractional coordinates do
not change).

# Acceptance

In the NPT ensemble the density of the volume and the fractional coordinates `s^N` is
`∝ V^N e^{-β(U + PV)}`. Since `dV = V d(ln V)`, the density in `(ln V, s^N)` is
`∝ V^(N+1) e^{-β(U + PV)}`, and with a symmetric proposal in `ln V` the Metropolis
acceptance is `min(1, A)` with

    ln A = -β (ΔU + P ΔV) + (N + 1) Δln V.

Boxes that violate the minimum image convention for the largest cutoff are rejected
(see `fits_cutoff`).

# Arguments

  - `δlnV`: initial maximum change of `ln V`.

# Keyword arguments

  - `target`: target acceptance ratio of the step size tuning.
  - `interval`: number of attempts between step size updates.
  - `bounds`: lower and upper bounds of the step size.
"""
struct VolumeMove <: MCMove
    step::AdaptiveStep
    stats::MoveStats
end

function VolumeMove(
    δlnV::Real=0.01;
    target::Real=0.3,
    interval::Integer=100,
    bounds::Tuple{Real,Real}=(1e-5, 1.0),
)
    step = AdaptiveStep(δlnV; target=target, bounds=bounds, interval=interval)
    return VolumeMove(step, MoveStats())
end

"""
    name(::VolumeMove)

The volume move is reported as `:volume`.
"""
name(::VolumeMove) = :volume

"""
    adaptive_step(move::VolumeMove)

The tunable maximum change of `ln V`.
"""
adaptive_step(m::VolumeMove) = m.step

"""
    check_compatible(::VolumeMove, ::NVT)

Volume moves change the volume, so they cannot be used at constant volume.
"""
function check_compatible(::VolumeMove, ::NVT)
    return throw(ArgumentError("VolumeMove requires the NPT ensemble"))
end

"""
    attempt!(move::VolumeMove, sys, cond, rng) -> Bool

Scale the box isotropically by a random change of `ln V` and accept it with the
Metropolis rule at temperature `cond.kT` and pressure `cond.pressure`.

The new energy is recomputed from scratch (including the long range correction, which
depends on the volume), except for hard potentials on expansion, which cannot create
overlaps. The cell list is rebuilt only when the number of cells has to change, which is
the only case in which the move allocates.
"""
function attempt!(
    m::VolumeMove, sys::System{D,T}, cond::NamedTuple, rng::AbstractRNG
) where {D,T}
    N = length(sys)
    kT = T(cond.kT)
    P = T(cond.pressure)
    box_old = sys.box
    dlnV = T(m.step.value) * (2 * rand(rng, T) - one(T))
    box_new = scale(box_old, exp(dlnV / D))
    fits_cutoff(box_new, sys.rc) || return record!(m, false)
    ΔV = volume(box_new) - volume(box_old)
    # ln A = -β(ΔU + PΔV) + (N + 1) Δln V, see the docstring of `VolumeMove`.
    U_max = sys.energy + kT * (randexp(rng, T) + T(N + 1) * dlnV) - P * ΔV
    sys.box = box_new
    cells_valid(sys.cells, sys.box, sys.rc) || rebuild_cells!(sys)
    # Expanding a box of hard particles cannot create overlaps (the energy stays zero).
    U_new =
        (is_hard(sys.potential) && dlnV > 0) ? sys.energy : total_energy(sys; emax=U_max)
    accepted = U_new <= U_max
    if accepted
        sys.energy = U_new
    else
        # Positions are fractional, so restoring the box restores the configuration.
        sys.box = box_old
        cells_valid(sys.cells, sys.box, sys.rc) || rebuild_cells!(sys)
    end
    return record!(m, accepted)
end
