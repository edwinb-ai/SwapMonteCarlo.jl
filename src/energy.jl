"""
    pair_sum(sys, s, σ, skip1, skip2, jmin, emax) -> e

Energy of a particle of diameter `σ` at the fractional position `s ∈ [0, 1)^D` with the
particles `j > jmin` in the system, excluding `skip1` and `skip2` (use `0` for none).

This is the kernel of every energy evaluation. When the potential is repulsive, it returns
as soon as the partial sum exceeds `emax`, so the result is then only known to be larger
than `emax`.
"""
@inline function pair_sum(
    sys::System{D,T}, s::SVector{D,T}, σ::T, skip1::Int, skip2::Int, jmin::Int, emax::T
) where {D,T}
    cells = sys.cells
    box = sys.box
    pot = sys.potential
    frac = sys.frac
    diameters = sys.diameters
    members = cells.members
    counts = cells.counts
    wrap = cells.wrap
    rc2 = sys.rc * sys.rc
    c = cell_coordinates(cells, s)
    e = zero(T)
    @inbounds for I in CartesianIndices(cells.stencil)
        cell = neighbor_cell(wrap, c, I)
        for k in 1:Int(counts[cell])
            j = Int(members[k, cell])
            ((j == skip1) | (j == skip2) | (j <= jmin)) && continue
            dx = min_image(box, frac[j] - s)
            r2 = dot(dx, dx)
            # Cheap global prefilter before loading the diameter of `j`.
            r2 < rc2 || continue
            e += convert(T, pair_energy(pot, r2, σ, diameters[j]))
            if is_repulsive(pot) && e > emax
                return e
            end
        end
    end
    return e
end

"""
    virial_sum(sys, i, jmin) -> w

Sum of the pair virials of particle `i` with the particles `j > jmin`, `j ≠ i`.
"""
@inline function virial_sum(sys::System{D,T}, i::Int, jmin::Int) where {D,T}
    cells = sys.cells
    box = sys.box
    pot = sys.potential
    frac = sys.frac
    diameters = sys.diameters
    members = cells.members
    counts = cells.counts
    wrap = cells.wrap
    rc2 = sys.rc * sys.rc
    s = frac[i]
    σ = diameters[i]
    c = cell_coordinates(cells, s)
    w = zero(T)
    @inbounds for I in CartesianIndices(cells.stencil)
        cell = neighbor_cell(wrap, c, I)
        for k in 1:Int(counts[cell])
            j = Int(members[k, cell])
            ((j == i) | (j <= jmin)) && continue
            dx = min_image(box, frac[j] - s)
            r2 = dot(dx, dx)
            r2 < rc2 || continue
            r2 < convert(T, cutoff(pot, σ, diameters[j]))^2 || continue
            w += convert(T, pair_virial(pot, r2, σ, diameters[j]))
        end
    end
    return w
end

"""
    particle_energy(sys, i)

Energy of particle `i` with all the other particles.
"""
function particle_energy(sys::System{D,T}, i::Integer) where {D,T}
    return pair_sum(sys, sys.frac[i], sys.diameters[i], Int(i), 0, 0, T(Inf))
end

"""
    particle_energy(sys, s, σ, skip1, skip2, emax)

Energy of a (virtual) particle of diameter `σ` at the fractional position `s ∈ [0, 1)^D`
with all the particles except `skip1` and `skip2` (use `0` for none).

For repulsive potentials (see [`is_repulsive`](@ref)) the sum stops as soon as it exceeds
`emax` and returns that partial value, which is larger than `emax`. With `emax = Inf` the
result is always exact.
"""
function particle_energy(
    sys::System{D,T}, s::SVector{D,T}, σ::Real, skip1::Integer, skip2::Integer, emax::Real
) where {D,T}
    return pair_sum(sys, s, T(σ), Int(skip1), Int(skip2), 0, T(emax))
end

"""
    total_energy(sys; emax=Inf)

Total potential energy recomputed from scratch, including the long range correction
[`energy_lrc`](@ref). Does not modify `sys` (see [`sync_energy!`](@ref)).

For repulsive potentials the sum stops as soon as it exceeds `emax`, and the returned
value is then only known to be larger than `emax`.
"""
function total_energy(sys::System{D,T}; emax::Real=T(Inf)) where {D,T}
    lrc = convert(T, energy_lrc(sys.potential, sys))
    limit = T(emax) - lrc
    e = zero(T)
    frac = sys.frac
    diameters = sys.diameters
    for i in eachindex(frac)
        # Half sum: particle `i` interacts with the particles `j > i` only.
        e += pair_sum(sys, frac[i], diameters[i], 0, 0, i, limit - e)
        if is_repulsive(sys.potential) && e > limit
            return e + lrc
        end
    end
    return e + lrc
end

"""
    total_virial(sys)

Total virial `W = Σ_{i<j} w(r_ij)` with the pair virial `w = -r du/dr`, see
[`pair_virial`](@ref).
"""
function total_virial(sys::System{D,T}) where {D,T}
    w = zero(T)
    for i in eachindex(sys.frac)
        w += virial_sum(sys, i, i)
    end
    return w
end

"""
    pressure(sys, kT)

Virial pressure at temperature `kT`, `N kT / V + W / (D V)` plus the long range correction
[`pressure_lrc`](@ref). Not defined for hard potentials (use `VirtualPressure`).
"""
function pressure(sys::System{D,T}, kT::Real) where {D,T}
    V = volume(sys)
    ideal = length(sys) * T(kT) / V
    excess = total_virial(sys) / (D * V)
    return ideal + excess + convert(T, pressure_lrc(sys.potential, sys))
end

"""
    check_energy(sys) -> (running, recomputed)

The running total energy `sys.energy` and the energy recomputed from scratch, to detect
the accumulation of rounding errors (or bugs) in energy differences.
"""
check_energy(sys::System) = (sys.energy, total_energy(sys))

"""
    sync_energy!(sys)

Set the running total energy to the energy recomputed from scratch and return it.
"""
function sync_energy!(sys::System)
    sys.energy = total_energy(sys)
    return sys.energy
end

"""
    rebuild_cells!(sys)

Rebuild the cell list of `sys` for its current box, choosing the number of cells again
(after volume moves). Allocates.
"""
function rebuild_cells!(sys::System)
    rebuild_cells!(sys.cells, sys.frac, sys.box, sys.rc)
    return sys
end
