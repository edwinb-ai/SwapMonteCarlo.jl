# BenchmarkTools suite of SwapMonteCarlo.jl.
#
#     julia --project=benchmark benchmark/run.jl    # runs the suite, writes RESULTS.md
#
# or interactively:
#
#     include("benchmark/benchmarks.jl"); tune!(SUITE); results = run(SUITE)
#
# All systems are built once, from a fixed seed, and equilibrated briefly (with step size
# tuning) before they are timed; the step sizes are then frozen. Moves are timed one
# attempt per evaluation, on the equilibrated system, so accepted attempts keep the system
# in (near) equilibrium. Per-particle and per-pair costs are normalized in `run.jl`
# with [`normalization`](@ref).

using BenchmarkTools: BenchmarkGroup, @benchmarkable
using Random: Xoshiro
using SwapMonteCarlo
using SwapMonteCarlo: attempt!, pair_energy, rebuild_cells!

"""Seed of all random number generators of the suite."""
const SEED = 20261010

"""Temperature of the soft sphere systems (as in `baseline/RESULTS.md`)."""
const KT = 0.1

"""Sweeps of equilibration (with step size tuning) before timing."""
const EQUILIBRATION_SWEEPS = 50

"""
    soft_sphere_simulation(D, N; ρ=1.0, ensemble=NVT(KT), extra_moves=())

Equilibrated simulation of `N` Ninarello soft spheres in `D` dimensions:
`PolySoftSphere(; xc=1.25, mixing=NonAdditive(0.2))`, diameters `P(σ) ∝ σ⁻³` on
`[0.73, 1.62]`, number density `ρ`, displacement and swap moves (weights 0.8 and 0.2).
In 2D at ρ = 1 and kT = 0.1 this is the system of the baseline benchmark of the original
script. In 3D the same parameters give a dense liquid (packing fraction ≈ 0.62).
"""
function soft_sphere_simulation(
    D::Int, N::Int; ρ::Real=1.0, ensemble=NVT(KT), extra_moves::Tuple=()
)
    rng = Xoshiro(SEED)
    σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N)
    box = box_for_density(Val(D), N, ρ)
    potential = PolySoftSphere(; xc=1.25, mixing=NonAdditive(0.2))
    sys = System(box, lattice_positions(box, N), σ, potential)
    moves = (Displacement(0.1) => 0.8, Swap() => 0.2, extra_moves...)
    sim = Simulation(sys, ensemble, moves...; seed=SEED)
    equilibrate!(sim, EQUILIBRATION_SWEEPS; energy_check=0)
    return sim
end

"""
    hard_sphere_simulation(N; φ=0.35)

Equilibrated simulation of `N` additive polydisperse hard spheres in 3D (diameters as in
[`soft_sphere_simulation`](@ref)) at packing fraction `φ`, with displacement moves only.
"""
function hard_sphere_simulation(N::Int; φ::Real=0.35)
    rng = Xoshiro(SEED)
    σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N)
    box = box_for_packing_fraction(Val(3), σ, φ)
    sys = System(box, insert_hard_spheres(rng, box, σ), σ, HardSphere())
    sim = Simulation(sys, NVT(1.0), Displacement(0.1) => 1; seed=SEED)
    equilibrate!(sim, EQUILIBRATION_SWEEPS; energy_check=0)
    return sim
end

"""
    lennard_jones_simulation(N; ρ=0.8, kT=1.0)

Equilibrated simulation of `N` monodisperse Lennard-Jones particles in 3D (`xc = 2.5`,
energy shifted) at density `ρ` and temperature `kT`, with displacement moves only.
"""
function lennard_jones_simulation(N::Int; ρ::Real=0.8, kT::Real=1.0)
    box = box_for_density(Val(3), N, ρ)
    sys = System(box, lattice_positions(box, N), ones(N), LennardJones())
    sim = Simulation(sys, NVT(kT), Displacement(0.1) => 1; seed=SEED)
    equilibrate!(sim, EQUILIBRATION_SWEEPS; energy_check=0)
    return sim
end

"""
    volume_simulation(D, N)

Equilibrated NPT simulation of the soft sphere system of [`soft_sphere_simulation`](@ref)
at the virial pressure of the NVT system, with an additional (tuned) `VolumeMove`.
"""
function volume_simulation(D::Int, N::Int)
    nvt = soft_sphere_simulation(D, N)
    P = pressure(nvt.system, KT)
    return soft_sphere_simulation(
        D, N; ensemble=NPT(KT, P), extra_moves=(VolumeMove(1e-3) => 1 / N,)
    )
end

"""
    kernel_sum(pot, r2, σi, σj) -> Float64

Sum of `pair_energy(pot, r2[k], σi[k], σj[k])` over all `k`: the throughput of a pair
kernel on (mostly) unpredictable inputs.
"""
function kernel_sum(pot, r2::Vector{Float64}, σi::Vector{Float64}, σj::Vector{Float64})
    e = 0.0
    @inbounds for k in eachindex(r2, σi, σj)
        e += pair_energy(pot, r2[k], σi[k], σj[k])
    end
    return e
end

"""Number of pairs per evaluation of the pair kernel benchmarks."""
const KERNEL_PAIRS = 1024

"""
    kernel_inputs() -> (r2, σi, σj)

Random pair inputs for the kernel benchmarks: diameters `P(σ) ∝ σ⁻³` on `[0.73, 1.62]`
and distances uniform in `[0.8, 1.6]` (about half of the pairs within the cutoff).
"""
function kernel_inputs()
    rng = Xoshiro(SEED)
    σi = sample_diameters(rng, InversePowerLaw(0.73, 1.62), KERNEL_PAIRS)
    σj = sample_diameters(rng, InversePowerLaw(0.73, 1.62), KERNEL_PAIRS)
    r2 = [(0.8 + 0.8 * rand(rng))^2 for _ in 1:KERNEL_PAIRS]
    return r2, σi, σj
end

"""
    normalization(key) -> (divisor, unit)

How the median time of the benchmark `key` (a vector of group names) is reported:
moves per attempt, sweeps, `total_energy` and `rebuild_cells!` per particle, pair kernels
per pair.
"""
function normalization(key::AbstractVector{<:AbstractString})
    leaf = last(key)
    if leaf in ("displacement", "swap", "volume")
        return 1, "ns/attempt"
    elseif leaf == "pair_energy"
        return KERNEL_PAIRS, "ns/pair"
    end
    # Sweeps, total energies and rebuilds are reported per particle.
    N = parse(Int, only(m.captures[1] for m in eachmatch(r"N=(\d+)", join(key, " "))))
    return N, leaf == "sweep (0.8/0.2)" ? "ns/attempt" : "ns/particle"
end

const SUITE = BenchmarkGroup()

let cond = (; kT=KT)
    # Soft sphere moves, sweeps, energies and cell rebuilds.
    for (D, N) in ((2, 1000), (2, 10_000), (3, 1000), (3, 8000))
        sim = soft_sphere_simulation(D, N)
        sys = sim.system
        δ = step_size(sim).displacement
        g = SUITE["PolySoftSphere $(D)D"]["N=$(N)"] = BenchmarkGroup()
        g["displacement"] = @benchmarkable attempt!(
            $(Displacement(δ)), $sys, $cond, $(Xoshiro(SEED + 1))
        )
        g["swap"] = @benchmarkable attempt!($(Swap()), $sys, $cond, $(Xoshiro(SEED + 2)))
        g["sweep (0.8/0.2)"] = @benchmarkable run!($sim, 1; energy_check=0)
        g["total_energy"] = @benchmarkable total_energy($sys)
        g["rebuild_cells!"] = @benchmarkable rebuild_cells!($sys)
    end

    # Volume moves (a full energy recomputation, sometimes a cell rebuild).
    for (D, N) in ((2, 1000), (3, 1000))
        sim = volume_simulation(D, N)
        cond_npt = (; kT=KT, pressure=sim.ensemble.pressure)
        move = VolumeMove(step_size(sim).volume)
        g = SUITE["PolySoftSphere $(D)D"]["N=$(N) NPT"] = BenchmarkGroup()
        g["volume"] = @benchmarkable attempt!(
            $move, $(sim.system), $cond_npt, $(Xoshiro(SEED + 3))
        )
    end

    # Other potentials.
    hs = hard_sphere_simulation(1000)
    SUITE["HardSphere 3D"]["N=1000 φ=0.35"]["displacement"] = @benchmarkable attempt!(
        $(Displacement(step_size(hs).displacement)),
        $(hs.system),
        $((; kT=1.0)),
        $(Xoshiro(SEED + 4)),
    )
    lj = lennard_jones_simulation(1000)
    SUITE["LennardJones 3D"]["N=1000 ρ=0.8"]["displacement"] = @benchmarkable attempt!(
        $(Displacement(step_size(lj).displacement)),
        $(lj.system),
        $((; kT=1.0)),
        $(Xoshiro(SEED + 5)),
    )

    # Pair kernels.
    (r2, σi, σj) = kernel_inputs()
    for (label, pot) in (
        ("PolySoftSphere", PolySoftSphere()),
        ("LennardJones", LennardJones(; mixing=NonAdditive(0.2))),
        ("HardSphere", HardSphere()),
        ("PseudoHS", PseudoHS()),
    )
        SUITE["kernels"][label]["pair_energy"] = @benchmarkable kernel_sum(
            $pot, $r2, $σi, $σj
        )
    end
end
