# Performance baseline of the original 2D swap Monte Carlo script (`swapmc_original.jl`).
#
# Run from the repository root:
#
#     julia --project=benchmark benchmark/baseline/run_baseline.jl
#
# Model: 2D polydisperse Ninarello soft spheres (SS12, C² smoothed at r_cut = 1.25 σij,
# non-additivity 0.2), diameters from P(σ) ∝ σ⁻³ on [0.73, 1.62], number density ρ = 1,
# all exactly as in the script's `run_simulation`/`main`. The script reads kT from the
# command line; here kT = 0.1 is used.
#
# Measured, per Monte Carlo attempt:
#   (a) `Orig.mc_step!` alone, for p_swap = 0 (displacements only), p_swap = 1 (swaps
#       only) and p_swap = 0.2 (the script's mix);
#   (b) "as-is": `mc_step!` followed by the O(N) `total_energy` call that the original
#       main loop performs after every attempt (p_swap = 0.2).
# Timings are the minimum over a few repetitions of `@elapsed` around a loop of attempts
# (GC time included, since the original allocates in its inner loops).

using Random: Random
using Printf: @printf
using InteractiveUtils: versioninfo

include(joinpath(@__DIR__, "swapmc_original.jl"))

const KT = 0.1
const DENSITY = 1.0
const R_CUT = 1.25
const P_SWAP = 0.2
const DELTA0 = 0.2
const SEED = 20261010

"""
    build_state(N; seed=SEED)

Build the initial state exactly as `Orig.run_simulation` does: seed the global RNG,
place `N` particles on a square lattice (`Orig.initialize_lattice`), draw diameters
with `Orig.inverse_diameters(; sigma_max=1.62)` in particle order, and build the cell
list with cutoff `r_cut * σmax * 1.1`. Returns a named tuple with the state.
"""
function build_state(N::Int; seed::Integer=SEED)
    Random.seed!(seed)
    L = sqrt(N / DENSITY)
    particles = Orig.Particle[]
    (x, y) = Orig.initialize_lattice(N, L)
    for i in 1:N
        pos = Orig.SVector(x[i], y[i])
        sigma = Orig.inverse_diameters(; sigma_max=1.62)
        push!(particles, Orig.Particle(pos, sigma))
    end
    sigma_max_current = maximum(p.sigma for p in particles)
    cell_cutoff = R_CUT * sigma_max_current * 1.1
    cell_list, particle_cells = Orig.build_cell_list(particles, L, cell_cutoff)
    return (; N, L, particles, cell_list, particle_cells)
end

"""
    deterministic_energy(N)

Energy per particle of an RNG-free reference configuration for later port checks: the
`Orig.initialize_lattice` square lattice at ρ = 1 with diameters
`σ_k = F⁻¹((k - 1/2)/N)` of the σ⁻³ distribution on [0.73, 1.62], assigned to lattice
sites in order (no shuffling).
"""
function deterministic_energy(N::Int)
    L = sqrt(N / DENSITY)
    (x, y) = Orig.initialize_lattice(N, L)
    smin, smax = 0.73, 1.62
    particles = Orig.Particle[]
    for k in 1:N
        u = (k - 0.5) / N
        σ = 1.0 / sqrt((1.0 - u) / smin^2 + u / smax^2)
        push!(particles, Orig.Particle(Orig.SVector(x[k], y[k]), σ))
    end
    σmax = maximum(p.sigma for p in particles)
    cl, _ = Orig.build_cell_list(particles, L, R_CUT * σmax * 1.1)
    return Orig.total_energy(particles, cl, R_CUT) / N
end

"""
    attempts!(st, n, beta, delta, p_swap)

Perform `n` calls of `Orig.mc_step!` on state `st`. Returns the number of accepted
displacements, displacement attempts, accepted swaps and swap attempts.
"""
function attempts!(st, n::Int, beta::Float64, delta::Float64, p_swap::Float64)
    da = dn = sa = sn = 0
    for _ in 1:n
        move, acc = Orig.mc_step!(
            st.particles, st.cell_list, st.particle_cells, beta, delta, p_swap, R_CUT
        )
        if move === :disp
            dn += 1
            da += acc
        else
            sn += 1
            sa += acc
        end
    end
    return (da, dn, sa, sn)
end

"""
    attempts_asis!(st, n, beta, delta, p_swap)

Perform `n` attempts as the original main loop does: `mc_step!` followed by
`total_energy(...) / N`. Returns the last energy per particle.
"""
function attempts_asis!(st, n::Int, beta::Float64, delta::Float64, p_swap::Float64)
    e = 0.0
    for _ in 1:n
        Orig.mc_step!(
            st.particles, st.cell_list, st.particle_cells, beta, delta, p_swap, R_CUT
        )
        e = Orig.total_energy(st.particles, st.cell_list, R_CUT) / st.N
    end
    return e
end

"""
    warmup!(st, nintervals, beta, delta)

Melt the lattice with the script's own scheme: intervals of `N` attempts at
p_swap = 0.2, after each of which δ is adjusted with `Orig.adjust_delta` toward 30 %
acceptance. Returns the final δ.
"""
function warmup!(st, nintervals::Int, beta::Float64, delta::Float64)
    for _ in 1:nintervals
        da, dn, _, _ = attempts!(st, st.N, beta, delta, P_SWAP)
        delta = Orig.adjust_delta(delta, da, dn; target=0.3)
    end
    return delta
end

"""
    time_per_attempt(f, n; reps=3)

Minimum over `reps` repetitions of the wall time of `f(n)` divided by `n`, in ns, and
the bytes allocated per attempt in the last repetition.
"""
function time_per_attempt(f, n::Int; reps::Int=3)
    best = Inf
    bytes = 0
    for _ in 1:reps
        GC.gc()
        bytes = @allocated(t = @elapsed f(n))
        best = min(best, t)
    end
    return 1e9 * best / n, bytes / n
end

function main()
    beta = 1 / KT
    println("# Baseline of the original swapmc.jl")
    versioninfo()
    println()
    println("kT = $KT, ρ = $DENSITY, r_cut = $R_CUT, p_swap = $P_SWAP, seed = $SEED")
    println()

    # Warm up compilation on a tiny system.
    st = build_state(100)
    attempts!(st, 1000, beta, DELTA0, P_SWAP)
    attempts_asis!(st, 10, beta, DELTA0, P_SWAP)

    for (N, nfast, nasis) in ((1000, 1_000_000, 5_000), (10_000, 1_000_000, 300))
        println("## N = $N")
        st = build_state(N)
        e0 = Orig.total_energy(st.particles, st.cell_list, R_CUT) / N
        σs = [p.sigma for p in st.particles]
        @printf("initial (seeded lattice) U/N = %.12f\n", e0)
        @printf("  sum(σ) = %.12f, σ[1] = %.15f, σ[end] = %.15f\n", sum(σs), σs[1], σs[end])
        @printf("deterministic lattice U/N   = %.12f\n", deterministic_energy(N))
        @printf(
            "cells per side = %d, cell size = %.4f, L = %.4f\n",
            st.cell_list.n_cells,
            st.cell_list.cell_size,
            st.L
        )

        t = @elapsed delta = warmup!(st, 100, beta, DELTA0)
        @printf(
            "warm-up: 100 sweeps in %.1f s, δ = %.4f, U/N = %.6f\n",
            t,
            delta,
            Orig.total_energy(st.particles, st.cell_list, R_CUT) / N
        )

        for (label, ps) in (
            ("displacement (p_swap=0)", 0.0),
            ("swap (p_swap=1)", 1.0),
            ("mixed (p_swap=0.2)", P_SWAP),
        )
            acc = Ref((0, 0, 0, 0))
            ns, b = time_per_attempt(
                n -> (acc[] = attempts!(st, n, beta, delta, ps)), nfast
            )
            da, dn, sa, sn = acc[]
            @printf(
                "mc_step! %-24s %9.1f ns/attempt %8.1f B/attempt  acc disp %.3f swap %.3f\n",
                label,
                ns,
                b,
                dn > 0 ? da / dn : NaN,
                sn > 0 ? sa / sn : NaN
            )
        end

        ns, b = time_per_attempt(
            n -> for _ in 1:n
                Orig.total_energy(st.particles, st.cell_list, R_CUT)
            end,
            max(10, nasis ÷ 2),
        )
        @printf("total_energy alone              %12.1f ns/call    %10.1f B/call\n", ns, b)

        ns, b = time_per_attempt(
            n -> attempts_asis!(st, n, beta, delta, P_SWAP), nasis; reps=2
        )
        @printf(
            "as-is (mc_step! + total_energy) %12.1f ns/attempt %10.1f B/attempt\n", ns, b
        )
        @printf("  -> as-is sweep (N attempts) ≈ %.3f s\n", ns * N * 1e-9)
        println()
    end
    return nothing
end

main()
