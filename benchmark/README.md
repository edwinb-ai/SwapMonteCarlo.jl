# Benchmarks

The benchmark environment is `benchmark/Project.toml`. It contains BenchmarkTools,
StableRNGs, StaticArrays and the standard libraries used by the original script, and
it loads `SwapMonteCarlo` from the parent directory through `[sources]`. Set it up
once from the repository root:

    julia --project=benchmark -e 'import Pkg; Pkg.instantiate()'

## Baseline: the original script

`baseline/swapmc_original.jl` is a verbatim copy of the original 2D swap Monte Carlo
script (`AthermalQuasistaticShearing/swapmc.jl`), wrapped in `module Orig` and without
its top-level `main()` call. `baseline/run_baseline.jl` builds the 2D Ninarello system
(ρ = 1, kT = 0.1, `r_cut = 1.25`, `p_swap = 0.2`) at N = 10³ and N = 10⁴ with the
script's own initialization functions and a seeded global RNG. It then times, in ns
per attempt:

- `mc_step!` alone, for displacements only (`p_swap = 0`), swaps only (`p_swap = 1`) and
  the script's mix (`p_swap = 0.2`);
- the "as-is" cost, `mc_step!` followed by the O(N) `total_energy` call that the
  original main loop makes after every attempt.

It also prints `U/N` of two initial configurations (a seeded one and an RNG-free one)
for checking the port against the original. Run it from the repository root (about
40 s):

    julia --project=benchmark benchmark/baseline/run_baseline.jl

The recorded numbers and the exact procedure are in `baseline/RESULTS.md`. Update
that file when you rerun on another machine or Julia version.

## Port check

`port_check.jl` builds the RNG-free lattice configuration of `baseline/RESULTS.md` both
with the original script and with SwapMonteCarlo.jl (`PolySoftSphere(; xc=1.25,
mixing=NonAdditive(0.2))`, same box, positions and diameters) and checks that the
energies per particle agree to a relative `1e-12`, with each other and with the recorded
reference values (30.092752732640 at N = 10³, 30.390921820627 at N = 10⁴). They agree to
about `1e-15`. Run it in a few seconds with

    julia --project=benchmark benchmark/port_check.jl

## Package suite

`benchmarks.jl` defines a BenchmarkTools `SUITE`; `run.jl` tunes and runs it and writes
`RESULTS.md` (median times normalized per attempt, per particle or per pair, the
allocations, and the speed-up over the original script). It takes about four minutes:

    julia --project=benchmark benchmark/run.jl [output.md]

Set `SWAPMC_BENCH_SECONDS` to change the time budget per benchmark (default 3 s). The
systems are built once from a fixed seed, equilibrated for 50 sweeps with step size
tuning, and then timed with frozen step sizes:

- `PolySoftSphere 2D`: the system of the baseline (Ninarello soft spheres,
  `NonAdditive(0.2)`, `xc = 1.25`, diameters `P(σ) ∝ σ⁻³` on `[0.73, 1.62]`, ρ = 1,
  kT = 0.1) at N = 10³ and 10⁴: one displacement attempt, one swap attempt, one sweep of
  the 0.8/0.2 mix (reported per attempt), `total_energy` and `rebuild_cells!` (per
  particle).
- `PolySoftSphere 3D`: the same model in 3D at ρ = 1 (packing fraction ≈ 0.62, a dense
  liquid) and kT = 0.1, N = 10³ and 8000.
- `N=1000 NPT / volume`: one `VolumeMove` attempt (a full energy recomputation) in the
  NPT ensemble at the virial pressure of the NVT system, with a tuned step.
- `HardSphere 3D`: additive polydisperse hard spheres at φ = 0.35, displacements.
- `LennardJones 3D`: monodisperse Lennard-Jones, `xc = 2.5`, ρ = 0.8, kT = 1,
  displacements.
- `kernels`: `pair_energy` throughput on random pairs (about half within the cutoff).

## Optimization log

Median ns per attempt (`SWAPMC_BENCH_SECONDS=2`), measured after each step on the
machine of `baseline/RESULTS.md`. Every step keeps the Markov chain unchanged: with fixed
seeds, the configurations of 2D and 3D trajectories (soft spheres NVT and NPT, Lennard-
Jones, pseudo hard spheres, hard spheres NPT) are bitwise identical before and after
steps 1 and 2. Step 3 only changes the order of the energy sums in systems that switch
to range 2 (energies agree to about `1e-15` relative and the configurations remained
identical). The random number draws are unchanged.

| benchmark                    | before | 1. no prefilter | 2. combined pass | 3. range 2 |
|:-----------------------------|-------:|----------------:|-----------------:|-----------:|
| 2D N=10³ displacement        |    597 |             369 |              227 |        233 |
| 2D N=10³ swap                |    819 |             571 |              460 |        460 |
| 2D N=10³ sweep (per attempt) |    679 |             438 |              292 |        296 |
| 2D N=10⁴ displacement        |    606 |             388 |              245 |        252 |
| 2D N=10⁴ swap                |    906 |             625 |              471 |        478 |
| 3D N=10³ displacement        |   2661 |            2137 |             1620 |       1188 |
| 3D N=10³ swap                |   3859 |            3594 |             3206 |       2320 |
| 3D N=10³ sweep (per attempt) |   2922 |            2440 |             1967 |       1515 |
| 3D N=8000 displacement       |   2473 |            1822 |             1358 |       1190 |
| 3D N=8000 swap               |   3579 |            3034 |             2651 |       2280 |
| HardSphere 3D displacement   |    363 |             246 |              249 |        245 |
| LennardJones 3D displacement |   3165 |            3379 |             2259 |       2186 |
| 2D N=10³ volume (µs)         |   55.2 |            66.2 |             66.4 |       66.1 |
| 3D N=10³ volume (µs)         |   1010 |             834 |              825 |        708 |
| 3D N=8000 `total_energy`/N   |    943 |             732 |              734 |        852 |

What was measured first: a displacement costs two energy evaluations of particle `i`
(the profile shows 98% of the time in `particle_energy`; the random numbers, `wrap`,
`cell_move!` and the bookkeeping are negligible). Timing `particle_energy` repeatedly
for the *same* particle is misleading (about 110 ns in 2D): the branch predictor then
learns the neighbor pattern. For random particles it costs about 200 ns in 2D, and the
inner loop (about 40 instructions per candidate, no bounds checks, `round` compiled to
one `vrndscale`, all powers multiplications) is limited by branch mispredictions and by
the number of candidates.

1. **No global cutoff prefilter** (`src/energy.jl`, `pair_sum`). The loop tested
   `r2 < rc²` with the largest cutoff before loading the diameter of `j`. About a third
   of the candidates pass it in 2D, in random order, so the branch is mispredicted very
   often; `pair_energy` tests the pair cutoff anyway. Removing it saves 30–40% in 2D and
   about 20% in 3D. Monodisperse Lennard-Jones, where both tests are the same, got 7%
   slower (fixed by step 2).
2. **Two energies in one pass** (`two_pair_sums` in `src/energy.jl`, used by
   `src/moves/swap.jl` and `src/moves/displacement.jl`). A swap needs the old and new
   energies of `i` and of `j`, each pair at the same position: one pass per particle
   computes both (two passes instead of four). A displacement that stays in its cell
   (most of them: the step is much smaller than a cell) has the same neighbor candidates at the old and new positions:
   one pass instead of two. The sums are bitwise equal to separate ones. The early exit
   of the new energy is lost in these cases, but it saved less than the shared pass
   costs. Hard potentials keep the old path (zero old energy, stop at the first
   overlap).
3. **Narrow cells when they are crowded** (`cell_range` in `src/celllist.jl`). Cells
   at least `rc/2` wide and a `5^D` stencil search `15.6 rc³` instead of `27 rc³` in 3D,
   but visit 125 instead of 27 cells. A first version used range 2 in every 3D system;
   it halved the speed of the hard sphere benchmark (mostly empty cells). Measured over
   systems with 1 to 16 particles per range-1 cell (soft spheres at ρ = 0.5–1, Lennard-
   Jones, pseudo hard spheres, hard spheres, 2D and 3D), range 2 wins from about 11
   particles per range-1 cell on (10–30% in 3D, up to 10% in 2D) and loses below 8
   (15% to 3 times slower), so it is chosen when there are at least 10. The cap on the
   number of cells now scales the counts isotropically (it used to give, e.g.,
   `(3, 18, 18)` cells for a cubic box). `total_energy` (the half sum over `j > i`) is
   16% slower with range 2 at N = 8000, but volume moves are rare.

Tried and dropped (prototyped outside the package, on `particle_energy` of random particles):

- A branch-free soft sphere kernel (compute every candidate, select with `ifelse`): 30%
  slower in 2D and 2× slower in 3D, since every candidate then pays two divisions.
  Computing `x²` and `1/x²` from one division made no difference.
- Packed storage of `(s, σ)` in the cell slots (no indirect loads of positions and
  diameters): within noise in 2D and 3D, both at N = 10³ (L2 resident) and 10⁴.
- Range 3 (cells at least `rc/3` wide): slower than range 1 or 2 in every system tried.

Ideas not tried: Verlet neighbor lists with a skin would cut the candidates of 3D soft
spheres by about 8 (pair cutoffs `1.25 σij` vs a `15.6 rc³` stencil with `rc = 1.25
σmax`), at the price of rebuild bookkeeping for displacements, swaps (cutoffs change
with σ) and volume moves.
