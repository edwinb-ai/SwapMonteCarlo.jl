# Baseline: original `swapmc.jl`

Date: 2026-10-10. Command (from the repository root):

    julia --project=benchmark benchmark/baseline/run_baseline.jl

Machine and toolchain (`versioninfo()`):

- Julia 1.13.1 (commit 96ca370cf0e, official release), LLVM 20.1.8 (znver5)
- CPU: 16 × AMD Ryzen AI 7 350 w/ Radeon 860M, Linux x86_64, 1 thread
- Total runtime of the script: about 37 s.

## Setup

The code under test is `swapmc_original.jl`, a verbatim copy of
`/home/eabm/Documents/phd-projects/AthermalQuasistaticShearing/swapmc.jl` wrapped in
`module Orig` without the top-level `main()` call.

- 2D Ninarello soft spheres: SS12 with C² smoothing at `r_cut = 1.25 σij`,
  non-additivity 0.2, diameters `P(σ) ∝ σ⁻³` on `[0.73, 1.62]`, density `ρ = 1`
  (`L = √N`), `kT = 0.1` (the script reads kT from the command line), `p_swap = 0.2`.
- State built exactly as in `Orig.run_simulation`: `Random.seed!(20261010)`, square
  lattice from `Orig.initialize_lattice`, diameters from
  `Orig.inverse_diameters(; sigma_max=1.62)` drawn in particle order, cell list with
  cutoff `r_cut · σmax · 1.1`.
- Warm-up: 100 sweeps (N attempts each) at `p_swap = 0.2` with the script's own
  `adjust_delta(…; target=0.3)` after every sweep, starting from `δ = 0.2`. δ ends at
  about 0.085. Timings use this frozen δ.
- Timing: minimum over 3 repetitions (2 for "as-is") of `@elapsed` of a loop of
  attempts, divided by the number of attempts; GC time included. Attempts per
  measurement: 10⁶ for `mc_step!`; 5000 (N = 10³) and 300 (N = 10⁴) for "as-is".

## Results (ns per attempt)

| N      | displacement (`p_swap = 0`) | swap (`p_swap = 1`) | mixed (`p_swap = 0.2`) | `total_energy` call | as-is (`mc_step!` + `total_energy`) |
|-------:|----------------------------:|--------------------:|-----------------------:|--------------------:|------------------------------------:|
| 1 000  | 923                         | 1 889               | 1 117                  | 2.997 × 10⁵         | 3.07 × 10⁵                          |
| 10 000 | 939                         | 1 896               | 1 142                  | 3.245 × 10⁶         | 3.35 × 10⁶                          |

Allocations per attempt: displacement ≈ 1.7 kB, swap ≈ 3.4 kB, mixed ≈ 2.1 kB (the
neighbor list `Int[]` built by `get_neighbor_indices` on every energy evaluation);
`total_energy` allocates ≈ 0.86 MB (N = 10³) and ≈ 8.6 MB (N = 10⁴) per call.

Acceptance at kT = 0.1 after warm-up: displacement ≈ 0.30–0.34, swap ≈ 0.13.

Derived: as-is, one sweep costs ≈ 0.31 s at N = 10³ and ≈ 33 s at N = 10⁴, i.e. the
original main loop is dominated (> 99.6 %) by the per-attempt `total_energy` call.
Without it, a sweep costs ≈ 1.1 ms (N = 10³) and ≈ 11 ms (N = 10⁴); the per-attempt cost
of `mc_step!` is independent of N, as expected with cell lists.

## Reference energies for port checks

`U/N` from `Orig.total_energy(particles, cell_list, 1.25) / N` (no tail correction; the
potential is zero beyond the cutoff).

1. Seeded lattice (the state above before warm-up; depends on the global RNG stream of
   `Random.seed!(20261010)`, which is Xoshiro256++ in Julia 1.13 but not guaranteed to be
   stable across Julia versions):

   | N      | U/N             | sum(σ)               | σ[1]              | σ[end]            |
   |-------:|-----------------|----------------------|-------------------|-------------------|
   | 1 000  | 6.609499658157  | 1001.775173833047    | 0.872506651283850 | 0.834890597890452 |
   | 10 000 | 6.227848901409  | 10110.824168564683   | 0.872506651283850 | 0.929202070368531 |

2. RNG-free lattice (preferred for port checks): positions from
   `Orig.initialize_lattice(N, √N)` (`side = ceil(√N)`, sites `((i-½)a, (j-½)a)` with
   `a = L/side`, `j` running fastest), diameters
   `σ_k = 1/√((1-u)/0.73² + u/1.62²)` with `u = (k-½)/N` assigned to sites in order
   (no shuffle):

   | N      | U/N             |
   |-------:|-----------------|
   | 1 000  | 30.092752732640 |
   | 10 000 | 30.390921820627 |

   This configuration has overlapping large particles (U/N ≈ 30), which makes it a
   sensitive test of the pair potential, the mixing rule and the cutoff.
