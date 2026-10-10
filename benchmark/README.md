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

## Package suite

TODO: add `benchmarks.jl` with a BenchmarkTools `SUITE` for SwapMonteCarlo.jl once the
move API (`Displacement`, `Swap`, `attempt!`, `Simulation`, `run!`) exists. It should
measure the same quantities as the baseline, on the same system (N = 10³ and 10⁴), so
that the two can be compared directly: ns per displacement attempt, ns per swap
attempt, ns per sweep with the 0.8/0.2 mix, `total_energy`, plus NPT volume moves and
3D systems. It will be run with:

    julia --project=benchmark benchmark/benchmarks.jl
