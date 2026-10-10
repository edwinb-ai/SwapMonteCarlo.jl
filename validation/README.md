# Validation

Physics checks of SwapMonteCarlo.jl against exact results and literature data. They are
longer than the statistical tests of the test suite (`SWAPMC_TEST_GROUP=stat`) and are
run by hand, not by `Pkg.test()`.

## How to run

The scripts use their own environment, which develops the package from the parent
directory (`[sources] SwapMonteCarlo = {path = ".."}`). From the repository root:

    julia --project=validation -e 'using Pkg; Pkg.instantiate()'   # once
    julia --project=validation validation/run_all.jl               # everything
    julia --project=validation validation/run_all.jl hs3d_eos      # one or more scripts
    julia --project=validation validation/lj_nist.jl               # or directly

Each script prints a markdown report and writes it to `results/<name>.md`, which is
meant to be committed (the results directory ignores only `*.dat` files). Every report
starts with the date, the Julia and package versions, the CPU, the base seed and the wall
time, followed by an `Overall:` verdict and one table per check. The base seed can be
changed with the environment variable `SWAPMC_VALIDATION_SEED`.

`run_all.jl` runs the scripts one after the other, each in a fresh Julia process, and
prints a summary with the verdicts. A script that throws is reported as `ERROR`; failed
checks are reported through the verdict (the script still exits normally).

## Scripts

| script | checks | runtime |
|---|---|---|
| `check_references.jl` | the reference equations of state of `references.jl` against values computed independently | < 1 s |
| `ideal_gas_npt.jl` | NPT ideal gas, N ∈ {1, 2, 4, 16, 64}: full volume distribution vs Gamma(N + 1, kT/P) | ~40 s |
| `hd2d_eos.jl` | NPT hard disks, η = 0.2–0.6, vs Henderson; virtual pressure in NVT | ~1.5 min |
| `hs3d_eos.jl` | NPT hard spheres (N = 500), fluid ρ = 0.5–0.9 vs Kolafa (and CS), FCC solid ρ = 1.05–1.15 vs Speedy; virtual pressure in NVT | ~10–16 min |
| `lj_nist.jl` | NVT Lennard-Jones (r_c = 3σ, no shift, tail corrections), U/N and P vs NIST SRSW | ~6 min |
| `swap_vs_noswap.jl` | 2D/3D Ninarello soft spheres: same energy with and without swaps, relaxation-time speedup, swap equilibration at low temperature | ~12–14 min |

Runtimes are for one core of an AMD Ryzen AI 7 350 (the reports give the measured
values; the upper ends were measured while other heavy jobs shared the machine); the
full set takes 30–40 minutes. To make a run shorter, reduce the `nsweeps` values at the
top of `main()` in each script: the standard errors grow as the inverse square root.

### What each check means

- **Ideal gas** (`ideal_gas_npt.jl`). With non-interacting particles the NPT volume
  distribution is exactly `p(V) ∝ V^N e^{-βPV}`. Since `VolumeMove` samples `ln V`, its
  acceptance contains the Jacobian term `(N + 1) Δln V`; a wrong exponent shifts ⟨V⟩ by
  kT/P, which the test detects at every N. Checked: ⟨V⟩, Var V and ⟨ln V⟩ (within 4
  blocked standard errors) and the full distribution (Kolmogorov–Smirnov, p > 1e-3) on
  samples thinned by six autocorrelation times.
- **Hard disks** (`hd2d_eos.jl`). Henderson's equation of state is only approximate: the
  deviations at η ≥ 0.5 are expected to be dominated by the EOS, not by the simulation
  (they have the same sign and size in the NPT density and in the NVT virtual pressure).
- **Hard spheres** (`hs3d_eos.jl`). The Kolafa–Labík–Malijevský fluid EOS is accurate to
  about 1e-4 in Z, so the fluid densities test the NPT sampler (volume move, hard-core
  overlap detection, early exit of energy sums) at the 0.1 % level. Carnahan–Starling is
  shown for comparison (it underestimates Z by up to 0.3 %). The FCC crystal is compared
  with Speedy's fit, which is accurate to a few 0.1 %. The virtual pressure
  (`VirtualPressure`) has a relative bias of order ξ; two values of ξ are reported.
- **Lennard-Jones** (`lj_nist.jl`). Same model and conventions as the NIST SRSW table
  "MC, r_c = 3σ, standard long-range corrections" (`LJ_NIST_MC_RC3_LRC` in
  `references.jl`). The pressure convention is discussed in the header of the script and
  in its report: NIST's pressure is the virial inside the cutoff plus the standard tail
  correction, which already contains the impulsive term of the truncation for g(r_c) = 1;
  adding a separate impulsive correction would be inconsistent with the NIST data.
- **Swap vs no swap** (`swap_vs_noswap.jl`). Swaps must not change equilibrium averages
  (checked at a temperature where the displacement-only dynamics also equilibrates) and
  should accelerate the structural relaxation (self-overlap Q(t), a = 0.3,
  cage-relative in 2D). At a low temperature, two independent swap runs must agree and
  relax within the run, while the displacement-only run does not relax.

### Interpreting the results

Deviations are reported as relative differences with their standard error, or as `z`,
the difference in units of the combined standard error. Standard errors come from block
averaging (the largest estimate over the blocking levels with at least 32 blocks). The
pass criteria allow 4 standard errors plus, where stated, a systematic allowance for
finite-size effects (N = 400–500 in NPT, effects of order 1/N) and for the accuracy of
the reference. A single failing row out of many at the 3–4 σ level can be a statistical
fluctuation: rerun with another `SWAPMC_VALIDATION_SEED` before suspecting a bug.

## Files

- `references.jl`: reference equations of state (Kolafa, Carnahan–Starling, Speedy,
  Henderson), `density_from_pressure`, NIST Lennard-Jones tables. Base only.
- `common.jl`: statistics (blocking, integrated autocorrelation time, KS test), helpers
  for hard particles and the markdown report writer.
- `swap_dynamics.jl`: unwrapped trajectories, self-overlap and relaxation times.
- `results/`: the latest reports.
