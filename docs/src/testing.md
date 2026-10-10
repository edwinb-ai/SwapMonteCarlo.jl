# Testing and validation

Everything runs locally; there is no continuous integration.

## Test groups

The tests are split into groups, selected with the environment variable
`SWAPMC_TEST_GROUP`:

| Group | Contents | Time |
|---|---|---|
| `unit` (default) | unit tests against brute-force references, invariants, allocations, type stability and code quality (Aqua, JET) | short |
| `stat` | statistical physics checks: acceptance probabilities, detailed balance, ideal gas, the NPT virial identity, hard spheres, swap physics | a few minutes |
| `all` | both | |

```shell
julia --project -e 'using Pkg; Pkg.test()'
SWAPMC_TEST_GROUP=stat julia --project -e 'using Pkg; Pkg.test()'
SWAPMC_TEST_GROUP=all julia --project -e 'using Pkg; Pkg.test()'
```

The test environment is `test/Project.toml`. Each test file runs on its own given the
shared helpers in `test/helpers.jl` (toy potentials, brute-force energies and virials,
statistics utilities):

```shell
julia --project=test -e 'include("test/helpers.jl"); include("test/test_box.jl")'
```

The statistical tests are seeded (`StableRNGs`, base seed from `SWAPMC_TEST_SEED`). They
compare averages with exact results using block-averaged standard errors, and they also
require an upper bound on the error, so that a broken, noisy sampler cannot pass by
having large error bars.

## Validation

`validation/` holds longer physics checks that are run by hand and compared with
literature data, such as the hard-sphere equation of state (Kolafa, Labík and Malijevský), hard
disks (Henderson), the Lennard-Jones fluid (NIST Standard Reference Simulation Website)
and the effect of swaps. `validation/references.jl` contains the reference equations of
state and data tables and depends only on Base, so it can be included from any
environment (the examples `hs_npt3d.jl` and `lj_nvt.jl` use it):

```shell
julia validation/check_references.jl
```

## Benchmarks

`benchmark/` compares the package with the original `swapmc.jl` script it replaces. The
benchmark environment is `benchmark/Project.toml`:

```shell
julia --project=benchmark -e 'import Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/baseline/run_baseline.jl
```

The procedure and the recorded timings of the original script are in
`benchmark/baseline/RESULTS.md`.

## Documentation

This manual is built locally with Documenter.jl:

```shell
julia --project=docs -e 'import Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build runs the examples and doctests in these pages and checks that every exported
and public name has a docstring in the manual. Open `docs/build/index.html`.

## Code style

The code follows [Blue style](https://github.com/JuliaDiff/BlueStyle), configured in
`.JuliaFormatter.toml`, and every function, including internal ones, has a docstring.
Format the code with JuliaFormatter from a shared environment:

```shell
julia --project=@formatter -e 'import Pkg; Pkg.add("JuliaFormatter")'  # once
julia --project=@formatter -e 'using JuliaFormatter; format(".")'
```
