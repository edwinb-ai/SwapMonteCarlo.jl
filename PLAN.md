# SwapMonteCarlo.jl — design plan and interface contract

Metropolis Monte Carlo of (polydisperse) particles in 2D and 3D, NVT and NPT ensembles,
displacement, swap and volume moves, user-defined pair potentials. Single threaded, fast,
readable. Sibling of MolecularDynamics.jl and follows its conventions.

## Ground rules

- **Julia 1.13 only** (`julia = "1.13"`), uses the `public` keyword.
- **Blue style**: `.JuliaFormatter.toml` has `style = "blue"`. Format with
  `julia --project=@formatter -e 'using JuliaFormatter; format(".")'`.
  Explicit `return`, `function` form for multi-line functions, `;` before keyword
  arguments, 92 columns, `using X: X, f`.
- **No GitHub CI.** Tests and validation run locally.
- Every function (also internal ones) gets a docstring: indented signature, short
  paragraph, `# Arguments`/`# Keyword arguments` lists when useful. Full-sentence comments.
- Generic float type `T`: no `Float64` literals in kernels — use `zero(T)`, `T(…)`,
  `oftype`. Never use `@fastmath` (hard spheres rely on `Inf`).
- Hot paths are type stable and allocation free (`@inbounds` only in kernels).
- No side effects in constructors (no files, no printing). RNG always explicit.

## Key design decisions

1. **Fractional coordinates.** `System.frac` holds `s ∈ [0,1)^D`. A volume move just
   replaces `sys.box`; nothing to save or restore. Cell membership is defined in
   fractional space so it is unchanged by isotropic box scaling (only the *number* of
   cells may need to change). Minimum image: `min_image(box, frac[j] - s)`.
2. **Dimension from types.** `D` comes from `SVector{D,T}` / `AbstractBox{D,T}`, never from
   a runtime `dimension::Int`. Use `Val(D)` in constructors like `CubicBox(Val(3), L)`.
3. **Potential interface** (`src/potentials/interface.jl`, done): `pair_energy(pot, r2, σi,
   σj)`, `cutoff(pot, σi, σj)`, optional `pair_virial`, traits `is_repulsive`, `is_hard`,
   `energy_lrc(pot, sys)`, `pressure_lrc(pot, sys)`, `max_cutoff(pot, diameters)`.
   MolecularDynamics.jl-style `evaluate(pot, r, σi, σj) -> (u, f)` also works.
   `pair_energy` must be symmetric in `(σi, σj)`; checked with `check_symmetry`.
4. **Mixing rules** (`src/mixing.jl`, done) are a type parameter of each potential:
   `Additive()`, `NonAdditive(ε)` with `0 ≤ ε < 1`; `System` rejects negative pair
   cutoffs (`ε (σmax - σmin) > 1`).
5. **Metropolis acceptance with a threshold drawn first.** Draw `ξ = randexp(rng, T)`;
   accept iff `ln A ≥ -ξ`, with `ln A = -ΔU/kT + bias`. Equivalently
   `U_new ≤ U_max := U_old + kT * (bias + ξ)` (`≤` so that `ΔU = 0` is always accepted,
   also at `kT = 0`). `U_max` is known before the new energy is
   computed, so for `is_repulsive` potentials the energy sum stops as soon as it exceeds
   `U_max` (first overlap for hard spheres). Exactly Metropolis; never evaluates `exp`.
6. **Step size tuning only in `equilibrate!`**, toward a per-move target acceptance
   (displacement 0.4, volume 0.3). Frozen in `run!` (keeps detailed balance).
7. **Swap ΔE excludes the i–j pair**, exact because pair energies are symmetric.

## File ownership and interface contract

### `src/box.jl` (done)
`AbstractBox{D,T}`, `OrthorhombicBox(lengths::SVector)`, `OrthorhombicBox(Lx, Ly, …)`,
`CubicBox(Val(D), L)`, `volume`, `scale(box, f)`, `perpendicular_widths`,
`cartesian(box, s)`, `fractional(box, x)`, `min_image(box, Δs)` (fractional in →
Cartesian out), `wrap_fractional(s)`, `fits_cutoff(box, rc)`, `dimension(box)`.

### `src/potentials/*.jl`, `src/distributions.jl` (wave 1, agent "physics")
- `PolySoftSphere(; epsilon=1.0, xc=1.25, mixing=NonAdditive(0.2))`: Ninarello SS12,
  `u = ε[(σij/r)^12 + c0 + c2 (r/σij)^2 + c4 (r/σij)^4]` for `r < xc σij`,
  `c0 = -28/xc^12, c2 = 48/xc^14, c4 = -21/xc^16` (C² at the cutoff). Precomputed
  constants, no `sqrt`, no `^` in the kernel. `is_repulsive = true`. `pair_virial`.
- `LennardJones(; epsilon=1.0, xc=2.5, shift=EnergyShift(), mixing=Additive(),
  tail_correction=false)`: cutoff relative to `σij`; shift modes are singleton types
  `NoShift`, `EnergyShift`, `ForceShift`. Tail corrections (energy and pressure, D = 2
  and 3) only for `Additive` mixing (pair moments from single-particle moments, O(N))
  and `NoShift` (the pressure correction contains the impulsive term of the jump at the
  cutoff, so it is inconsistent with a shifted potential).
- `HardSphere(; mixing=Additive())`: `Inf` if `r2 < σij²`, else `0` (touching = no
  overlap). `is_repulsive = is_hard = true`. `pair_virial` throws (use `VirtualPressure`).
- `PseudoHS(; epsilon=1.0, mixing=Additive())`: (50,49) pseudo hard sphere, as in
  MolecularDynamics.jl. `is_repulsive = true`.
- Size distributions: `abstract type SizeDistribution`, `Monodisperse(σ)`,
  `Bidisperse(σa, σb, fraction_a)` (exact counts), `InversePowerLaw(σmin, σmax;
  exponent=3)`, `UniformSizes(σmin, σmax)`; `quantile(d, u)`;
  `sample_diameters(rng, d, N; stratified=false) -> Vector{Float64}` (`stratified=true`:
  `σ_k = F⁻¹((k-½)/N)` then `shuffle!`); `polydispersity(σs)` = population std/mean.

### `src/celllist.jl`, `src/system.jl`, `src/energy.jl`, `src/initialization.jl` (wave 1, agent "core")
- `System` struct is defined in `src/system.jl` (done; do not change field names). Add
  the constructor `System(box::AbstractBox{D,T}, positions, diameters, potential)`:
  convert to `T`, wrap into fractional coordinates, check lengths, `rc = max_cutoff`,
  `fits_cutoff(box, rc)` else `ArgumentError`, `check_symmetry` (with an internal fixed
  `Xoshiro`), build cells, compute energy; throw `ArgumentError` if the energy is `Inf`.
  The running energy `sys.energy` has the type `E = energy_type(T) = float(promote_type(T,
  Float64))` (last type parameter of `System{D,T,B,P,C,E}`), `Float64` for `Float32`
  systems, so that volume moves (differences of total energies) are not dominated by
  rounding; pair and particle energies stay in `T`.
- Accessors: `Base.length(sys)`, `dimension(sys)`, `positions(sys)` (Cartesian copy),
  `diameters(sys)`, `volume(sys)`, `density(sys)`, `packing_fraction(sys)` (D = 2, 3).
- Cell list (internals free) with these operations:
  - `cell_move!(cells, i::Int, s_new)` — O(1) update after an accepted displacement.
  - `cells_valid(cells, box, rc) -> Bool` — cell widths still ≥ rc/range.
  - `rebuild_cells!(sys)` — full rebuild for the current `sys.box` (may change the
    number of cells; may allocate, it is rare).
  - Handles any number of cells per side ≥ 1 without double counting, `rc = 0`
    (non-interacting; cap the number of cells), and `s * n` rounding up to `n`.
- Energy API (zero allocation):
  - `particle_energy(sys, i)` — energy of particle `i` with all others.
  - `particle_energy(sys, s, σ, skip1, skip2, emax)` — energy of a (virtual) particle of
    diameter `σ` at fractional position `s` with all particles except `skip1` and
    `skip2` (use 0 for none). Stops early and returns a value `> emax` as soon as the
    partial sum exceeds `emax` when `is_repulsive(sys.potential)`.
  - `total_energy(sys; emax=Inf)` — full recompute including `energy_lrc`; same early
    exit. Does not modify `sys`. Summed in (and returns) `energy_type(T)`.
  - `total_virial(sys)` — `W = Σ_{i<j} pair_virial`, in `energy_type(T)`.
  - `pressure(sys, kT)` — `ρ kT + W / (D V) + pressure_lrc`, in `energy_type(T)`.
  - `check_energy(sys) -> (running, recomputed)`; `sync_energy!(sys)` sets the running
    energy to the recomputed one.
- Initialization: `box_for_density(Val(D), N, ρ)`, `box_for_packing_fraction(Val(D),
  diameters, φ)`, `lattice_positions(box, N)` (hypercubic, `ceil(N^(1/D))` per side,
  offset by half a spacing), `random_positions(rng, box, N)`,
  `insert_hard_spheres(rng, box, diameters; max_tries=10^6)` (random sequential
  insertion, largest first).

### `src/ramps.jl`, `src/io/extxyz.jl` (wave 1, agent "io")
- `LinearRamp(initial, final, nsweeps)`, `ExponentialRamp(initial, final, nsweeps)`,
  callable `(ramp)(sweep::Integer)`, constant after `nsweeps`. `value_at(x::Real, sweep)
  = x`, `value_at(f, sweep) = f(sweep)`.
- `write_extxyz(io::IO, box, positions, diameters; sweep=0)` and
  `write_extxyz(io::IO, sys::System; sweep=0)`, path variants with `mode="w"`/`"a"`.
  MolecularDynamics.jl compatible extended XYZ (9-component `Lattice`, 2D padded with
  `pbc="T T F"`), full precision floats. `read_extxyz(path; frame=:last)` →
  `(; box, positions, diameters, sweep)` (`frame` may be an `Int`); `read_extxyz(path,
  Val(D); frame)` type-stable variant.

### `src/ensembles.jl`, `src/moves/*.jl`, `src/simulation.jl` (wave 2)
- `abstract type Ensemble`; `NVT(kT)`, `NPT(kT, pressure)` (each a `Real` or a callable
  of the sweep number). `conditions(ens, sweep, T) -> NamedTuple` (`(; kT)` or
  `(; kT, pressure)`), evaluated once per sweep. Finite `kT ≥ 0` (`> 0` in NPT) and a
  finite pressure, checked by the constructors for numbers and by `conditions` (an
  `ArgumentError` naming the sweep) for callables.
- `abstract type MCMove`; `attempt!(move, sys, cond, rng) -> Bool`; `name(move)::Symbol`.
  - `Displacement(δ=0.1; target=0.4, interval=1000, bounds=(1e-4, Inf))` — uniform in
    the cube `[-δ, δ]^D`; δ is also capped at half the smallest box width.
  - `Swap(; max_dsigma=Inf)` — uniform pair `i ≠ j`; pairs with `|σi-σj| > max_dsigma`
    are rejected without computing energies (still counted).
  - `VolumeMove(δlnV=0.01; target=0.3, interval=100, bounds=(1e-5, 1.0))` — isotropic,
    uniform in `ln V`; `ln A = -(ΔU + PΔV)/kT + (N+1) ΔlnV`. Rejected if the new box
    violates `fits_cutoff`. Hard potentials skip the energy on expansion (`is_hard`
    requires pair energies `0`/`Inf` and non-increasing in `r`).
  - Shared `MoveStats` (attempts/accepts since `reset_stats!`) and `AdaptiveStep`
    (window counters, bounded multiplicative update `δ *= clamp(ratio/target, 2/3, 3/2)`,
    clamped to bounds).
- `Simulation(system, ensemble, moves::Pair{<:MCMove,<:Real}...; seed, rng)` with
  fields `system, ensemble, moves::Tuple, cumweights::NTuple, rng, seed::UInt64,
  sweep::Int`. Validates: `VolumeMove` requires `NPT` and `NPT` requires a
  `VolumeMove`; unique move names; positive weights (normalized). A task-local `rng`
  (`Random.default_rng()`) is replaced by a `copy` (a `Xoshiro`), so that checkpoints
  store its state.
- `run!(sim, nsweeps; observers=(), adapt=false, energy_check=1000)`,
  `equilibrate!(sim, nsweeps; kw...) = run!(…; adapt=true)`. One sweep = N attempts.
  `reset_stats!(sim)`, `acceptance_rate(sim) -> NamedTuple`, `step_size(sim) ->
  NamedTuple`.

### `src/observers/*.jl`, `src/checkpoint.jl` (wave 3)
- `abstract type Schedule`, `isdue(schedule, sweep)`; `Every(n)`, `LogSchedule(; base=1.35,
  first=1)`.
- `abstract type Quantity`; `measure(q, sim)`, `label(q)`: `PotentialEnergy()` (per
  particle), `Pressure()` (virial), `VirtualPressure(; xi=1e-3)` (hard potentials),
  `Density()`, `PackingFraction()`, `Volume()`, `Acceptance(name)`, `StepSize(name)`.
- `abstract type Observer`; `observe!(obs, sim)` called after every sweep;
  `Base.flush`, `Base.close`. `ThermoWriter(path, quantities...; schedule=Every(100))`,
  `TrajectoryWriter(path; schedule=Every(1000))`, `TimeSeries(quantities...; schedule)`,
  `Callback(f; schedule)`, `ProgressPrinter(; schedule)`. Files opened lazily and kept
  open; headers include the seed.
- `save_checkpoint(path, sim)`, `load_checkpoint(path)` (Serialization, includes RNG).

## Tests

`test/runtests.jl` runs groups selected by `SWAPMC_TEST_GROUP` (`unit` default, `stat`,
`all`). Shared helpers in `test/helpers.jl` (toy potentials `GaussianCore`,
`EvaluateGaussian`, `Ideal`, `NoMethods`; brute-force `brute_energy`, `brute_virial`,
`brute_neighbors`; statistics `block_stderr`, `check_mean`, `chi2_crit`, `ks_stat`;
`allocs`). Each test file is standalone given the helpers:

    julia --project=test -e 'include("test/helpers.jl"); include("test/test_box.jl")'

Statistical tests use block-averaged standard errors with an upper bound on the error
(so a broken, noisy sampler cannot pass), seeded with `StableRNG`.

Heavier physics checks live in `validation/` (run by hand): hard-sphere EOS (Kolafa),
hard disks (Henderson), Lennard-Jones vs NIST SRSW data, swap vs no-swap, comparison
with MolecularDynamics.jl. Benchmarks in `benchmark/` against the original `swapmc.jl`.

## Milestones

0. Scaffold ✔ 1. Physics primitives ✔ 2. Cell list + energy ✔ 3. NVT + displacement +
driver 4. Swap + IO (IO ✔) 5. NPT + pressure 6. Ramps (✔), checkpoints, docs
7. Performance (half-width cells, packed layout), anisotropic volume moves, triclinic
boxes.
