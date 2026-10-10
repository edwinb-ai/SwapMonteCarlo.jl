# Moves and acceptance

A [`Simulation`](@ref) attempts ``N`` moves per sweep, each chosen at random with
probability proportional to its weight. Every move is a Metropolis step: a symmetric
proposal followed by acceptance with probability ``\min(1, A)``.

| Move | Proposal | ``\ln A`` |
|---|---|---|
| [`Displacement`](@ref)`(δ)` | particle ``i`` uniform, ``\mathbf{r}_i \to \mathbf{r}_i + \boldsymbol{\delta}``, ``\boldsymbol{\delta}`` uniform in ``[-\delta, \delta]^D`` | ``-\beta\Delta U`` |
| [`Swap`](@ref)`()` | ordered pair ``i \ne j`` uniform, ``\sigma_i \leftrightarrow \sigma_j`` | ``-\beta\Delta U`` |
| [`VolumeMove`](@ref)`(δlnV)` | ``\ln V \to \ln V + \Delta``, ``\Delta`` uniform in ``[-\delta, \delta]``, positions scaled | ``-\beta(\Delta U + P\Delta V) + (N + 1)\Delta\ln V`` |

with ``\beta = 1/k_BT``.

## Metropolis with the threshold drawn first

The textbook implementation computes ``A`` and compares it with a uniform random number.
Here the random number is drawn *before* the new energy is computed. Let ``\xi`` be
exponentially distributed (`randexp`), so that ``P(\xi \ge x) = e^{-x}`` for ``x \ge 0``.
A move is accepted if and only if ``\ln A \ge -\xi``, which happens with probability

```math
P(\xi \ge -\ln A) = \min(1, A),
```

exactly the Metropolis probability. Writing ``\ln A = -\beta\Delta U + b`` with a bias
``b`` that does not depend on the new energy (``b = 0`` for displacements and swaps), the
criterion becomes

```math
U_\text{new} \le U_\text{max} = U_\text{old} + k_BT\,(b + \xi).
```

The threshold ``U_\text{max}`` is known before ``U_\text{new}`` is computed. For a
repulsive potential ([`is_repulsive`](@ref SwapMonteCarlo.is_repulsive)), the partial sums
of ``U_\text{new}`` only grow, so the sum stops as soon as it exceeds ``U_\text{max}``: the
move is then certainly rejected. Most rejected moves in a dense liquid stop after a few
pairs, and for hard spheres after the first overlap. In addition, no exponential is ever
evaluated, and ``\Delta U = 0`` is always accepted, also at ``k_BT = 0``, where the rule
becomes ``U_\text{new} \le U_\text{old}`` (a quench).

## Displacements

A particle chosen uniformly is displaced by a vector uniform in the cube
``[-\delta, \delta]^D``. The reverse displacement has the same probability, so the
proposal is symmetric. The step is capped at half the smallest box width, beyond which
periodic images would make the proposal asymmetric. The energy of the particle at its new
position excludes the particle itself, so the cell list is updated only if the move is
accepted (in ``O(1)``).

## Swaps

A *swap* exchanges the diameters of two particles (Grigera and Parisi, PRE 63, 045102,
2001). The diameters are a fixed set; swaps sample their assignment to the particles
together with the positions. Since particles differ only by their diameters, relabeling
them shows that the equilibrium distribution of any quantity that does not depend on the
labels is the same as in the ordinary polydisperse system, but relaxation is much faster:
a particle that cannot move because it is caged can change its size instead.

**Detailed balance.** The ordered pair ``(i, j)``, ``i \ne j``, is chosen with probability
``1/[N(N-1)]``. The reverse move swaps the same two particles back and is proposed with
the same probability, so the proposal is symmetric and ``\ln A = -\beta\Delta U``.
With `Swap(; max_dsigma)`, pairs with ``|\sigma_i - \sigma_j| >`` `max_dsigma` are rejected
without computing energies. This condition is the same for a move and its reverse, so the
proposal stays symmetric (the rejected attempts count as rejections).

**Energy change.** Only the pairs involving ``i`` or ``j`` change. The energy of the
pair ``i``-``j`` itself does not, because ``u(r_{ij}; \sigma_i, \sigma_j) = u(r_{ij};
\sigma_j, \sigma_i)``, so it is left out:

```math
\Delta U = \sum_{k \ne i,j} \left[u(r_{ik}; \sigma_j, \sigma_k) + u(r_{jk}; \sigma_i, \sigma_k)
    - u(r_{ik}; \sigma_i, \sigma_k) - u(r_{jk}; \sigma_j, \sigma_k)\right].
```

This is why pair energies must be symmetric in the diameters, which [`System`](@ref)
checks. Swapping two equal diameters changes nothing and is accepted without computing
anything.

## Volume moves

In the ``NPT`` ensemble, with fractional coordinates ``\mathbf{s}_i = \mathbf{r}_i / L``
(componentwise), the configurational integral is ``\int d\mathbf{r}^N = V^N \int
d\mathbf{s}^N``, so the probability density of ``(V, \mathbf{s}^N)`` is

```math
p(V, \mathbf{s}^N) \propto V^N e^{-\beta [U(V, \mathbf{s}^N) + PV]}.
```

[`VolumeMove`](@ref) proposes ``\ln V' = \ln V + \Delta`` with ``\Delta`` uniform in
``[-\delta, \delta]``, which is symmetric in ``\ln V`` (so it is equally efficient at all
volumes and never proposes a negative volume). Since ``dV = V\,d\ln V``, the density of
``(\ln V, \mathbf{s}^N)`` is ``\propto V^{N+1} e^{-\beta(U + PV)}``, and the Metropolis
ratio is

```math
\ln A = -\beta\left[U(V') - U(V) + P(V' - V)\right] + (N + 1)\ln\frac{V'}{V}.
```

The extra ``+1`` is the Jacobian of sampling ``\ln V`` instead of ``V``; without it the
density would be biased by a factor ``1/V``. The box is scaled isotropically and the
fractional coordinates do not change, so a rejected move just restores the old box. Boxes
too small for the largest cutoff (minimum image convention) are rejected. The new energy
is recomputed from scratch, ``O(N)``, with early termination for repulsive potentials; for
hard potentials an expansion cannot create overlaps and is accepted without computing
anything when the ``PV`` and Jacobian terms allow it. The cell list is rebuilt only when
the number of cells has to change.

Since a volume move costs about as much as ``N`` displacements, a weight of order
``1/N`` to ``10/N`` relative to displacements (one to ten volume moves per sweep) is a
good choice.

## Step sizes: tuned during equilibration only

Displacement and volume moves have an [`AdaptiveStep`](@ref SwapMonteCarlo.AdaptiveStep):
a step size with a target acceptance ratio (0.4 for displacements, 0.3 for volume moves),
an adaptation interval (1000 and 100 attempts) and bounds. During
[`equilibrate!`](@ref), after every `interval` attempts of a move its step is multiplied
by the ratio of the measured acceptance in that window to the target, limited to
``[2/3, 3/2]``, and clamped to the bounds:

```math
\delta \leftarrow \operatorname{clamp}\!\left(\delta \cdot
    \operatorname{clamp}\!\left(\frac{r}{r_\text{target}}, \tfrac23, \tfrac32\right),
    \delta_\text{min}, \delta_\text{max}\right).
```

[`run!`](@ref) never changes the step sizes. A proposal that depends on the past
acceptance history is not a Markov chain with a fixed transition matrix, and detailed
balance only holds for fixed step sizes, so production runs must use `run!`. The usual
pattern is

```julia
equilibrate!(sim, nequil)  # tunes the steps
reset_stats!(sim)          # acceptance counters from zero
run!(sim, nprod; observers=obs)
```

All parameters are keywords, e.g. `Displacement(0.1; target=0.3, interval=500,
bounds=(1e-4, 0.5))` or `VolumeMove(0.01; target=0.4)`. [`step_size`](@ref) and
[`acceptance_rate`](@ref) report the current values per move, and
[`StepSize`](@ref) and [`Acceptance`](@ref) record them in output files.

## Pressure estimators

### Virial pressure

[`Pressure`](@ref) and [`pressure`](@ref) give the virial pressure

```math
P = \frac{N k_BT}{V} + \frac{W}{DV} + P_\text{lrc}, \qquad
W = \sum_{i<j} w(r_{ij}), \quad w(r) = -r\,\frac{du}{dr},
```

with the temperature of the current sweep and the pair virial
[`pair_virial`](@ref SwapMonteCarlo.pair_virial). For a potential whose energy is a
differentiable function of the volume, ``-\partial U/\partial V = W/(DV)`` and the
identity

```math
\left\langle \frac{N k_BT}{V} - \frac{\partial U}{\partial V} \right\rangle_{NPT} = P
```

(from integrating ``\partial_V [V^N e^{-\beta(U + PV)}]`` over all volumes) holds exactly,
also for finite ``N``. The statistical tests use it to check the volume move.

### Virtual compressions for hard potentials

Hard spheres have no pair virial: their force is a delta function at contact.
[`VirtualPressure`](@ref)`(; xi=1e-3)` scales all lengths by ``(1 - \xi)^{1/D}``, which
shrinks the volume by the fraction ``\xi``, and counts the number ``n`` of pairs that
would then overlap (Eppenga and Frenkel, Mol. Phys. 52, 1303, 1984). The excess pressure
is ``\beta P_\text{ex} = -\ln P_\text{no overlap}/(\xi V)`` for ``\xi \to 0``, and the
instantaneous estimator

```math
P = k_BT \left(\frac{N}{V} + \frac{n}{\xi V}\right)
```

has a bias of relative order ``\xi`` (several overlaps, curvature of the contact value).
Smaller ``\xi`` reduces the bias but increases the noise, since fewer configurations have
any overlap; ``\xi N`` of order ``10^{-2}`` to ``1`` is a good range, and the bias can be
checked by comparing two values of ``\xi``.

### Truncated potentials and the impulsive term

A potential with an energy jump at its cutoff, such as `LennardJones(; shift=NoShift())`,
has an extra *impulsive* contribution to the pressure. The force of the truncated
potential ``u(r)\theta(r_c - r)`` contains ``u(r_c)\,\delta(r - r_c)``, which the pair
virial (computed for ``r < r_c``) misses. For one component it is

```math
P_\text{imp} = \frac{S_D}{2D}\,\rho^2 r_c^D\, u(r_c)\, g(r_c),
```

negative for Lennard-Jones (``u(r_c) < 0``); ``S_2 = 2\pi``, ``S_3 = 4\pi``. Integrating
the pressure tail correction by parts,

```math
\int_{r_c}^\infty w(r)\, r^{D-1} dr = r_c^D u(r_c) + D \int_{r_c}^\infty u(r)\, r^{D-1} dr,
```

shows that ``P_\text{lrc} = U_\text{lrc}/V + P_\text{imp}|_{g(r_c) = 1}``: the standard
pressure correction already contains the impulsive term with ``g(r_c) = 1``. In an
``NPT`` simulation the volume moves sample the thermodynamic pressure of the energy that
is actually simulated (including ``U_\text{lrc}(V)``), so the average of [`Pressure`](@ref)
compares with the imposed pressure as follows:

| Potential | ``\langle P_\text{virial}\rangle_{NPT} - P`` |
|---|---|
| continuous at ``r_c`` (`PolySoftSphere`, `PseudoHS`, `EnergyShift`, `ForceShift`), no tail correction | ``0`` |
| `NoShift`, no tail correction | ``-P_\text{imp}`` |
| `NoShift`, `tail_correction=true` | ``\frac{S_D}{2D}\rho^2 r_c^D u(r_c)\,[1 - g(r_c)] \approx 0`` |
| `EnergyShift` or `ForceShift`, `tail_correction=true` | ``P_\text{imp}\rvert_{g(r_c) = 1}`` |

In ``NVT`` the same terms decide which pressure the virial estimates. The NIST
Lennard-Jones reference data, for example, use `NoShift` with tail corrections, so that
the virial pressure approximates that of the full potential (see `examples/lj_nvt.jl`).

## Energy bookkeeping

The system keeps a running total energy, updated with the energy difference of every
accepted move. [`run!`](@ref) compares it with a full recomputation every `energy_check`
sweeps (1000 by default); a relative mismatch larger than ``10^{-8}`` gives a warning and
the running energy is replaced by the recomputed one.
[`check_energy`](@ref SwapMonteCarlo.check_energy) and
[`sync_energy!`](@ref SwapMonteCarlo.sync_energy!) do the same by hand.

## Custom moves

A move is a subtype of [`SwapMonteCarlo.MCMove`](@ref) with a method of
[`attempt!`](@ref SwapMonteCarlo.attempt!)`(move, sys, cond, rng) -> Bool`, a
[`name`](@ref SwapMonteCarlo.name) and a field `stats::MoveStats`; see the docstring of
`MCMove`. `cond` is the `NamedTuple` of external conditions of the current sweep,
`(; kT)` or `(; kT, pressure)` (see [`conditions`](@ref SwapMonteCarlo.conditions)), and
the move must record its outcome with [`record!`](@ref SwapMonteCarlo.record!).
