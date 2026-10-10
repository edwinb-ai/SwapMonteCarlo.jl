# Shared test helpers: toy potentials, brute-force references and statistics utilities.
#
# Run a single test file with
#     julia --project=test -e 'include("test/helpers.jl"); include("test/test_box.jl")'

using SwapMonteCarlo
using Test
using LinearAlgebra: dot, norm
using Random: Random, Xoshiro
using StableRNGs: StableRNG
using StaticArrays: SVector
using Statistics: mean, std, var

import SwapMonteCarlo: Potential, pair_energy, pair_virial, cutoff, evaluate

const SMC = SwapMonteCarlo

const TEST_SEED = parse(Int, get(ENV, "SWAPMC_TEST_SEED", "20261010"))

"""
    test_rng(k)

Reproducible random number generator for the testset numbered `k`.
"""
test_rng(k::Integer) = StableRNG(TEST_SEED + k)

# --------------------------------------------------------------------------------------
# Toy potentials (they also exercise the user extension interface)
# --------------------------------------------------------------------------------------

"""
Smooth, bounded and truncated Gaussian core, `u = A exp(-r²)` for `r < rc`. Implements
only `pair_energy` and `cutoff`.
"""
struct GaussianCore <: Potential
    A::Float64
    rc::Float64
end
GaussianCore(; A=2.0, rc=3.0) = GaussianCore(A, rc)
pair_energy(p::GaussianCore, r2, σi, σj) = r2 < p.rc^2 ? p.A * exp(-r2) : zero(r2)
cutoff(p::GaussianCore, σi, σj) = p.rc

"""
Gaussian core with size dependent range defined through the MolecularDynamics.jl-style
`evaluate(pot, r, σi, σj) -> (u, f)` interface, `u = A exp(-(r/σij)²)` for `r < 2.5σij`.
"""
struct EvaluateGaussian <: Potential
    A::Float64
end
function evaluate(p::EvaluateGaussian, r, σi, σj)
    σ = (σi + σj) / 2
    r < 2.5σ || return (zero(r), zero(r))
    u = p.A * exp(-(r / σ)^2)
    return (u, 2 * r / σ^2 * u)
end
cutoff(::EvaluateGaussian, σi, σj) = 2.5 * (σi + σj) / 2

"""Non-interacting particles (ideal gas)."""
struct Ideal <: Potential end
pair_energy(::Ideal, r2, σi, σj) = zero(r2)
pair_virial(::Ideal, r2, σi, σj) = zero(r2)
cutoff(::Ideal, σi, σj) = 0.0

"""A potential that implements nothing, to check error messages."""
struct NoMethods <: Potential end

# --------------------------------------------------------------------------------------
# Brute-force references (independent of the package internals)
# --------------------------------------------------------------------------------------

"""
    min_image_brute(d, L)

Minimum image of a Cartesian separation `d` in an orthorhombic box with sides `L`.
"""
min_image_brute(d, L) = d .- L .* round.(d ./ L)

"""
    brute_energy(L, x, σ, pot) -> U

All-pairs minimum-image total energy, `O(N²)`. `L` are the box lengths and `x` Cartesian
positions.
"""
function brute_energy(L, x, σ, pot)
    U = 0.0
    for i in eachindex(x), j in (i + 1):lastindex(x)
        d = min_image_brute(x[i] - x[j], L)
        U += pair_energy(pot, dot(d, d), σ[i], σ[j])
    end
    return U
end

"""
    brute_virial(L, x, σ, pot) -> W

All-pairs minimum-image total virial `W = Σ r_ij f_ij`.
"""
function brute_virial(L, x, σ, pot)
    W = 0.0
    for i in eachindex(x), j in (i + 1):lastindex(x)
        d = min_image_brute(x[i] - x[j], L)
        r2 = dot(d, d)
        r2 < cutoff(pot, σ[i], σ[j])^2 || continue
        W += pair_virial(pot, r2, σ[i], σ[j])
    end
    return W
end

"""
    brute_neighbors(L, x, σ, pot) -> Set{Tuple{Int,Int}}

All pairs `i < j` closer than their pair cutoff.
"""
function brute_neighbors(L, x, σ, pot)
    pairs = Set{Tuple{Int,Int}}()
    for i in eachindex(x), j in (i + 1):lastindex(x)
        d = min_image_brute(x[i] - x[j], L)
        dot(d, d) < cutoff(pot, σ[i], σ[j])^2 && push!(pairs, (i, j))
    end
    return pairs
end

"""
    uniform_positions(rng, L, N)

`N` uniformly random Cartesian positions in a box with sides `L::SVector`.
"""
function uniform_positions(rng, L::SVector{D,T}, N) where {D,T}
    return [L .* SVector(ntuple(_ -> rand(rng, T), Val(D))) for _ in 1:N]
end

# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------

"""
    block_stderr(x; min_blocks=32)

Standard error of the mean of a correlated time series with the Flyvbjerg–Petersen
blocking method. Returns the largest estimate over the blocking levels that still have at
least `min_blocks` blocks, which is conservative.
"""
function block_stderr(x::AbstractVector; min_blocks::Int=32)
    y = collect(float.(x))
    se = std(y) / sqrt(length(y))
    while length(y) ÷ 2 >= min_blocks
        n = length(y) ÷ 2
        y = [(y[2k - 1] + y[2k]) / 2 for k in 1:n]
        se = max(se, std(y) / sqrt(n))
    end
    return se
end

"""
    check_mean(xbar, se, ref; k=4, sys=0.0, se_max=Inf) -> Bool

Whether `xbar` agrees with `ref` within `k` standard errors plus a systematic allowance,
*and* the standard error is small enough for the test to be meaningful.
"""
function check_mean(xbar, se, ref; k=4, sys=0.0, se_max=Inf)
    return abs(xbar - ref) <= k * se + sys && se <= se_max
end

"""
    chi2_crit(k; z=3.719)

Approximate upper critical value of the χ² distribution with `k` degrees of freedom
(Wilson–Hilferty); the default `z` corresponds to `p = 1e-4`.
"""
chi2_crit(k; z=3.719) = k * (1 - 2 / (9k) + z * sqrt(2 / (9k)))^3

"""
    gamma_cdf_int(k, x)

CDF of the Gamma distribution with integer shape `k` and unit scale.
"""
function gamma_cdf_int(k::Integer, x)
    x <= 0 && return 0.0
    s = 0.0
    term = 1.0
    for j in 0:(k - 1)
        j > 0 && (term *= x / j)
        s += term
    end
    return 1 - exp(-x) * s
end

"""
    ks_stat(samples, cdf)

Kolmogorov–Smirnov statistic of `samples` against the continuous `cdf`.
"""
function ks_stat(samples, cdf)
    x = sort(samples)
    n = length(x)
    d = 0.0
    for (i, xi) in enumerate(x)
        F = cdf(xi)
        d = max(d, i / n - F, F - (i - 1) / n)
    end
    return d
end

"""Critical value of the KS statistic for `n` samples at level `α`."""
ks_crit(n; α=1e-4) = sqrt(-log(α / 2) / (2n))

"""
    allocs(f, args...)

Allocations of `f(args...)` after a warm-up call.
"""
function allocs(f::F, args...) where {F}
    f(args...)
    return @allocations f(args...)
end
