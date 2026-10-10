"""
    SizeDistribution

Supertype of particle size (diameter) distributions. A new distribution must implement
[`quantile`](@ref); [`sample_diameters`](@ref) then works through inverse transform
sampling.
"""
abstract type SizeDistribution end

"""
    check_diameter(σ, name)

Throw an `ArgumentError` unless the diameter `σ` is positive and finite.
"""
function check_diameter(σ::Real, name::AbstractString)
    (σ > 0 && isfinite(σ)) ||
        throw(ArgumentError("$(name) must be positive and finite, got $(σ)"))
    return nothing
end

"""
    Monodisperse(σ)

All particles have the same diameter `σ`.
"""
struct Monodisperse <: SizeDistribution
    sigma::Float64
    function Monodisperse(σ::Real)
        check_diameter(σ, "σ")
        return new(σ)
    end
end

"""
    Bidisperse(σa, σb, fraction_a)

Binary mixture with a fraction `fraction_a` of particles of diameter `σa` and the rest of
diameter `σb`. [`sample_diameters`](@ref) produces exactly `round(Int, N * fraction_a)`
particles of diameter `σa`, in random order.
"""
struct Bidisperse <: SizeDistribution
    sigma_a::Float64
    sigma_b::Float64
    fraction_a::Float64
    function Bidisperse(σa::Real, σb::Real, fraction_a::Real)
        check_diameter(σa, "σa")
        check_diameter(σb, "σb")
        0 <= fraction_a <= 1 ||
            throw(ArgumentError("fraction_a must be in [0, 1], got $(fraction_a)"))
        return new(σa, σb, fraction_a)
    end
end

"""
    InversePowerLaw(σmin, σmax; exponent=3)

Continuous distribution with density `P(σ) ∝ σ^(-exponent)` on `[σmin, σmax]`. The
default `exponent = 3` is the distribution of Ninarello, Berthier and Coslovich (PRX 7,
021039, 2017), whose mean is `2 σmin σmax / (σmin + σmax)`. The exponent must differ from
`1`.
"""
struct InversePowerLaw <: SizeDistribution
    sigma_min::Float64
    sigma_max::Float64
    exponent::Float64
    function InversePowerLaw(σmin::Real, σmax::Real; exponent::Real=3)
        check_diameter(σmin, "σmin")
        check_diameter(σmax, "σmax")
        σmin < σmax || throw(ArgumentError("need σmin < σmax, got $(σmin) and $(σmax)"))
        (isfinite(exponent) && exponent != 1) ||
            throw(ArgumentError("exponent must be finite and different from 1"))
        return new(σmin, σmax, exponent)
    end
end

"""
    UniformSizes(σmin, σmax)

Diameters uniformly distributed in `[σmin, σmax]`.
"""
struct UniformSizes <: SizeDistribution
    sigma_min::Float64
    sigma_max::Float64
    function UniformSizes(σmin::Real, σmax::Real)
        check_diameter(σmin, "σmin")
        check_diameter(σmax, "σmax")
        σmin < σmax || throw(ArgumentError("need σmin < σmax, got $(σmin) and $(σmax)"))
        return new(σmin, σmax)
    end
end

"""
    quantile(d::SizeDistribution, u)

Inverse cumulative distribution function of `d` at `u ∈ [0, 1]`.
"""
function quantile(d::SizeDistribution, u::Real)
    0 <= u <= 1 || throw(DomainError(u, "quantile needs 0 ≤ u ≤ 1"))
    return unsafe_quantile(d, Float64(u))
end

"""
    unsafe_quantile(d::SizeDistribution, u::Float64)

[`quantile`](@ref) without the range check on `u`.
"""
unsafe_quantile(d::Monodisperse, u::Float64) = d.sigma
unsafe_quantile(d::Bidisperse, u::Float64) = u < d.fraction_a ? d.sigma_a : d.sigma_b
unsafe_quantile(d::UniformSizes, u::Float64) = d.sigma_min + u * (d.sigma_max - d.sigma_min)
function unsafe_quantile(d::InversePowerLaw, u::Float64)
    (a, b, p) = (d.sigma_min, d.sigma_max, d.exponent)
    # Invert F(σ) = (a^(1-p) - σ^(1-p)) / (a^(1-p) - b^(1-p)).
    if p == 3
        σ = inv(sqrt((1 - u) / (a * a) + u / (b * b)))
    else
        q = 1 - p
        σ = ((1 - u) * a^q + u * b^q)^inv(q)
    end
    # Guard against rounding just outside the support.
    return clamp(σ, a, b)
end

"""
    diameter_moment(d::SizeDistribution, k::Integer)

Analytic moment `⟨σ^k⟩` of the distribution `d`.
"""
diameter_moment(d::Monodisperse, k::Integer) = d.sigma^k
function diameter_moment(d::Bidisperse, k::Integer)
    return d.fraction_a * d.sigma_a^k + (1 - d.fraction_a) * d.sigma_b^k
end
function diameter_moment(d::UniformSizes, k::Integer)
    (a, b) = (d.sigma_min, d.sigma_max)
    return (b^(k + 1) - a^(k + 1)) / ((k + 1) * (b - a))
end
function diameter_moment(d::InversePowerLaw, k::Integer)
    (a, b, p) = (d.sigma_min, d.sigma_max, d.exponent)
    # Normalization Z = ∫_a^b σ^(-p) dσ.
    Z = (b^(1 - p) - a^(1 - p)) / (1 - p)
    q = k + 1 - p
    I = q == 0 ? log(b / a) : (b^q - a^q) / q
    return I / Z
end

"""
    mean_diameter(d::SizeDistribution)

Analytic mean diameter of `d`.
"""
mean_diameter(d::SizeDistribution) = diameter_moment(d, 1)

"""
    sample_diameters(rng, d::SizeDistribution, N; stratified=false) -> Vector{Float64}

Draw `N` diameters from `d` by inverse transform sampling.

With `stratified=true` the diameters are the quantiles `σ_k = F⁻¹((k - 1/2)/N)`,
`k = 1, …, N`, in random order, so that the moments of the sample match those of the
distribution up to `O(1/N²)`. A [`Bidisperse`](@ref) distribution always produces exactly
`round(Int, N * fraction_a)` small particles, in random order.
"""
function sample_diameters(
    rng::AbstractRNG, d::SizeDistribution, N::Integer; stratified::Bool=false
)
    N >= 0 || throw(ArgumentError("number of particles must be non-negative, got $(N)"))
    if stratified
        σs = [unsafe_quantile(d, (k - 0.5) / N) for k in 1:N]
        shuffle!(rng, σs)
    else
        σs = [unsafe_quantile(d, rand(rng, Float64)) for _ in 1:N]
    end
    return σs
end

function sample_diameters(
    rng::AbstractRNG, d::Bidisperse, N::Integer; stratified::Bool=false
)
    N >= 0 || throw(ArgumentError("number of particles must be non-negative, got $(N)"))
    na = round(Int, N * d.fraction_a)
    σs = [k <= na ? d.sigma_a : d.sigma_b for k in 1:N]
    return shuffle!(rng, σs)
end

function sample_diameters(
    rng::AbstractRNG, d::Monodisperse, N::Integer; stratified::Bool=false
)
    N >= 0 || throw(ArgumentError("number of particles must be non-negative, got $(N)"))
    return fill(d.sigma, N)
end

"""
    polydispersity(σs)

Polydispersity of a set of diameters: the population standard deviation divided by the
mean.

    polydispersity(d::SizeDistribution)

Analytic polydispersity of the distribution `d`.
"""
function polydispersity(σs::AbstractVector{<:Real})
    n = length(σs)
    n > 0 || throw(ArgumentError("polydispersity of an empty set of diameters"))
    m = sum(σs) / n
    v = sum(σ -> (σ - m)^2, σs) / n
    return sqrt(v) / m
end

function polydispersity(d::SizeDistribution)
    m1 = diameter_moment(d, 1)
    m2 = diameter_moment(d, 2)
    return sqrt(max(m2 - m1 * m1, zero(m1))) / m1
end
