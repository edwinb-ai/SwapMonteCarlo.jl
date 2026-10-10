# Structural relaxation measurements for `swap_vs_noswap.jl`: unwrapped trajectories and
# the self-overlap function.

"""
    Trajectory{D}

Unwrapped Cartesian positions recorded every `every` sweeps. Consecutive snapshots are
joined with the minimum image convention, which is valid as long as no particle moves by
more than half a box length between two snapshots.
"""
struct Trajectory{D}
    every::Int
    last::Vector{SVector{D,Float64}}
    unwrapped::Vector{SVector{D,Float64}}
    snapshots::Vector{Vector{SVector{D,Float64}}}
end

function Trajectory(sys, every::Integer)
    x = positions(sys)
    return Trajectory(Int(every), copy(x), copy(x), [copy(x)])
end

"""
    record!(traj, sys)

Append the current unwrapped positions of `sys` (fixed orthorhombic box) to `traj`.
"""
function record!(traj::Trajectory, sys)
    L = sys.box.lengths
    x = positions(sys)
    for i in eachindex(x)
        d = x[i] - traj.last[i]
        traj.unwrapped[i] += d - L .* round.(d ./ L)
        traj.last[i] = x[i]
    end
    push!(traj.snapshots, copy(traj.unwrapped))
    return nothing
end

"""
    cage_neighbors(x, L, rcage) -> Vector{Vector{Int}}

Neighbors of every particle closer than `rcage` (minimum image, `O(N²)`).
"""
function cage_neighbors(x, L, rcage)
    N = length(x)
    nbrs = [Int[] for _ in 1:N]
    for i in 1:N, j in (i + 1):N
        d = x[j] - x[i]
        d = d - L .* round.(d ./ L)
        if sum(abs2, d) < rcage^2
            push!(nbrs[i], j)
            push!(nbrs[j], i)
        end
    end
    return nbrs
end

"""
    self_overlap(traj, L; a=0.3, cage=false, rcage=1.5, norigins=50) -> (lags, Q)

Self-overlap `Q(t) = ⟨(1/N) Σ_i θ(a - |Δr_i(t)|)⟩` averaged over up to `norigins` time
origins, at all lags (in sweeps) that are powers of about 1.2 times the snapshot
interval. With `cage=true` the displacements are cage-relative: the mean displacement of
the neighbors within `rcage` at the time origin is subtracted, which removes the
long-wavelength fluctuations that dominate displacements in 2D (Mermin–Wagner).
"""
function self_overlap(traj::Trajectory, L; a=0.3, cage=false, rcage=1.5, norigins=50)
    S = traj.snapshots
    nsnap = length(S)
    N = length(first(S))
    lagidx = unique(round.(Int, 1.2 .^ (0:200)))
    filter!(m -> m < nsnap, lagidx)
    Q = zeros(length(lagidx))
    counts = zeros(Int, length(lagidx))
    stride = max(1, (nsnap - 1) ÷ norigins)
    for o in 1:stride:nsnap
        x0 = S[o]
        nbrs = cage ? cage_neighbors(x0, L, rcage) : nothing
        for (k, m) in enumerate(lagidx)
            o + m <= nsnap || break
            x1 = S[o + m]
            q = 0
            for i in 1:N
                d = x1[i] - x0[i]
                if cage && !isempty(nbrs[i])
                    c = zero(d)
                    for j in nbrs[i]
                        c += x1[j] - x0[j]
                    end
                    d -= c / length(nbrs[i])
                end
                q += sum(abs2, d) < a^2
            end
            Q[k] += q / N
            counts[k] += 1
        end
    end
    keep = counts .> 0
    return lagidx[keep] .* traj.every, Q[keep] ./ counts[keep]
end

"""
    relaxation_time(lags, Q; level=exp(-1)) -> Float64

First time at which `Q` falls below `level`, interpolated linearly in `log t`; `Inf` if
it never does.
"""
function relaxation_time(lags, Q; level=exp(-1))
    for k in 2:length(Q)
        if Q[k] < level
            t1, t2 = log(lags[k - 1]), log(lags[k])
            f = (Q[k - 1] - level) / (Q[k - 1] - Q[k])
            return exp(t1 + f * (t2 - t1))
        end
    end
    return Inf
end
