# Run all validation scripts, each in its own Julia process, and summarize.
#
#     julia --project=validation validation/run_all.jl            # everything
#     julia --project=validation validation/run_all.jl lj_nist    # a subset
#
# Each script writes `validation/results/<name>.md`. The total wall time is 30–40
# minutes on a recent laptop (one core; see `README.md`).

using Printf: @printf

const SCRIPTS = [
    "check_references", "ideal_gas_npt", "hd2d_eos", "hs3d_eos", "lj_nist", "swap_vs_noswap"
]

"""
    run_script(name) -> (ok::Bool, seconds)

Run `validation/<name>.jl` in a new Julia process with the validation environment.
"""
function run_script(name)
    path = joinpath(@__DIR__, "$(name).jl")
    cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) $(path)`
    println("=" ^ 80, "\n", name, "\n", "=" ^ 80)
    t = time()
    ok = success(pipeline(cmd; stdout=stdout, stderr=stderr))
    return ok, time() - t
end

"""
    overall_verdict(name) -> String

The "Overall" line of `results/<name>.md`, if any.
"""
function overall_verdict(name)
    path = joinpath(@__DIR__, "results", "$(name).md")
    isfile(path) || return "-"
    for line in eachline(path)
        startswith(line, "Overall:") && return strip(line[9:end])
    end
    return "-"
end

function main(args)
    names = isempty(args) ? SCRIPTS : args
    results = [(name, run_script(name)...) for name in names]
    println("\nSummary")
    for (name, ok, t) in results
        @printf(
            "  %-18s %-8s %7.0f s  %s\n",
            name,
            ok ? "ran" : "ERROR",
            t,
            overall_verdict(name)
        )
    end
    return all(r -> r[2], results)
end

main(ARGS) || exit(1)
