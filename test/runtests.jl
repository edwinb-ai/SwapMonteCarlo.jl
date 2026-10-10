# Test groups, selected with the environment variable SWAPMC_TEST_GROUP:
#
#   - "unit" (default): fast unit tests, invariants, allocations and code quality.
#   - "stat": statistical physics checks (a few minutes).
#   - "all": both.
#
# Example: SWAPMC_TEST_GROUP=all julia --project -e 'using Pkg; Pkg.test()'

include("helpers.jl")

const GROUP = get(ENV, "SWAPMC_TEST_GROUP", "unit")

const UNIT_TESTS = [
    "test_box.jl",
    "test_potentials.jl",
    "test_sizes.jl",
    "test_celllist.jl",
    "test_energy.jl",
    "test_initialization.jl",
    "test_ramps.jl",
    "test_io.jl",
    "test_moves.jl",
    "test_simulation.jl",
    "test_observers.jl",
    "test_checkpoint.jl",
    "test_quality.jl",
]

const STAT_TESTS = [
    "test_stats_helpers.jl",
    "test_acceptance.jl",
    "test_detailed_balance.jl",
    "test_ideal_gas.jl",
    "test_npt_virial.jl",
    "test_hard_spheres.jl",
    "test_swap_physics.jl",
]

files = String[]
GROUP in ("unit", "all") && append!(files, UNIT_TESTS)
GROUP in ("stat", "all") && append!(files, STAT_TESTS)

@testset "SwapMonteCarlo.jl" begin
    for file in files
        isfile(joinpath(@__DIR__, file)) || continue
        @testset "$(file)" begin
            include(file)
        end
    end
end
