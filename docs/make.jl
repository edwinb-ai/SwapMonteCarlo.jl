# Build the documentation locally, from the repository root:
#
#     julia --project=docs -e 'import Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make.jl
#
# and open `docs/build/index.html`. The documentation is not deployed.

using Documenter
using SwapMonteCarlo

DocMeta.setdocmeta!(SwapMonteCarlo, :DocTestSetup, :(using SwapMonteCarlo); recursive=true)

makedocs(;
    modules=[SwapMonteCarlo],
    sitename="SwapMonteCarlo.jl",
    authors="edwinb-ai",
    format=Documenter.HTML(; prettyurls=false, edit_link=nothing, repolink=nothing),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Potentials" => "potentials.md",
        "Moves and acceptance" => "moves.md",
        "Simulations and output" => "simulation.md",
        "Testing and validation" => "testing.md",
        "API reference" => "api.md",
    ],
    doctest=true,
    checkdocs=:public,
    remotes=nothing,
)
