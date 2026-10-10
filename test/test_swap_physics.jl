# Swap moves change the dynamics, not the thermodynamics.
#
# Swaps of diameters satisfy detailed balance with respect to the same Boltzmann
# distribution as displacements, so in a state where the displacement-only dynamics also
# equilibrates (a normal fluid) the averages must agree with and without swaps. The model
# is the 2D Ninarello soft-sphere mixture (SS12 with C² smoothing, non-additivity 0.2,
# P(σ) ∝ σ⁻³ on [0.73, 1.62]) at ρ = 1 and kT = 1, far above its glassy regime.
#
# Without swaps the energy has a slow component even in the fluid, because the
# arrangement of the diameters relaxes only by diffusion: at kT = 0.5 the energy needs
# about 10⁴ sweeps to forget the lattice start (N = 128), and at kT = 0.25 the
# displacement-only runs need of order 10⁵ sweeps, while the swap runs equilibrate
# within a few thousand. The displacement-only run is therefore longer, and the standard
# errors use at least 16 blocks so that the slow component is not underestimated.

"""
    swap_energy_run(rng, σ, kT, moves...; nequil, nsweeps)

Equilibrate and run the 2D soft-sphere mixture with diameters `σ` at `ρ = 1` and
temperature `kT` with the given moves. Returns the energy time series and the simulation.
"""
function swap_energy_run(rng, σ, kT, moves...; nequil, nsweeps)
    N = length(σ)
    box = box_for_density(Val(2), N, 1.0)
    pot = PolySoftSphere(; mixing=NonAdditive(0.2))
    sys = System(box, lattice_positions(box, N), copy(σ), pot)
    sim = Simulation(sys, NVT(kT), moves...; rng=rng)
    equilibrate!(sim, nequil)
    ts = TimeSeries(PotentialEnergy())
    run!(sim, nsweeps; observers=ts)
    return ts["energy"], sim
end

@testset "swap does not change the equilibrium energy (2D, kT = 1)" begin
    rng = test_rng(2400)
    N = 128
    kT = 1.0
    σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N; stratified=true)
    e0, _ = swap_energy_run(
        rng, σ, kT, Displacement(0.1) => 1.0; nequil=20_000, nsweeps=150_000
    )
    se0 = block_stderr(e0; min_blocks=16)
    # The energy per particle is about 1.13; a standard error below 0.3 % of it makes
    # the comparison meaningful.
    @test se0 <= 0.003
    for wswap in (0.2, 0.5)
        e1, sim = swap_energy_run(
            rng,
            σ,
            kT,
            Displacement(0.1) => 1 - wswap,
            Swap() => wswap;
            nequil=10_000,
            nsweeps=60_000,
        )
        se1 = block_stderr(e1; min_blocks=16)
        @test se1 <= 0.003
        @test abs(mean(e1) - mean(e0)) <= 4 * sqrt(se0^2 + se1^2)
        # Swaps are frequently accepted in this fluid, so they did act on the system,
        # and they only permute the diameters.
        @test acceptance_rate(sim).swap > 0.1
        @test sort(diameters(sim.system)) == sort(σ)
    end
end
