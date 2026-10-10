# Exact NPT identity: the average virial pressure equals the imposed pressure.
#
# The NPT partition function is `Δ = ∫ dV V^N e^{-βPV} ∫ ds e^{-βU(V, s)}` with fractional
# coordinates `s`. Integrating `∂/∂V [V^N e^{-βPV} e^{-βU}]` over all volumes gives zero
# (the integrand vanishes at both ends), hence
#
#     ⟨N kT / V - ∂U/∂V⟩_NPT = P   exactly, also for finite N,
#
# and `-∂U/∂V = W / (D V)` with the pair virial `W = -Σ r u'(r)`, so the left-hand side is
# the average of `pressure(sys, kT)`. This holds for potentials whose energy is a
# differentiable function of the volume. A truncated potential with a jump at the cutoff
# has an extra impulsive contribution that the pair virial misses; `PolySoftSphere` is
# C² at its cutoff, so the identity is exact. Any error in the acceptance rule of
# `VolumeMove` (pressure term, Boltzmann factor or Jacobian), in the virial or in the
# energy update after a volume change shows up as a bias. The temperatures differ from
# 1 so that confusing `P` with `βP` would be detected.

"""
    npt_virial_run(rng, D, N, kT, P; ρ0, nequil, nsweeps)

Polydisperse soft spheres (`PolySoftSphere`, non-additive, diameters ∝ σ⁻³ in
`[0.73, 1.62]`) in `D` dimensions at temperature `kT` and pressure `P`, started from a
lattice at density `ρ0`. Returns the time series of the virial pressure and the density.
"""
function npt_virial_run(rng, D, N, kT, P; ρ0, nequil, nsweeps)
    σ = sample_diameters(rng, InversePowerLaw(0.73, 1.62), N; stratified=true)
    box = box_for_density(Val(D), N, ρ0)
    sys = System(box, lattice_positions(box, N), σ, PolySoftSphere())
    # About two volume attempts per sweep.
    sim = Simulation(
        sys, NPT(kT, P), Displacement(0.1) => 1.0, VolumeMove(0.01) => 2 / N; rng=rng
    )
    equilibrate!(sim, nequil)
    ts = TimeSeries(Pressure(), Density(); schedule=Every(2))
    run!(sim, nsweeps; observers=ts)
    @test 0.15 <= acceptance_rate(sim).volume <= 0.5
    return ts["pressure"], ts["density"]
end

# (testset number, D, N, kT, P, initial density)
const NPT_VIRIAL_CASES = ((1, 2, 128, 0.5, 7.0, 1.0), (2, 3, 108, 0.75, 8.2, 1.0))

@testset "⟨P_virial⟩_NPT = P" begin
    for (k, D, N, kT, P, ρ0) in NPT_VIRIAL_CASES
        @testset "D = $D, N = $N, kT = $kT, P = $P" begin
            rng = test_rng(2200 + k)
            p, ρ = npt_virial_run(rng, D, N, kT, P; ρ0=ρ0, nequil=3000, nsweeps=20_000)
            se = block_stderr(p)
            # The instantaneous pressure fluctuates by about 8 % of P; the blocked
            # standard error of the mean is about 0.1 %, and the test requires it to be
            # below 0.5 %.
            @test check_mean(mean(p), se, P; se_max=0.005 * P)
            # The density must stay that of a dense fluid (no runaway expansion or
            # collapse).
            @test 0.85 <= mean(ρ) <= 1.15
        end
    end
end
