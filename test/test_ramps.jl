# Tests of the temperature and pressure ramps and of `value_at`.

@testset "LinearRamp" begin
    ramp = LinearRamp(2.0, 0.5, 100)
    @test ramp(0) === 2.0
    @test ramp(100) === 0.5
    @test ramp(101) === 0.5
    @test ramp(10^9) === 0.5
    # Sweeps before the start of the ramp give the initial value.
    @test ramp(-5) === 2.0
    @test ramp(50) ≈ (2.0 + 0.5) / 2
    @test ramp(25) ≈ 2.0 - 1.5 / 4
    values = ramp.(0:150)
    @test issorted(values; rev=true)
    @test all(0.5 .<= values .<= 2.0)

    # Increasing ramps, e.g. compressing with a pressure ramp, and integer arguments.
    pressure = LinearRamp(1, 10, 9)
    @test pressure isa LinearRamp{Float64}
    @test pressure(0) === 1.0
    @test pressure(3) ≈ 4.0
    @test pressure(9) === 10.0
    @test issorted(pressure.(-3:20))

    # A single sweep ramp jumps to the final value.
    jump = LinearRamp(1.0, 3.0, 1)
    @test (jump(0), jump(1), jump(2)) === (1.0, 3.0, 3.0)

    # The floating point type is preserved.
    ramp32 = LinearRamp(2.0f0, 1.0f0, 10)
    @test ramp32 isa LinearRamp{Float32}
    @test ramp32(5) isa Float32
    @test ramp32(5) ≈ 1.5f0

    # Constant ramps and negative values (e.g. tension) are allowed.
    @test all(==(0.7), LinearRamp(0.7, 0.7, 10).(0:20))
    @test LinearRamp(-1.0, 1.0, 2)(1) == 0.0

    @test_throws ArgumentError LinearRamp(1.0, 2.0, 0)
    @test_throws ArgumentError LinearRamp(1.0, 2.0, -3)
    @test_throws ArgumentError LinearRamp(Inf, 2.0, 10)
    @test_throws ArgumentError LinearRamp(1.0, NaN, 10)
end

@testset "ExponentialRamp" begin
    ramp = ExponentialRamp(2.0, 0.02, 100)
    @test ramp(0) === 2.0
    @test ramp(100) === 0.02
    @test ramp(1000) === 0.02
    @test ramp(-1) === 2.0
    @test ramp(50) ≈ sqrt(2.0 * 0.02)
    values = ramp.(0:150)
    @test issorted(values; rev=true)
    @test all(0.02 .<= values .<= 2.0)
    # Equal numbers of sweeps change the value by equal factors.
    ratios = values[2:101] ./ values[1:100]
    @test all(r -> r ≈ ratios[1], ratios)

    # Increasing pressure ramp, integer arguments.
    pressure = ExponentialRamp(1, 100, 10)
    @test pressure isa ExponentialRamp{Float64}
    @test pressure(5) ≈ 10.0
    @test pressure(10) === 100.0
    @test issorted(pressure.(0:20))

    # Negative end points of the same sign are allowed.
    negative = ExponentialRamp(-1.0, -4.0, 2)
    @test negative(1) ≈ -2.0
    @test issorted(negative.(0:3); rev=true)

    ramp32 = ExponentialRamp(4.0f0, 1.0f0, 2)
    @test ramp32 isa ExponentialRamp{Float32}
    @test ramp32(1) isa Float32
    @test ramp32(1) ≈ 2.0f0

    @test ExponentialRamp(1.0, 2.0, 1)(1) === 2.0
    @test all(==(3.0), ExponentialRamp(3.0, 3.0, 5).(0:10))

    @test_throws ArgumentError ExponentialRamp(1.0, 2.0, 0)
    @test_throws ArgumentError ExponentialRamp(0.0, 2.0, 10)
    @test_throws ArgumentError ExponentialRamp(1.0, 0.0, 10)
    @test_throws ArgumentError ExponentialRamp(-1.0, 2.0, 10)
    @test_throws ArgumentError ExponentialRamp(1.0, Inf, 10)
end

@testset "value_at" begin
    @test SMC.value_at(1.5, 0) === 1.5
    @test SMC.value_at(2, 10^6) === 2
    @test SMC.value_at(LinearRamp(1.0, 2.0, 10), 5) ≈ 1.5
    @test SMC.value_at(ExponentialRamp(1.0, 4.0, 10), 5) ≈ 2.0
    @test SMC.value_at(sweep -> sweep / 2, 3) === 1.5
    @test (@inferred SMC.value_at(LinearRamp(1.0, 2.0, 10), 3)) isa Float64
    @test (@inferred SMC.value_at(ExponentialRamp(1.0, 2.0, 10), 3)) isa Float64
    ramp = LinearRamp(1.0, 2.0, 10)
    @test allocs(SMC.value_at, ramp, 3) == 0
end
