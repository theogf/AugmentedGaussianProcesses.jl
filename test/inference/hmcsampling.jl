@testset "HMC Sampling" begin
    seed!(42)
    L = 3
    D = 10
    N = 20
    nSamples = 10
    b = 5
    i = HMCSampling(; nBurnin=0, thinning=1)

    @test repr(i) == "Hamilton Monte Carlo Sampler"

    @test AGP.ρ(i) == 1
    @test AGP.is_stochastic(i) == false

    i = AGP.init_sampler!(i, L, N, nSamples, false)

    @test i.sample_store == zeros(Float64, nSamples, N, L)

    i.n_iter = nSamples
    i = AGP.init_sampler!(i, L, N, nSamples, true)

    @test i.sample_store == zeros(Float64, 2 * nSamples, N, L)

    i.n_iter = 2 * nSamples
    i = AGP.init_sampler!(i, L, N, 2, false)

    @test i.sample_store == zeros(Float64, 2, N, L)

    i.n_iter = 1
    fs = [rand(N) for _ in 1:L]
    AGP.store_variables!(i, fs)
    @test i.sample_store[1, :, 1] == fs[1]
end
