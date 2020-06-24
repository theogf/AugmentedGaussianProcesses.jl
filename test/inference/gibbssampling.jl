seed!(42)
L = 3
D = 10
N = 20
nSamples = 10
b = 5
x = rand(N, D)
y = rand(N)

@testset "Gibbs Sampling" begin
    i = GibbsSampling(nBurnin = 0)
    @test repr(i) == "Gibbs Sampler"
    i = AGP.tuple_inference(i, L, D, N, b)

    @test AGP.getρ(i) == 1
    @test AGP.isStochastic(i) == false

    i = AGP.init_sampler!(i, L, N, nSamples, false)

    @test i.sample_store == zeros(Float64, nSamples, N, L)

    i.nIter = nSamples
    i = AGP.init_sampler!(i, L, N, nSamples, true)

    @test i.sample_store == zeros(Float64, 2*nSamples, N, L)

    i.nIter = 2*nSamples
    i = AGP.init_sampler!(i, L, N, 2, false)

    @test i.sample_store == zeros(Float64, 2, N, L)

    i.nIter = 1
    fs = [rand(N) for _ in 1:L]
    AGP.store_variables!(i, fs)
    @test i.sample_store[1, :, 1] == fs[1]
end
