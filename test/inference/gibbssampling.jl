@testset "Gibbs Sampling" begin
    seed!(42)
    L = 3
    D = 10
    N = 20
    nSamples = 10
    b = 5
    i = GibbsSampling(; nBurnin=0)
    @test repr(i) == "Gibbs Sampler"

    @test AGP.ρ(i) == 1
    @test AGP.is_stochastic(i) == false
end
