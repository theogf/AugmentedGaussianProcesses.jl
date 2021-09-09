seed!(42)
L = 3
D = 10
N = 20
nSamples = 10
b = 5
x = rand(N, D)
y = rand(N)

@testset "Gibbs Sampling" begin
    i = GibbsSampling(; nBurnin=0)
    @test repr(i) == "Gibbs Sampler"
    i = AGP.tuple_inference(i, L, D, N, b, [], [])

    @test AGP.getρ(i) == 1
    @test AGP.isStochastic(i) == false
end
