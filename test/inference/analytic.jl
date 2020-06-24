using Test
using AugmentedGaussianProcesses

L = 3
D = 10
N = 20
b = 5
x = rand(N, D)
y = rand(N)

@testset "Analytic" begin
    i = Analytic()
    @test Analytic(ϵ = 0.0001f0) isa Analytic{Float32}
    @test repr(i) == "Analytic Inference"
    i = AGP.init_inference(i, L, D, N, b)
    i.xview = view(x, : , :)
    i.yview = view(y, :)

    @test AGP.xview(i) == view(x, : , :)
    @test AGP.yview(i) == view(y, :)

    @test AGP.nMinibatch(i) == N
    @test AGP.getρ(i) == 1.0
    @test AGP.isStochastic(i) == false
    @test AGP.MBIndices(i) == 1:N
end
