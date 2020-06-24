seed!(42)
L = 3
D = 10
N = 20
b = 5
x = rand(N, D)
y = rand(N)


@testset "AnalyticVI" begin
    i = AnalyticVI()
    @test AnalyticVI(ϵ = 0.0001f0) isa AnalyticVI{Float32,1}
    @test AnalyticSVI(10, ϵ = 0.0001f0) isa AnalyticVI{Float32,1}
    @test repr(i) == "Analytic Variational Inference"
    i = AGP.tuple_inference(i, L, D, N, N)
    @test AGP.getρ(i) == 1.0
    @test AGP.isStochastic(i) == false
    @test i isa AnalyticVI{Float64,L}

    i = AGP.tuple_inference(AnalyticSVI(b), L, D, N, b)
    @test i isa AnalyticVI{Float64,L}
    @test AGP.getρ(i) == N / b
    @test AGP.isStochastic(i) == true
end
