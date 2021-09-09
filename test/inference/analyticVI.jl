seed!(42)
L = 3
D = 10
N = 20
b = 5
x = [rand(D) for _ in 1:N]
y = rand(N)

@testset "AnalyticVI" begin
    i = AnalyticVI()
    @test AnalyticVI(; ϵ=0.0001f0) isa AnalyticVI{Float32,1}
    @test AnalyticSVI(10; ϵ=0.0001f0) isa AnalyticVI{Float32,1}
    @test repr(i) == "Analytic Variational Inference"
    xview = view(x, 1:N)
    yview = view(y, 1:N)
    i = AGP.tuple_inference(i, L, D, N, N, xview, yview)
    @test AGP.getρ(i) == 1.0
    @test AGP.isStochastic(i) == false
    @test i isa AnalyticVI{Float64,L}

    i = AGP.tuple_inference(AnalyticSVI(b), L, D, N, b, xview, yview)
    @test i isa AnalyticVI{Float64,L}
    @test AGP.getρ(i) == N / b
    @test AGP.isStochastic(i) == true
end
