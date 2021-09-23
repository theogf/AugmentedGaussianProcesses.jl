@testset "AnalyticVI" begin
    seed!(42)
    L = 3
    D = 10
    N = 20
    b = 5
    i = AnalyticVI()
    @test i isa AnalyticVI{Float64}
    @test AnalyticVI(; ϵ=0.0001f0) isa AnalyticVI{Float32}
    @test AnalyticSVI(10; ϵ=0.0001f0) isa AnalyticVI{Float32}
    @test repr(i) == "Analytic Variational Inference"
    @test AGP.ρ(i) == 1.0
    @test AGP.is_stochastic(i) == false

    i = AnalyticSVI(b)
    @test i isa AnalyticVI{Float64}
    AGP.set_ρ!(i, N / b)
    @test AGP.ρ(i) == N / b
    @test AGP.is_stochastic(i) == true
end
