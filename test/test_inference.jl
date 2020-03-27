using Test
using AugmentedGaussianProcesses

L = 3
D = 10
N = 20
b = 5

@testset "Inference methods" begin
    @testset "Analytic" begin
        i = Analytic()
        @test_nowarn show(i)
        @test Analytic(ϵ=0.0001f0) isa Analytic{Float32}
        @test i.ρ == 1.0
        @test i.Stochastic == false
    end
    @testset "AnalyticVI" begin
        i = AnalyticVI()
        @test_nowarn show(i)
        @test AnalyticVI(ϵ = 0.0001f0) isa AnalyticVI{Float32,1}
        @test AnalyticSVI(10, ϵ = 0.0001f0) isa AnalyticVI{Float32,1}
        @test i.ρ == [1.0]
        @test i.Stochastic == false
        i = AGP.tuple_inference(i, L, D, N, N)
        @test i isa AnalyticVI{Float64,L}

        i = AGP.tuple_inference(AnalyticSVI(b), L, D, N, b)
        @test i isa AnalyticVI{Float64, L}
        @test i.ρ == fill(N/b, L)
        @test i.Stochastic == true
    end
    @testset "NumericalVI" begin
        @test NumericalVI(:quad) isa QuadratureVI
        @test NumericalVI(:mc) isa MCIntegrationVI
        @test_throws ErrorException NumericalVI(:blah)
        @test NumericalSVI(b,:quad) isa QuadratureVI
        @test NumericalSVI(b,:mc) isa MCIntegrationVI
        @test_throws ErrorException NumericalSVI(b,:blah)
        @test_nowarn show(NumericalVI(:quad))
    end
    @testset "QuadratureVI" begin
        C = 10.0
        i = AGP.tuple_inference(QuadratureVI(clipping=C),1,D,N,N)
        l = GaussianLikelihood()
        @test AGP.grad_quad(l,200.0,0.0,1.0,i) == (C,-C)
    end
    @testset "GibbsSampling" begin
        i = GibbsSampling()
        @test_nowarn show(i)
    end
end
