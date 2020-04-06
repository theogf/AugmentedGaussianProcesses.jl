using Test
using AugmentedGaussianProcesses

L = 3
D = 10
N = 20
b = 5

@testset "Inference methods" begin
    @testset "QuadratureVI" begin
        C = 10.0
        i = AGP.tuple_inference(QuadratureVI(clipping=C),1,D,N,N)
        l = GaussianLikelihood()
        @test AGP.grad_quad(l,200.0,0.0,1.0,i) == (C,-C)
    end

end
