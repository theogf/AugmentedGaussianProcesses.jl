@testset "QuadratureVI" begin
    C = 10.0
    i = QuadratureVI(; clipping=C)
    l = GaussianLikelihood()
    @test AGP.grad_quad(l, 200.0, 0.0, 1.0, i) == (C, -C)
end
