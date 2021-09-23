@testset "NumericalVI" begin
    b = 10
    @test NumericalVI(:quad) isa QuadratureVI
    @test NumericalVI(:mc) isa MCIntegrationVI
    @test_throws ErrorException NumericalVI(:blah)
    @test NumericalSVI(b, :quad) isa QuadratureVI
    @test NumericalSVI(b, :mc) isa MCIntegrationVI
    @test_throws ErrorException NumericalSVI(b, :blah)
    @test repr(NumericalVI(:quad)) == "Numerical Inference by Quadrature"
    @test repr(NumericalVI(:mc)) == "Numerical Inference by Monte Carlo Integration"
end
