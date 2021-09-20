@testset "MCIntegrationVI" begin
    C = 10.0
    i = MCIntegrationVI(; clipping=C)
    @test repr(i) == "Numerical Inference by Monte Carlo Integration"
    l = GaussianLikelihood()
end
