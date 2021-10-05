@testset "event" begin
    seed!(42)
    y = rand(1:10, 10)
    l = PoissonLikelihood(3.0)
    @test AGP.treat_labels!(y, l) == y
    @test_throws ErrorException AGP.treat_labels!(rand(10), l)
end
