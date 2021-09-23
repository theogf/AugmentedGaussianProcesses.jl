@testset "event" begin
    seed!(42)
    y = rand(1:10, 10)
    l = PoissonLikelihood()
    @test AGP.treat_labels!(y, l) == y
    @test_throws ErrorException AGP.treat_labels!(rand(10), l)
end
