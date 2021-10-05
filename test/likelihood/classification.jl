@testset "classification" begin
    seed!(42)
    l = LogisticLikelihood()
    y = rand([1, -1], 10)
    @test AGP.treat_labels!(y, l) == y
    y = rand([0, 1], 10)
    @test AGP.treat_labels!(y, l) == 2 * (y .- 0.5)
    y = randn(10)
    @test_throws InexactError AGP.treat_labels!(y, l)
    y = rand([2, 3], 10)
    @test_throws ArgumentError AGP.treat_labels!(y, l)
    y = randn(10)
    @test predict_y(l, y) == (y .> 0)
    @test predict_y(l, (y,)) == (y .> 0)
end
