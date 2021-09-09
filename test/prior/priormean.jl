@testset "Automatic Convert" begin
    x = rand()
    v = rand(3)
    @test convert(PriorMean, x) isa ConstantMean
    @test convert(PriorMean, v) isa EmpiricalMean
end
