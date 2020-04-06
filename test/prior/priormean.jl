using AugmentedGaussianProcesses
using Test

N = 20
D = 3
x = rand()
X = rand(N, D)

@testset "Automatic Convert" begin
        x = rand()
        v = rand(3)
        @test convert(PriorMean, x) isa ConstantMean
        @test convert(PriorMean, v) isa EmpiricalMean
end
