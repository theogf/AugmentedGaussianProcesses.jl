using AugmentedGaussianProcesses
using Test

N = 20
D = 3
x = rand()
X = rand(N,D)

@testset "ZeroMean" begin
        μ₀ = ZeroMean()
        @test ZeroMean() isa ZeroMean{Float64}
        @test_nowarn println(μ₀)
        @test isnothing(AGP.update!(μ₀,[x],X))
        @test all(μ₀(X) .== zeros(N))
        @test μ₀(x) == zero(x)
end
