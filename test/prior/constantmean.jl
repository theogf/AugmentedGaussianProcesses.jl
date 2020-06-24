using AugmentedGaussianProcesses
using Test

N = 20
D = 3
x = rand()
X = rand(N, D)

@testset "ConstantMean" begin
    c = rand()
    μ₀ = ConstantMean(c, opt = Descent(1.0))
    # @test μ₀(x) == c
    @test μ₀ isa ConstantMean{Float64, Descent}
    @test repr(μ₀) == "Constant Mean Prior (c = $c)"
    @test μ₀(X) == c.*ones(N)
    @test μ₀(x) == c
    AGP.update!(μ₀,[1.0],X)
    @test μ₀.C[] == (c + 1.0)
end
