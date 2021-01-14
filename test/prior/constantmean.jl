using AugmentedGaussianProcesses
using Test

N = 20
D = 3
x = rand()
X = rand(N, D)

@testset "ConstantMean" begin
    c = rand()
    μ₀ = ConstantMean(c, opt = Descent(1.0))
    @test μ₀ isa ConstantMean{Float64, Descent}
    @test repr(μ₀) == "Constant Mean Prior (c = $c)"
    @test μ₀(X) == c .* ones(N)
    @test μ₀(x) == c
    global g = Zygote.gradient(μ₀) do m
        norm(m(X))
    end
    AGP.update!(μ₀, first(g)[])
    @test μ₀.C[1] == (c + first(g)[].C[1])
end
