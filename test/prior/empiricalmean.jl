using AugmentedGaussianProcesses
using Test

N = 20
D = 3
x = rand()
X = rand(N,D)

@testset "EmpiricalMean" begin
        v = randn(N)
        μ₀ = EmpiricalMean(v,opt=Descent(1.0))
        @test μ₀ isa EmpiricalMean{Float64, Vector{Float64}, Descent}
        @test_nowarn println(μ₀)
        @test μ₀(X) == v
        @test_throws AssertionError μ₀(rand(N+1,D))
        AGP.update!(μ₀,ones(N),X)
        @test μ₀.C == v .+ 1.0
end
