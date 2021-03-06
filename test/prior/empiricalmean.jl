@testset "EmpiricalMean" begin
    N = 20
    D = 3
    x = rand()
    X = rand(N, D)
    v = randn(N)
    μ₀ = EmpiricalMean(v, opt = Descent(1.0))
    @test μ₀ isa EmpiricalMean{Float64,Vector{Float64},Descent}
    @test repr("text/plain", μ₀) == "Empirical Mean Prior (length(c) = $N)"
    @test μ₀(X) == v
    @test_throws ErrorException μ₀(rand(N + 1, D)
    )
    g = Zygote.gradient(μ₀) do m
        return sum(m(X))
    end
    AGP.update!(μ₀, first(g))
    @test μ₀.C == v .+ first(g).C
end
