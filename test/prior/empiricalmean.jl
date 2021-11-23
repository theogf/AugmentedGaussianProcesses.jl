@testset "EmpiricalMean" begin
    N = 20
    D = 3
    x = rand()
    X = rand(N)
    v = randn(N)
    μ₀ = EmpiricalMean(v; opt=Descent(1.0))
    st = AGP.init_priormean_state((;), μ₀)
    @test μ₀ isa EmpiricalMean{Float64,Vector{Float64},<:Descent}
    @test repr("text/plain", μ₀) == "Empirical Mean Prior (length(c) = $N)"
    @test μ₀(X) == v
    @test_throws ErrorException μ₀(rand(N + 1))
    g = Zygote.gradient(μ₀) do m
        return sum(abs2, m(X))
    end
    AGP.update!(μ₀, st, only(g))
    @test μ₀.C == v .+ only(g).C
end
