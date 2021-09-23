@testset "ConstantMean" begin
    N = 20
    D = 3
    x = rand()
    X = rand(N, D)
    c = rand()
    μ₀ = ConstantMean(c; opt=Descent(1.0))
    st = AGP.init_priormean_state((;), μ₀)
    @test μ₀ isa ConstantMean{Float64,<:Descent}
    @test repr("text/plain", μ₀) == "Constant Mean Prior (c = $c)"
    @test μ₀(X) == c .* ones(N)
    @test μ₀(x) == c
    g = Zygote.gradient(μ₀) do m
        sum(m(X))
    end
    AGP.update!(μ₀, st, Vector(first(g)[].C))
    @test μ₀.C[1] == (c + first(g)[].C[1])
end
