@testset "ConstantMean" begin
    N = 20
    x = rand()
    X = rand(N)
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
    AGP.update!(μ₀, st, only(g).C)
    @test μ₀.C[1] == (c + only(g).C[1])
end
