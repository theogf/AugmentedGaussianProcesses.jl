@testset "AffineMean" begin
    N = 20
    D = 3
    x = rand()
    X = [rand(D) for _ in 1:N]
    b = randn()
    w = randn(D)
    μ₀ = AffineMean(w, b; opt=Descent(1.0))
    st = AGP.init_priormean_state((;), μ₀)
    @test μ₀ isa AffineMean{Float64,Vector{Float64},<:Descent}
    @test_nowarn AffineMean(3)(X)
    @test repr("text/plain", μ₀) == "Affine Mean Prior (size(w) = $D, b = $b)"
    @test μ₀(X) == dot.(X, Ref(w)) .+ b
    @test_throws DimensionMismatch AffineMean(4)(X)
    g = Zygote.gradient(μ₀) do m
        sum(m(X))
    end
    AGP.update!(μ₀, st, only(g))
    @test all(μ₀.w .== (w + only(g).w))
    @test first(μ₀.b) == b + only(g).b[1]
end
