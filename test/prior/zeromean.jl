@testset "ZeroMean" begin
    N = 20
    D = 3
    x = rand()
    X = rand(N)

    μ₀ = ZeroMean()
    @test ZeroMean() isa ZeroMean{Float64}
    @test repr("text/plain", μ₀) == "Zero Mean Prior"
    @test isnothing(AGP.update!(μ₀, (;), []))
    @test all(μ₀(X) .== zeros(N))
    @test μ₀(x) == zero(x)
end
