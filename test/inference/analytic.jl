@testset "Analytic" begin
    seed!(42)
    D = 10
    N = 20
    b = 5
    x = [rand(D) for i in 1:D]
    y = rand(N)
    i = Analytic()
    @test Analytic(; ϵ=0.0001f0) isa Analytic{Float32}
    @test repr(i) == "Analytic Inference"

    @test AGP.batchsize(i) == 0
    AGP.set_batchsize!(i, N)
    @test AGP.batchsize(i) == N
    @test AGP.ρ(i) == 1.0
    @test AGP.is_stochastic(i) == false
end
