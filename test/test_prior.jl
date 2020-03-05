using AugmentedGaussianProcesses
using Test

N = 20
D = 3
x = rand()
X = rand(N,D)

@testset "Prior Mean" begin
        @testset  "Automatic Convert" begin
                x = rand()
                v = rand(3)
                @test convert(PriorMean,x) isa ConstantMean
                @test convert(PriorMean,v) isa EmpiricalMean
        end
        @testset "ZeroMean" begin
                μ₀ = ZeroMean()
                @test ZeroMean() isa ZeroMean{Float64}
                @test isnothing(AGP.update!(μ₀,[x],X))
                @test all(μ₀(X) .== zeros(N))
                @test μ₀(x) == zero(x)
        end
        @testset "ConstantMean" begin
                c = rand()
                μ₀ = ConstantMean(c,opt=Descent(1.0))
                @test μ₀(x) == c
                @test μ₀(X) == c.*ones(N)
                AGP.update!(μ₀,[1.0],X)
                @test μ₀.C[] == (c + 1.0)
        end
        @testset "EmpiricalMean" begin
                v = randn(N)
                μ₀ = EmpiricalMean(v,opt=Descent(1.0))
                @test μ₀(X) == v
                @test_throws AssertionError μ₀(rand(N+1,D))
                AGP.update!(μ₀,ones(N),X)
                @test μ₀.C == v .+ 1.0
        end
        @testset "AffineMean" begin
                b = randn()
                w = randn(D)
                μ₀ = AffineMean(w,b,opt=Descent(1.0))
                @test μ₀(X) == X*w .+ b
                @test_nowarn AffineMean(3)(X)
                @test_throws AssertionError AffineMean(4)(X)
                AGP.update!(μ₀,ones(N),X)
                @test all(μ₀.w .== (w + X'*ones(N)))
                @test μ₀.b[] == b + N
        end
end
