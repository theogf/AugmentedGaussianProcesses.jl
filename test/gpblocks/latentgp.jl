@testset "LatentGP" begin
    
    kernel = SqExponentialKernel()
    T = Float64
    D = 5
    mean = ZeroMean()
    opt = ADAM()
    X = rand(1, D)
    K = kernelmatrix(kernel, X, obsdim = 1)
    @testset "LatentGP" begin
        m = LatentGP(T, D, kernel, mean, opt)
        @test m isa LatentGP
        @test m isa AbstractLatentGP
        @test prior(m) isa GPPrior
        @test posterior(m) isa VarPosterior
        @test kernel(m) == kernel
        @test pr_mean(m) == mean
        @test pr_mean(m, X) == zeros(D)
        @test pr_cov(m) == PDMat(Matrix{T}(I(D)))
        pr_cov!(m, K)
        @test pr_cov(m) == K
        @test dim(m) == D
    end
end