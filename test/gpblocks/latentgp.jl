@testset "AbstractLatent" begin
    kernel = SqExponentialKernel()
    T = Float64
    D = 5
    m0 = ZeroMean()
    opt = ADAM()
    X = rand(1, D)
    K = kernelmatrix(kernel, X; obsdim=1)
    @testset "LatentGP" begin
        m = AGP.LatentGP(T, D, kernel, m0, opt)
        @test m isa AGP.LatentGP
        @test AGP.prior(m) isa AGP.GPPrior
        @test AGP.posterior(m) isa AGP.Posterior
        # Get functions
        @test AGP.kernel(m) == kernel
        @test AGP.pr_mean(m) == m0
        @test AGP.pr_mean(m, vec(X)) == zeros(D)
        @test AGP.pr_cov(m) == PDMat(Matrix{T}(I(D)))
        @test mean(m) == zeros(D)
        @test cov(m) == PDMat(Matrix{T}(I(D)))
        AGP.pr_cov!(m, PDMat(K))
        @test AGP.pr_cov(m) == K
        @test dim(m) == D
    end
    @testset "VarLatent" begin
        m = AGP.VarLatent(T, D, kernel, m0, opt)
        @test m isa AGP.VarLatent
        @test AGP.prior(m) isa AGP.GPPrior
        @test AGP.posterior(m) isa AGP.VarPosterior
        # Get functions
        @test AGP.kernel(m) == kernel
        @test AGP.pr_mean(m) == m0
        @test AGP.pr_mean(m, vec(X)) == zeros(D)
        @test AGP.pr_cov(m) == PDMat(Matrix{T}(I(D)))
        @test mean(m) == zeros(D)
        @test cov(m) == PDMat(Matrix{T}(I(D)))
        AGP.pr_cov!(m, PDMat(K))
        @test AGP.pr_cov(m) == K
        @test dim(m) == D
    end
end
