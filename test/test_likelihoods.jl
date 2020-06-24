testdir = "likelihood"

@testset "Likelihoods Base Functions" begin
    @testset "Generic functions" begin
        include(joinpath(testdir, "likelihood.jl"))
    end
    @testset "Regression" begin
        include(joinpath(testdir, "regression.jl"))
    end
    @testset "Classification" begin
        include(joinpath(testdir, "classification.jl"))
    end
    @testset "Multiclass" begin
        include(joinpath(testdir, "multiclass.jl"))
    end
    @testset "Event" begin
        include(joinpath(testdir, "event.jl"))
    end
end
@testset "Likelihoods" begin
    ## Regression
    @testset "Gaussian" begin
        include(joinpath(testdir,"gaussian.jl"))
    end
    @testset "StudentT" begin
        include(joinpath(testdir,"studentt.jl"))
    end
    @testset "Laplace" begin
        include(joinpath(testdir,"laplace.jl"))
    end
    @testset "Heteroscedastic" begin
        include(joinpath(testdir,"heteroscedastic.jl"))
    end
    ## Classification
    @testset "Bayesian SVM" begin
        include(joinpath(testdir,"bayesiansvm.jl"))
    end
    @testset "Logistic" begin
        include(joinpath(testdir,"logistic.jl"))
    end
    @testset "Logistic Softmax" begin
        include(joinpath(testdir,"logisticsoftmax.jl"))
    end
    @testset "Softmax" begin
        include(joinpath(testdir,"softmax.jl"))
    end
    @testset "Poisson" begin
        include(joinpath(testdir,"poisson.jl"))
    end
    ## Event likelihoods
    @testset "Negative Binomial" begin
        include(joinpath(testdir,"negativebinomial.jl"))
    end
end
