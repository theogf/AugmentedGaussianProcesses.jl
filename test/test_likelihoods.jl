using AugmentedGaussianProcesses;
using Test
using Distributions

@testset "Likelihoods Base Functions" begin
    @testset "Generic functions" begin
        struct NewLikelihood{T} <: AGP.Likelihood{T} end
        @test_throws ErrorException AGP.pdf(NewLikelihood{Float64}(), rand(), rand())
        @test length(NewLikelihood{Float64}()) == 1
    end
    @testset "Regression" begin
        y = randn(10)
        l = GaussianLikelihood()
        @test AGP.treat_labels!(y, l) == (y, 1, l)
        @test predict_y(l, y) == y
        @test predict_y(l, [y]) == y
    end
    @testset "Classification" begin
        l = LogisticLikelihood()
        y = rand([1, -1], 10)
        @test AGP.treat_labels!(y, l) == (y, 1, l)
        y = rand([0, 1], 10)
        @test AGP.treat_labels!(y, l) == ((y .- 0.5) * 2, 1, l)
        y = randn(10)
        @test_throws AssertionError AGP.treat_labels!(y, l)
        y = rand([2,3], 10)
        @test_throws AssertionError AGP.treat_labels!(y, l)
        y = randn(10)
        @test predict_y(l, y) == sign.(y)
        @test predict_y(l, [y]) == sign.(y)
    end
    @testset "Multiclass" begin
        y = [1, 2, 3, 1, 1, 2, 3]
        l = LogisticSoftMaxLikelihood(3)
        AGP.create_mapping!(l, y)
        @test sort(l.class_mapping) == [1, 2, 3]
        @test l.ind_mapping == Dict(1 => 1, 2 => 2, 3 => 3)
        AGP.create_one_hot!(l,y[1:3])
        @test l.Y == [BitArray([1, 0, 0]),BitArray([0, 1, 0]),BitArray([0, 0, 1])]
        @test l.y_class == [1, 2, 3]

        l = LogisticSoftMaxLikelihood(2)
        @test_throws ErrorException AGP.create_mapping!(l,y)

        y = [1, 2, 1, 1]
        l = LogisticSoftMaxLikelihood(3)
        AGP.create_mapping!(l,y)
        @test l.class_mapping == [1,2,3]
        @test l.ind_mapping == Dict(1 => 1, 2 => 2, 3 => 3)
        AGP.create_one_hot!(l,y)
        @test l.Y == [BitArray([1,0,1,1]),BitArray([0,1,0,0]),BitArray([0,0,0,0])]
        @test l.y_class == [1,2,1,1]
        @test AGP.num_latent(l) == 3

        y = ["b","a","c","a","a"]
        l = LogisticSoftMaxLikelihood(3)
        AGP.create_mapping!(l, y)
        @test l.class_mapping == ["b", "a", "c"]
        @test l.ind_mapping == Dict("b" => 1, "a" => 2, "c" => 3)
        AGP.create_one_hot!(l,y)
        @test l.Y == [BitArray([1,0,0,0,0]),BitArray([0,1,0,1,1]),BitArray([0,0,1,0,0])]
        @test l.y_class == [1, 2, 3, 2, 2]

        l = LogisticSoftMaxLikelihood(["a", "b", "c"])
        @test l.ind_mapping == Dict("a" => 1, "b" => 2, "c" => 3)
        AGP.create_one_hot!(l,y)
        @test l.Y == [BitArray([0,1,0,1,1]), BitArray([1,0,0,0,0]), BitArray([0,0,1,0,0])]
        @test l.y_class == [2,1,3,1,1]

        l = LogisticSoftMaxLikelihood(3)
        AGP.treat_labels!(y,l)
        @test l.class_mapping == ["b", "a", "c"]
        @test l.ind_mapping == Dict("b" => 1, "a" => 2, "c" => 3)
        @test l.Y == [BitArray([1, 0, 0, 0, 0]),BitArray([0, 1, 0, 1, 1]),BitArray([0, 0, 1, 0, 0])]
        @test l.y_class == [1, 2, 3, 2, 2]
    end
    @testset "Event" begin
        y = rand(1:10, 10)
        l = PoissonLikelihood()
        @test AGP.treat_labels!(y, l) == (y, 1, l)
        @test_throws AssertionError AGP.treat_labels!(rand(10), l)
    end
end
@testset "Likelihoods" begin
    @testset "Gaussian" begin
        include("likelihoods/test_Gaussian.jl")
    end
    @testset "StudentT" begin
        include("likelihoods/test_StudentT.jl")
    end
    @testset "Laplace" begin
        include("likelihoods/test_Laplace.jl")
    end
    @testset "Heteroscedastic" begin
        include("likelihoods/test_Heteroscedastic.jl")
    end
    @testset "Bayesian SVM" begin
        include("likelihoods/test_BayesianSVM.jl")
    end
    @testset "Logistic" begin
        include("likelihoods/test_Logistic.jl")
    end
    @testset "Logistic Softmax" begin
        include("likelihoods/test_LogisticSoftMax.jl")
    end
    @testset "Softmax" begin
        include("likelihoods/test_SoftMax.jl")
    end
    @testset "Poisson" begin
        include("likelihoods/test_Poisson.jl")
    end
    @testset "Negative Binomial" begin
        include("likelihoods/test_NegativeBinomial.jl")
    end
end
