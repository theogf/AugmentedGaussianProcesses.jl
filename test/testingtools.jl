using LinearAlgebra, Distributions
using AugmentedGaussianProcesses
using MLDataUtils
using Test

M = 10
function generate_f(N, d, k; X = rand(N, d))
    f = rand(MvNormal(zeros(N), kernelmatrix(k, X, obsdim = 1) + 1e-5I))
    return X, f
end

function tests(model1, model2, X, f, y, problem)
    train!(model1, 1)
    L = AGP.objective(model1)
    @test L < 0
    train!(model1, 5)
    @test testconv(model1, problem, X, f, y)
    @test all(proba_y(model1, X)[2] .> 0)
    train!(model2, 6)
    @test testconv(model2, problem, X, f, y)
    @test all(proba_y(model2, X)[2] .> 0)
end

function tests(model1::OnlineSVGP, model2, X, f, y, problem)
    for (X_, y_) in eachbatch((X, y), obsdim = 1, size = 10)
        train!(model1, X_, y_, iterations = 1)
    end
    L = AGP.objective(model1)
    @test testconv(model1, problem, X, f, y)
    @test all(proba_y(model1, X)[2] .> 0)
    for (X_, y_) in eachbatch((X, y), obsdim = 1, size = 10)
        train!(model2, X_, y_, iterations = 1)
    end
    @test testconv(model2, problem, X, f, y)
    @test all(proba_y(model2, X)[2] .> 0)
end

function tests(
    model1::AbstractGP{T,<:BayesianSVM},
    model2,
    X,
    f,
    y,
    problem,
) where {T}
    train!(model1, 1)
    L = AGP.objective(model1)
    train!(model1, 5)
    @test testconv(model1, problem, X, f, y)
    @test all(proba_y(model1, X)[2] .> 0)
    train!(model2, 6)
    @test testconv(model2, problem, X, f, y)
    @test all(proba_y(model2, X)[2] .> 0)
end

function tests_likelihood(
    l::Likelihood,
    ltype::Type{<:Likelihood},
    dict::Dict,
    floattypes,
    problem,
    nLatent,
    X,
    f,
    y,
    k
)
    @testset "VGP" begin
        for floattype in floattypes
            dictvgp = dict["VGP"]
            @testset "AnalyticVI" begin
                if dictvgp["AVI"]
                    model = VGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticVI(),
                        optimiser = false,
                        verbose = 0,
                    )
                    @test model isa VGP{
                        floattype,
                        ltype{floattype},
                        AnalyticVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = VGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticVI(),
                        optimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws AssertionError VGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticVI(),
                        optimiser = false,
                        verbose = 0,
                    )
                end
            end  # Analytic VI
            @testset "NumericalVI" begin
                if dictvgp["QVI"]
                    model = VGP(X, y, k, l, QuadratureVI(), optimiser = false)
                    @test model isa VGP{
                        floattype,
                        ltype{floattype},
                        QuadratureVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = VGP(
                        X,
                        y,
                        k,
                        l,
                        QuadratureVI(),
                        optimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws AssertionError VGP(
                        X,
                        y,
                        k,
                        l,
                        QuadratureVI(),
                        optimiser = false,
                        verbose = 0,
                    )
                end
                if dictvgp["MCVI"]
                    model =
                        VGP(X, y, k, l, MCIntegrationVI(), optimiser = false)
                    @test model isa VGP{
                        floattype,
                        ltype{floattype},
                        MCIntegrationVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = VGP(
                        X,
                        y,
                        k,
                        l,
                        MCIntegrationVI(),
                        optimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws AssertionError VGP(
                        X,
                        y,
                        k,
                        l,
                        MCIntegrationVI(),
                        optimiser = false,
                        verbose = 0,
                    )
                end
            end # Loop on Numerical VI
        end # Loop on float types
    end # VGP
    @testset "OSVGP" begin
        dictosvgp = dict["OSVGP"]
        for floattype in floattypes
            dictvgp = dict["VGP"]
            @testset "AnalyticVI" begin
                if dictosvgp["AVI"]
                    model = OnlineSVGP(
                        k,
                        l,
                        AnalyticVI(),
                        OIPS(),
                        optimiser = false,
                        verbose = 0,
                    )
                    @test model isa OnlineSVGP{
                        floattype,
                        ltype{floattype},
                        AnalyticVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = OnlineSVGP(
                        k,
                        l,
                        AnalyticVI(),
                        OIPS(),
                        optimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws AssertionError OnlineSVGP(
                        k,
                        l,
                        AnalyticVI(),
                        OIPS(),
                        optimiser = false,
                        verbose = 0,
                    )
                end
            end  # Analytic VI
            @testset "NumericalVI" begin
                if dictosvgp["QVI"]
                    model = OnlineSVGP(
                        k,
                        l,
                        QuadratureVI(),
                        OIPS(),
                        optimiser = false,
                    )
                    @test model isa OnlineSVGP{
                        floattype,
                        ltype{floattype},
                        QuadratureVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = OnlineSVGP(
                        k,
                        l,
                        QuadratureVI(),
                        OIPS(),
                        optimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws AssertionError OnlineSVGP(
                        k,
                        l,
                        QuadratureVI(),
                        OIPS(),
                        optimiser = false,
                        verbose = 0,
                    )
                end
                if dictosvgp["MCVI"]
                    model = OnlineSVGP(
                        k,
                        l,
                        MCIntegrationVI(),
                        OIPS(),
                        optimiser = false,
                    )
                    @test model isa OnlineVGP{
                        floattype,
                        ltype{floattype},
                        MCIntegrationVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = OnlineSVGP(
                        k,
                        l,
                        MCIntegrationVI(),
                        OIPS(),
                        optimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws AssertionError OnlineSVGP(
                        k,
                        l,
                        MCIntegrationVI(),
                        OIPS(),
                        optimiser = false,
                        verbose = 0,
                    )
                end
            end # Loop on Numerical VI
        end # Loop on float types
    end #Loop on Online SVGP
    @testset "SVGP" begin
        dictsvgp = dict["SVGP"]
        for floattype in floattypes
            @testset "AnalyticVI" begin
                if dictsvgp["AVI"]
                    model = SVGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticVI(),
                        M,
                        optimiser = false,
                        verbose = 0,
                    )
                    @test model isa SVGP{
                        floattype,
                        ltype{floattype},
                        AnalyticVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = SVGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticVI(),
                        M,
                        optimiser = true,
                        Zoptimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)

                    model_svi = SVGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticSVI(10),
                        M,
                        optimiser = false,
                        verbose = 0,
                    )
                    tests(model_svi, model, X, f, y, problem)
                else
                    @test_throws AssertionError SVGP(
                        X,
                        y,
                        k,
                        l,
                        AnalyticVI(),
                        M,
                        optimiser = false,
                        verbose = 0,
                    )
                end
            end # Analytic VI
            @testset "NumericalVI" begin
                if dictsvgp["QVI"]
                    model =
                        SVGP(X, y, k, l, QuadratureVI(), M, optimiser = false)
                    @test model isa SVGP{
                        floattype,
                        ltype{floattype},
                        QuadratureVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = SVGP(
                        X,
                        y,
                        k,
                        l,
                        QuadratureVI(),
                        M,
                        optimiser = true,
                        Zoptimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)

                    model_svi = SVGP(
                        X,
                        y,
                        k,
                        l,
                        QuadratureSVI(10),
                        M,
                        optimiser = false,
                        verbose = 0,
                    )
                    tests(model_svi, model, X, f, y, problem)
                else
                    @test_throws AssertionError SVGP(
                        X,
                        y,
                        k,
                        l,
                        QuadratureVI(),
                        M,
                        optimiser = false,
                        verbose = 0,
                    )
                end
                if dictsvgp["MCVI"]
                    model = SVGP(
                        X,
                        y,
                        k,
                        l,
                        MCIntegrationVI(),
                        M,
                        optimiser = false,
                    )
                    @test model isa SVGP{
                        floattype,
                        ltype{floattype},
                        MCIntegrationVI{floattype,nLatent},
                        nLatent,
                    }
                    model_opt = SVGP(
                        X,
                        y,
                        k,
                        l,
                        MCIntegrationVI(),
                        M,
                        optimiser = true,
                        Zoptimiser = true,
                        verbose = 0,
                    )
                    tests(model, model_opt, X, f, y, problem)

                    model_svi = SVGP(
                        X,
                        y,
                        k,
                        l,
                        MCIntegrationSVI(10),
                        10,
                        optimiser = false,
                        verbose = 0,
                    )
                    tests(model_svi, model, X, f, y, problem)
                else
                    @test_throws AssertionError SVGP(
                        X,
                        y,
                        k,
                        l,
                        MCIntegrationVI(),
                        M,
                        optimiser = false,
                        verbose = 0,
                    )
                end
            end # Loop Numerical VI
        end # Loop on float types
    end # SVGP
    @testset "MCGP" begin
        dictmcgp = dict["MCGP"]
        for floattype in floattypes
            @testset "Gibbs Sampling" begin
                if dictmcgp["Gibbs"]
                    model = MCGP(X, y, k, l, GibbsSampling())
                    @test model isa MCGP{
                        floattype,
                        ltype{floattype},
                        GibbsSampling{floattype,nLatent},
                        nLatent,
                    }
                    samples = AGP.sample(model, 100)
                    @test_broken samples2 =
                        AGP.sample(model, 100, cat_samples = true)
                else
                    @test_throws AssertionError MCGP(
                        X,
                        y,
                        k,
                        l,
                        GibbsSampling(),
                    )
                end
                if dictmcgp["HMC"]
                    model = MCGP(X, y, k, l, HMCSampling())
                    @test model isa MCGP{
                        floattype,
                        ltype{floattype},
                        HMCSampling{floattype,nLatent},
                        nLatent,
                    }
                    samples = AGP.sample(model, 20)
                    @test_broken samples2 =
                        AGP.sample(model, 20, cat_samples = true)
                else
                    @test_throws AssertionError MCGP(X, y, k, l, HMCSampling())
                end
            end # Gibbs Sampling
        end # Loop floattypes
    end # MCGP
end

function testconv(
    model::AbstractGP,
    problem_type::String,
    X::AbstractArray,
    f::AbstractArray,
    y::AbstractArray,
)
    μ, Σ = predict_f(model, X, covf = true, fullcov = true)
    μ, diagΣ = predict_f(model, X, covf = true, fullcov = false)
    y_pred = predict_y(model, X)
    py_pred = proba_y(model, X)
    if problem_type == "Regression"
        err = mean(abs.(y_pred - f))
        # @info "Regression Error" err
        return err < 1.5
    elseif problem_type == "Classification"
        err = mean(y_pred .!= y)
        # @info "Classification Error" err
        return err < 0.5
    elseif problem_type == "MultiClass"
        err = mean(y_pred .!= y)
        # @info "Multiclass Error" err
        return err < 0.9
    elseif problem_type == "Poisson" || problem_type == "NegBinomial"
        err = mean(abs.(y_pred - y))
        # @info "Event Error" err
        return err < 20.0
    else
        throw(ErrorException("Wrong problem description $problem_type"))
    end
end
