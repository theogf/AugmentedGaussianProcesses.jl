const M = 10
function generate_f(N, d, k; X=rand(N, d))
    f = rand(MvNormal(zeros(N), kernelmatrix(k, X; obsdim=1) + 1e-5I))
    return X, f
end
const VIs = ["AnalyticVI", "QuadratureVI", "MCIntegrationVI"]
const VIcodes = ["AVI", "QVI", "MCVI"]

function tests(model1, model2, X, f, y, problem)
    model1, state1 = train!(model1, X, y, 1)
    L = AGP.objective(model1, X, y)
    train!(model1, X, y, 5)
    @test testconv(model1, problem, X, f, y)
    @test all(proba_y(model1, X)[2] .> 0)
    train!(model2, X, y, 6)
    @test testconv(model2, problem, X, f, y)
    @test all(proba_y(model2, X)[2] .> 0)
end

function tests(model1::OnlineSVGP, model2, X, f, y, problem)
    state1 = nothing
    for (X_, y_) in eachbatch((X, y); obsdim=1, size=10)
        model1, state1 = train!(model1, X_, y_, state1; iterations=5)
        L = AGP.ELBO(model1, state1, X_, y_)
    end
    @test testconv(model1, problem, X, f, y)
    @test all(proba_y(model1, X)[2] .> 0)
    state2 = nothing
    for (X_, y_) in eachbatch((X, y); obsdim=1, size=10)
        model2, state2 = train!(model2, X_, y_, state2; iterations=5)
    end
    @test testconv(model2, problem, X, f, y)
    @test all(proba_y(model2, X)[2] .> 0)
end

function tests(
    model1::AbstractGPModel{T,<:BernoulliLikelihood{<:AGP.SVMLink}},
    model2,
    X,
    f,
    y,
    problem,
) where {T}
    model1, state1 = train!(model1, X, y, 1)
    @test_nowarn AGP.objective(model1, X, y)
    train!(model1, X, y, 5)
    @test testconv(model1, problem, X, f, y)
    @test all(proba_y(model1, X)[2] .> 0)
    model2, state2 = train!(model2, X, y, 6)
    @test testconv(model2, problem, X, f, y)
    @test all(proba_y(model2, X)[2] .> 0)
end

function tests_likelihood(
    l, # likelihood
    ltype, # likelihood types
    dict::Dict,
    floattypes,
    problem,
    n_latent,
    X,
    f,
    y,
    k,
)
    Z = inducingpoints(KmeansAlg(M), X)
    @testset "VGP" begin
        for floattype in floattypes
            dictvgp = dict["VGP"]
            for (name, code, inference) in
                zip(VIs, VIcodes, [AnalyticVI(), QuadratureVI(), MCIntegrationVI()])
                @testset "$name" begin
                    test_inference_VGP(
                        X,
                        y,
                        f,
                        k,
                        l,
                        ltype,
                        floattype,
                        n_latent,
                        problem,
                        inference;
                        valid=dictvgp[code],
                    )
                end
            end
        end # Loop on float types
    end # VGP
    @testset "OSVGP" begin
        dictosvgp = dict["OSVGP"]
        for floattype in floattypes
            @testset "AnalyticVI" begin
                if dictosvgp["AVI"]
                    model = OnlineSVGP(
                        k,
                        l,
                        AnalyticVI(),
                        AGP.InducingPoints.OIPS();
                        optimiser=false,
                        verbose=0,
                    )
                    @test eltype(model) == floattype
                    @test AGP.likelihood(model) isa ltype
                    @test AGP.inference(model) isa AnalyticVI
                    @test AGP.getf(model) isa NTuple{n_latent,AGP.OnlineVarLatent}
                    @test AGP.n_latent(model) == n_latent
                    model_opt = OnlineSVGP(
                        k,
                        l,
                        AnalyticVI(),
                        AGP.InducingPoints.OIPS();
                        optimiser=true,
                        verbose=0,
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws ErrorException OnlineSVGP(
                        k, l, AnalyticVI(), UniGrid(10); optimiser=false, verbose=0
                    )
                end
            end  # Analytic VI
            @testset "NumericalVI" begin
                if dictosvgp["QVI"]
                    model = OnlineSVGP(k, l, QuadratureVI(), UniGrid(10); optimiser=false)
                    @test eltype(model) == floattype
                    @test AGP.likelihood(model) isa ltype
                    @test AGP.inference(model) isa QuadratureVI
                    @test AGP.getf(model) isa NTuple{n_latent,AGP.OnlineVarLatent}
                    @test AGP.n_latent(model) == n_latent
                    model_opt = OnlineSVGP(
                        k, l, QuadratureVI(), UniGrid(10); optimiser=true, verbose=0
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws ErrorException OnlineSVGP(
                        k, l, QuadratureVI(), UniGrid(10); optimiser=false, verbose=0
                    )
                end
                if dictosvgp["MCVI"]
                    model = OnlineSVGP(
                        k, l, MCIntegrationVI(), UniGrid(10); optimiser=false
                    )
                    @test eltype(model) == floattype
                    @test AGP.likelihood(model) isa ltype
                    @test AGP.inference(model) isa MCIntegrationVI
                    @test AGP.getf(model) isa NTuple{n_latent,AGP.SparseVarLatent}
                    @test AGP.n_latent(model) == n_latent
                    model_opt = OnlineSVGP(
                        k, l, MCIntegrationVI(), UniGrid(10); optimiser=true, verbose=0
                    )
                    tests(model, model_opt, X, f, y, problem)
                else
                    @test_throws ErrorException OnlineSVGP(
                        k, l, MCIntegrationVI(), UniGrid(10); optimiser=false, verbose=0
                    )
                end
            end # Loop on Numerical VI
        end # Loop on float types
    end #Loop on Online SVGP
    @testset "SVGP" begin
        dictsvgp = dict["SVGP"]
        for floattype in floattypes
            for (name, code, inference, stoch_inference) in zip(
                VIs,
                VIcodes,
                [AnalyticVI(), QuadratureVI(), MCIntegrationVI()],
                [AnalyticSVI(10), QuadratureSVI(10), MCIntegrationSVI(10)],
            )
                @testset "$name" begin
                    test_inference_SVGP(
                        X,
                        y,
                        f,
                        k,
                        l,
                        ltype,
                        floattype,
                        Z,
                        n_latent,
                        problem,
                        inference,
                        stoch_inference;
                        valid=dictsvgp[code],
                    )
                end #
            end
        end # Loop on float types
    end # SVGP
    @testset "MCGP" begin
        dictmcgp = dict["MCGP"]
        for floattype in floattypes
            @testset "Gibbs Sampling" begin
                if dictmcgp["Gibbs"]
                    model = MCGP(X, y, k, l, GibbsSampling())
                    @test eltype(model) == floattype
                    @test AGP.likelihood(model) isa ltype
                    @test AGP.inference(model) isa GibbsSampling
                    @test AGP.getf(model) isa NTuple{n_latent,AGP.SampledLatent}
                    @test AGP.n_latent(model) == n_latent
                    samples = sample(model, 100; progress=false)
                    @test_broken samples2 = sample(model, 100; cat=true, progress=false)
                else
                    @test_throws ErrorException MCGP(X, y, k, l, GibbsSampling())
                end
                if dictmcgp["HMC"]
                    model = MCGP(X, y, k, l, HMCSampling())
                    @test eltype(model) == floattype
                    @test AGP.likelihood(model) isa ltype
                    @test AGP.inference(model) isa HMCSampling
                    @test AGP.getf(model) isa NTuple{n_latent,AGP.SampledLatent}
                    @test AGP.n_latent(model) == n_latent
                    samples = sample(model, 20; progress=false)
                    @test_broken samples2 = sample(model, 20; cat=true)
                else
                    @test_throws ErrorException MCGP(X, y, k, l, HMCSampling())
                end
            end # Gibbs Sampling
        end # Loop floattypes
    end # MCGP
end

function testconv(
    model::AbstractGPModel,
    problem_type::String,
    X::AbstractArray,
    f::AbstractArray,
    y::AbstractArray,
)
    μ, Σ = predict_f(model, X; cov=true, diag=false)
    μ, diagΣ = predict_f(model, X; cov=true, diag=true)
    y_pred = predict_y(model, X)
    py_pred = proba_y(model, X)
    if problem_type == "Regression"
        err = mean(abs.(y_pred - f))
        # @info "Regression Error" err
        return err < 15
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

function test_inference_VGP(
    X, y, f, k, l, ltype, floattype, n_latent, problem, inference; valid=false
)
    if valid
        model = VGP(X, y, k, l, inference; optimiser=false, verbose=0)
        @test eltype(model) == floattype
        @test AGP.likelihood(model) isa ltype
        @test AGP.inference(model) isa typeof(inference)
        @test AGP.getf(model) isa NTuple{n_latent,AGP.VarLatent}
        @test AGP.n_latent(model) == n_latent
        model_opt = VGP(X, y, k, l, inference; optimiser=true, verbose=0)
        tests(model, model_opt, X, f, y, problem)
    else
        @test_throws ErrorException VGP(X, y, k, l, inference; optimiser=false, verbose=0)
    end
end

function test_inference_SVGP(
    X,
    y,
    f,
    k,
    l,
    ltype,
    floattype,
    Z,
    n_latent,
    problem,
    inference,
    stoch_inference;
    valid=false,
)
    if valid
        model = SVGP(k, l, inference, Z; optimiser=false, verbose=0)
        @test eltype(model) == floattype
        @test AGP.likelihood(model) isa ltype
        @test AGP.inference(model) isa typeof(inference)
        @test AGP.n_latent(model) == n_latent
        model_opt = SVGP(k, l, inference, Z; optimiser=true, Zoptimiser=true, verbose=0)
        tests(model, model_opt, X, f, y, problem)
        @test AGP.getf(model) isa NTuple{n_latent,AGP.SparseVarLatent}

        model_svi = SVGP(k, l, stoch_inference, Z; optimiser=false, verbose=0)
        tests(model_svi, model, X, f, y, problem)
    else
        @test_throws ErrorException SVGP(k, l, inference, Z; optimiser=false, verbose=0)
    end
end
