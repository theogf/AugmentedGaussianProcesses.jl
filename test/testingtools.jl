methods_implemented = Dict{String,Vector{String}}()
methods_implemented["GaussianLikelihood"] = []
methods_implemented["StudentTLikelihood"] = ["AnalyticVI","AnalyticSVI","QuadratureVI","QuadratureSVI"] # ["QuadratureVI","QuadratureSVI"]
methods_implemented["LaplaceLikelihood"] = ["AnalyticVI","AnalyticSVI","QuadratureVI","QuadratureSVI"]
methods_implemented["HeteroscedasticLikelihood"] = []
methods_implemented["LogisticLikelihood"] = ["AnalyticVI","AnalyticSVI","QuadratureVI","QuadratureSVI"]
methods_implemented["BayesianSVM"] = ["AnalyticVI","AnalyticSVI"]
methods_implemented["LogisticSoftMaxLikelihood"] = ["AnalyticVI","AnalyticSVI"]# "NumericalVI","NumericalSVI"]
methods_implemented["SoftMaxLikelihood"] = ["MCIntegrationVI","MCIntegrationSVI"]
methods_implemented["PoissonLikelihood"] = ["AnalyticVI","AnalyticSVI"]
methods_implemented["NegBinomialLikelihood"] = ["AnalyticVI","AnalyticSVI"]


methods_implemented_VGP = deepcopy(methods_implemented)
push!(methods_implemented_VGP["StudentTLikelihood"],"GibbsSampling")
push!(methods_implemented_VGP["LaplaceLikelihood"],"GibbsSampling")
push!(methods_implemented_VGP["LogisticLikelihood"],"GibbsSampling")
push!(methods_implemented_VGP["LogisticSoftMaxLikelihood"],"GibbsSampling")
push!(methods_implemented_VGP["HeteroscedasticLikelihood"],"AnalyticVI")
methods_implemented_SVGP = deepcopy(methods_implemented)
methods_implemented_SVGP["GaussianLikelihood"] = ["AnalyticVI","AnalyticSVI"]

function nlatent(l::String)
    if l == "LogisticSoftMaxLikelihood" || l == "SoftMaxLikelihood"
        "$(n_class)"
    elseif l== "HeteroscedasticLikelihood"
        "2"
    else
        "1"
    end
end
stoch(s::Bool,inference::String) = inference != "GibbsSampling" ? (s ? inference[1:end-2]*"SVI" : inference) : inference
addiargument(s::Bool,inference::String) = inference == "GibbsSampling" ? "nBurnin=0" : (s ? "b" : "")
addlargument(likelihood::String) = begin
    if (likelihood == "StudentTLikelihood")
        return "ν"
    elseif (likelihood == "NegBinomialLikelihood")
        return "r"
    else
        return ""
    end
end

function testconv(model::AbstractGP,problem_type::String,X::AbstractArray,y::AbstractArray)
    μ,Σ = predict_f(model,X,covf=true)
    y_pred = predict_y(model,X)
    py_pred,sigy_pred = proba_y(model,X)
    if problem_type == "Regression"
        err=mean(abs.(y_pred-y))
        @info "Regression Error" err
        return err < 1.5
    elseif problem_type == "Classification"
        err=mean(y_pred.!=y)
        @info "Classification Error" err
        return err < 0.5
    elseif problem_type == "MultiClass"
        err = mean(y_pred.!=y)
        @info "Multiclass Error" err
        return err < 0.5
    elseif problem_type == "Poisson" || problem_type == "NegBinomial"
        err = mean(abs.(y_pred-y))
        @info "Event Error" err
        return err < 20.0
    end
end
