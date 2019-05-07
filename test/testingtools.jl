methods_implemented = Dict{String,Vector{String}}()
methods_implemented["GaussianLikelihood"] = []
methods_implemented["StudentTLikelihood"] = ["AnalyticVI","AnalyticSVI"] # ["QuadratureVI","QuadratureSVI"]
methods_implemented["LaplaceLikelihood"] = ["AnalyticVI","AnalyticSVI"]
methods_implemented["LogisticLikelihood"] = ["AnalyticVI","AnalyticSVI"]# ["NumericalVI","NumericalSVI"]
methods_implemented["BayesianSVM"] = ["AnalyticVI","AnalyticSVI"]
methods_implemented["LogisticSoftMaxLikelihood"] = ["AnalyticVI","AnalyticSVI"]# "NumericalVI","NumericalSVI"]
methods_implemented["SoftMaxLikelihood"] = ["QuadratureVI","QuadratureSVI"]
methods_implemented["PoissonLikelihood"] = ["AnalyticVI","AnalyticSVI"]

methods_implemented_VGP = deepcopy(methods_implemented)
push!(methods_implemented_VGP["StudentTLikelihood"],"GibbsSampling")
push!(methods_implemented_VGP["LogisticLikelihood"],"GibbsSampling")
push!(methods_implemented_VGP["LogisticSoftMaxLikelihood"],"GibbsSampling")
methods_implemented_SVGP = deepcopy(methods_implemented)
methods_implemented_SVGP["GaussianLikelihood"] = ["AnalyticVI","AnalyticSVI"]


stoch(s::Bool,inference::String) = inference != "GibbsSampling" ? (s ? inference[1:end-2]*"SVI" : inference) : inference
addiargument(s::Bool,inference::String) = inference == "GibbsSampling" ? "nBurnin=0" : (s ? "b" : "")
addlargument(likelihood::String) = begin
    if (likelihood == "StudentTLikelihood")
        return "ν"
    else
        return ""
    end
end

function testconv(model::AbstractGP,problem_type::String,X::AbstractArray,y::AbstractArray)
    μ,Σ = predict_f(model,X,covf=true)
    y_pred = predict_y(model,X)
    py_pred = proba_y(model,X)
    if problem_type == "Regression"
        @show err = mean(abs.(y_pred-y))
        return err < 0.5
    elseif problem_type == "Classification"
        @show err = mean(y_pred.!=y)
        return err < 0.5
    elseif problem_type == "MultiClass"
        @show err = mean(y_pred.!=y)
        return err < 0.5
    elseif problem_type == "Event"
        @show err = mean(abs.(y_pred-y))
        return err < 1.0
    end
end
