methods_implemented = Dict{String,Vector{String}}()
methods_implemented["GaussianLikelihood"] = []
methods_implemented["AugmentedStudentTLikelihood"] = ["AnalyticVI","AnalyticSVI"]
methods_implemented["StudentTLikelihood"] = []# ["NumericalVI","NumericalSVI"]
methods_implemented["AugmentedLogisticLikelihood"] = ["AnalyticVI","AnalyticSVI"]#,"GibbsSampling"]
methods_implemented["BayesianSVM"] = ["AnalyticVI","AnalyticSVI"]#,"GibbsSampling"]
methods_implemented["LogisticLikelihood"] = []# ["NumericalVI","NumericalSVI"]
methods_implemented["AugmentedLogisticSoftMaxLikelihood"] = ["AnalyticVI","AnalyticSVI"]#,"GibbsSampling"]
methods_implemented["LogisticSoftMaxLikelihood"] = ["NumericalVI","NumericalSVI"]
methods_implemented["SoftMaxLikelihood"] = ["NumericalVI","NumericalSVI"]

methods_implemented_VGP = deepcopy(methods_implemented)
methods_implemented_SVGP = deepcopy(methods_implemented)
methods_implemented_SVGP["GaussianLikelihood"] = ["AnalyticVI","AnalyticSVI"]



isStochastic(inference::String) = (inference == "StochasticAnalyticVI" || inference == "StochasticNumericalVI")
stoch(s::Bool,inference::String) = s ? inference[1:end-2]*"SVI" : inference
addlargument(likelihood::String) = begin
    if (likelihood == "StudentTLikelihood" || likelihood == "AugmentedStudentTLikelihood")
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
        err = mean(abs2.(y_pred-y))
        return err < 0.2
    elseif problem_type == "Classification"
        err = mean(y_pred.!=y)
        return err < 0.2
    elseif problem_type == "MultiClass"
        err = mean(y_pred.!=y)
        return err < 0.5
    end
end
