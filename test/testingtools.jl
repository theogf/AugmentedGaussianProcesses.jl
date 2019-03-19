methods_implemented = Dict{String,Vector{String}}()
methods_implemented["GaussianLikelihood"] = []
methods_implemented["AugmentedStudentTLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]
methods_implemented["StudentTLikelihood"] = []# ["NumericalInference","StochasticNumericalInference"]
methods_implemented["AugmentedLogisticLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]#,"GibbsSampling"]
methods_implemented["BayesianSVM"] = ["AnalyticInference","StochasticAnalyticInference"]#,"GibbsSampling"]
methods_implemented["LogisticLikelihood"] = []# ["NumericalInference","StochasticNumericalInference"]
methods_implemented["AugmentedLogisticSoftMaxLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]#,"GibbsSampling"]
methods_implemented["LogisticSoftMaxLikelihood"] = ["NumericalInference","StochasticNumericalInference"]
methods_implemented["SoftMaxLikelihood"] = ["NumericalInference","StochasticNumericalInference"]

methods_implemented_VGP = deepcopy(methods_implemented)
methods_implemented_SVGP = deepcopy(methods_implemented)
methods_implemented_SVGP["GaussianLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]



isStochastic(inference::String) = (inference == "StochasticAnalyticInference" || inference == "StochasticNumericalInference")
stoch(s::Bool,inference::String) = s ? "Stochastic"*inference : inference
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
