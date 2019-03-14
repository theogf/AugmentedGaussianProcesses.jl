methods_implemented = Dict{String,Vector{String}}()
methods_implemented["GaussianLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]
methods_implemented["AugmentedStudentTLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]
methods_implemented["StudentTLikelihood"] = []# ["NumericalInference","StochasticNumericalInference"]
methods_implemented["AugmentedLogisticLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]#,"GibbsSampling"]
methods_implemented["LogisticLikelihood"] = []# ["NumericalInference","StochasticNumericalInference"]
methods_implemented["AugmentedLogisticSoftMaxLikelihood"] = ["AnalyticInference","StochasticAnalyticInference"]#,"GibbsSampling"]
methods_implemented["LogisticSoftMaxLikelihood"] = ["NumericalInference","StochasticNumericalInference"]
methods_implemented["SoftMaxLikelihood"] = ["NumericalInference","StochasticNumericalInference"]

isStochastic(inference::String) = (inference == "StochasticAnalyticInference" || inference == "StochasticNumericalInference")

addlargument(likelihood::String) = begin
    if (likelihood == "StudentTLikelihood" || likelihood == "AugmentedStudentTLikelihood")
        return "ν"
    else
        return ""

    end
end
