using Test
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses
include("compat.jl")

nData = 100; nDim = 2
m = 50; b= 10
k = AGP.RBFKernel()

X = rand(nData,nDim)
y = Dict("Regression"=>norm.(eachrow(X)),"Classification"=>sign.(norm.(eachrow(X)).-1.0),"MultiClass"=>floor.(norm.(eachrow(X.*2))))
reg_likelihood = ["GaussianLikelihood","AugmentedStudentTLikelihood","StudentTLikelihood"]
class_likelihood = ["AugmentedLogisticLikelihood"]#,"LogisticLikelihood"]
multiclass_likelihood = ["AugmentedLogisticSoftMaxLikelihood","LogisticSoftMaxLikelihood","SoftMaxLikelihood"]
likelihood_types = [reg_likelihood,class_likelihood,multiclass_likelihood]
likelihood_names = ["Regression","Classification","MultiClass"]
inferences = ["AnalyticInference","StochasticAnalyticInference","NumericalInference","StochasticNumericalInference","GibbsSampling"]
floattypes = [Float64]
@testset "SVGP" begin
    for (likelihoods,l_names) in zip(likelihood_types,likelihood_names)
        @testset "$l_names" begin
            for l in likelihoods
                @testset "$(string(l))" begin
                    for inference in inferences
                        @testset "$(string(inference))" begin
                            if in(inference,methods_implemented[l])
                                for floattype in floattypes                              println(SVGP(floattype.(X),y[l_names],k,eval(Meta.parse(l*"()")),eval(Meta.parse(inference*"("*(isStochastic(inference) ? string(b) : "")*")")),m))
                                # for floattype in floattypes                              @test typeof(SVGP(floattype.(X),y[l_names],k,eval(Meta.parse(l*"()")),eval(Meta.parse(inference*"("*(isStochastic(inference) ? string(b) : "")*")")),m)) <: SVGP{eval(Meta.parse(l*"{"*string(floattype)*"}")),eval(Meta.parse(inference*"{"*string(floattype)*"}")),floattype,Vector{floattype}}
                                end
                            else
                                @test_throws AssertionError SVGP(floattype.(X),y[l_names],k,eval(Meta.parse(l*"()")),eval(Meta.parse(inference*"("*(isStochastic(inference) ? string(b) : "")*")")),m)
                            end
                        end
                    end
                end
            end
        end
    end
end
