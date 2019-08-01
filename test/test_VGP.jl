using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
using Distributions
const AGP = AugmentedGaussianProcesses
include("testingtools.jl")

nData = 100; nDim = 2
k = AGP.RBFKernel(0.1)
ν = 5.0
K = 4
X = rand(nData,nDim)
f = ones(nData)
while !(maximum(f) > 0 && minimum(f) < 0)
    global f = rand(MvNormal(zeros(nData),AGP.kernelmatrix(X,k)+1e-3I))
end
width = maximum(f)-minimum(f)
normf = (f.-minimum(f))/width*K

y = Dict("Regression"=>f,"Classification"=>sign.(f),"MultiClass"=>floor.(Int64,normf),"Event"=>rand.(Poisson.(2.0*AGP.logistic.(f))))
reg_likelihood = ["GaussianLikelihood","StudentTLikelihood","LaplaceLikelihood","HeteroscedasticLikelihood"]
# reg_likelihood = ["LaplaceLikelihood"]
class_likelihood = ["BayesianSVM","LogisticLikelihood"]
multiclass_likelihood = ["LogisticSoftMaxLikelihood","SoftMaxLikelihood"]
event_likelihood = ["PoissonLikelihood"]
likelihood_types = [reg_likelihood,class_likelihood,multiclass_likelihood,event_likelihood]
likelihood_names = ["Regression","Classification","MultiClass","Event"]
# likelihood_names = ["Regression"]
# inferences = ["GibbsSampling"]#,"NumericalInference"]
inferences = ["AnalyticVI","GibbsSampling","QuadratureVI"]#,"NumericalInference"]
floattypes = [Float64]
@testset "VGP" begin
    for (likelihoods,l_names) in zip(likelihood_types,likelihood_names)
        @testset "$l_names" begin
            for l in likelihoods
                @testset "$(string(l))" begin
                    for inference in inferences
                        @testset "$(string(inference))" begin
                            if in(inference,methods_implemented_VGP[l])
                                for floattype in floattypes
                                    @test typeof(VGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"("*addiargument(false,inference)*")")))) <: VGP{floattype,eval(Meta.parse(l*"{"*string(floattype)*"}")),eval(Meta.parse(inference*"{"*string(floattype)*"}")),Vector{floattype}}
                                    global model = VGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"("*addiargument(false,inference)*")")),verbose=2)
                                    @test train!(model,iterations=50)
                                    @test testconv(model,l_names,X,y[l_names])
                                end
                            else
                                @test_throws AssertionError VGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"()")))
                            end
                        end
                    end
                end
            end
        end
    end
end
