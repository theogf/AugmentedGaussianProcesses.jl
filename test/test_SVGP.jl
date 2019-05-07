using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
const AGP = AugmentedGaussianProcesses
include("testingtools.jl")

nData = 100; nDim = 2
m = 50; b = 50
k = AGP.RBFKernel()
ν = 5.0

X = rand(nData,nDim)
y = Dict("Regression"=>norm.(eachrow(X)),"Classification"=>sign.(norm.(eachrow(X)).-1.0),"MultiClass"=>floor.(norm.(eachrow(X.*2))),"Event"=>rand.(Poisson.(norm.(eachrow(X)))))
reg_likelihood = ["GaussianLikelihood","StudentTLikelihood","LaplaceLikelihood"]
class_likelihood = ["BayesianSVM","LogisticLikelihood"]
multiclass_likelihood = ["LogisticSoftMaxLikelihood","SoftMaxLikelihood"]
event_likelihood = ["PoissonLikelihood"]
likelihood_types = [reg_likelihood,class_likelihood,multiclass_likelihood,event_likelihood]
likelihood_names = ["Regression","Classification","MultiClass","Event"]
stochastic = [true,false]
inferences = ["AnalyticVI","GibbsSampling"]#,"QuadratureVI","MCMCIntegrationVI"]
floattypes = [Float64]
@testset "SVGP" begin
    for (likelihoods,l_names) in zip(likelihood_types,likelihood_names)
        @testset "$l_names" begin
            for l in likelihoods
                @testset "$(string(l))" begin
                    for inference in inferences
                        for s in stochastic
                            @testset "$(string(stoch(s,inference)))" begin
                                if in(stoch(s,inference),methods_implemented_SVGP[l])
                                    for floattype in floattypes
                                        @test typeof(SVGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(stoch(s,inference)*"("*addiargument(s,inference)*")")),m)) <: SVGP{eval(Meta.parse(l*"{"*string(floattype)*"}")),eval(Meta.parse(inference*"{"*string(floattype)*"}")),floattype,Vector{floattype}}
                                        model = SVGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(stoch(s,inference)*"("*(addiargument(s,inference))*")")),m,Autotuning=true,verbose=2)
                                        @test train!(model,iterations=50)
                                        @test testconv(model,l_names,X,y[l_names])
                                    end
                                else
                                    @test_throws AssertionError SVGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(stoch(s,inference)*"("*(addiargument(s,inference))*")")),m)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
