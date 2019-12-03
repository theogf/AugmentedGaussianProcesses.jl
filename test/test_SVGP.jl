using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
using Distributions
using KernelFunctions
const AGP = AugmentedGaussianProcesses
include("testingtools.jl")

nData = 100; nDim = 2
m = 50; b = 50
k = SqExponentialKernel(10.0)
K = 4
ν = 5.0
r = 10

X = rand(nData,nDim)
f_ = ones(nData)
while !(maximum(f_) > 0 && minimum(f_) < 0)
    global f_ = rand(MvNormal(zeros(nData),kernelmatrix(k,X,obsdim=1)+1e-3I))
end
width = maximum(f_)-minimum(f_)
normf = (f_.-minimum(f_))/width*K

y = Dict("Regression"=>f_,"Classification"=>sign.(f_),"MultiClass"=>floor.(Int64,normf),"Poisson"=>rand.(Poisson.(2.0*AGP.logistic.(f_))),"NegBinomial"=>rand.(NegativeBinomial.(r,AGP.logistic.(f_))))
n_class = length(unique(y["MultiClass"]))

reg_likelihood = ["GaussianLikelihood","StudentTLikelihood","LaplaceLikelihood"]
class_likelihood = ["BayesianSVM","LogisticLikelihood"]
multiclass_likelihood = ["LogisticSoftMaxLikelihood","SoftMaxLikelihood"]
poisson_likelihood = ["PoissonLikelihood"]
negbin_likelihood = ["NegBinomialLikelihood"]
likelihood_types = [reg_likelihood,class_likelihood,multiclass_likelihood,poisson_likelihood,negbin_likelihood]
likelihood_names = ["Regression","Classification","MultiClass","Poisson","NegBinomial"]
stochastic = [true,false]
inferences = ["AnalyticVI"]#,"GibbsSampling","QuadratureVI"]#,"MCIntegrationVI"]
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
                                        @test typeof(SVGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(stoch(s,inference)*"("*addiargument(s,inference)*")")),m)) <: SVGP{floattype,eval(Meta.parse(l*"{"*string(floattype)*"}")),eval(Meta.parse(inference*"{"*string(floattype)*","*nlatent(l)*"}")),AGP._SVGP{floattype},eval(Meta.parse(nlatent(l)))}
                                        model = SVGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(stoch(s,inference)*"("*(addiargument(s,inference))*")")),m,verbose=2)
                                        @test train!(model,50)
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
