using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
using Distributions
using MLDataUtils
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
likelihood_types = [reg_likelihood,class_likelihood,poisson_likelihood,negbin_likelihood] #,multiclass_likelihood #Need to fix the one hot encoding for streaming
likelihood_names = ["Regression","Classification","Poisson","NegBinomial"]#"MultiClass", #Need to fix the one hot encoding for streaming
stochastic = [true,false]
indpoints = OIPS(0.8);
floattypes = [Float64]
inference = "AnalyticVI"

@testset "OnlineSVGP" begin
    # Loop over likelihoods
    for (likelihoods,l_names) in zip(likelihood_types,likelihood_names)
        @testset "$l_names" begin
            for l in likelihoods
                @testset "$(string(l))" begin
                    if in(inference,methods_implemented_SVGP[l])
                        for floattype in floattypes
                            @test typeof(OnlineSVGP(k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"("*addiargument(false,inference)*")")),parse(Int,nlatent(l)),indpoints)) <: OnlineSVGP{floattype,eval(Meta.parse(l*"{"*string(floattype)*"}")),eval(Meta.parse(inference*"{"*string(floattype)*","*nlatent(l)*"}")),eval(Meta.parse(nlatent(l)))}
                            model = OnlineSVGP(k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"("*addiargument(false,inference)*")")),parse(Int,nlatent(l)),indpoints,verbose=2)
                            for (Xbatch,ybatch) in eachbatch((X,y[l_names]),obsdim=1,size=10)
                                train!(model,Xbatch, ybatch,iterations=5)
                            end
                            @test testconv(model,l_names,X,y[l_names])
                        end
                    else
                        @test_throws AssertionError OnlineSVGP(k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference)*"("*(addiargument(false,inference))*")"),m)
                    end
                end
            end
        end # for l in likelihoods
    end # for (likelihoods,l_names) in zip(likelihood_types,likelihood_names)
end
