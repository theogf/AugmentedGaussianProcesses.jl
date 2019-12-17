using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
using Distributions
const AGP = AugmentedGaussianProcesses
include("testingtools.jl")

nData = 50; nDim = 2
k = SqExponentialKernel(10.0)
ν = 5.0
r = 10
K = 4
X = rand(nData,nDim)
f_ = ones(nData)
while !(maximum(f_) > 0 && minimum(f_) < 0)
    global f_ = rand(MvNormal(zeros(nData),kernelmatrix(k,X,obsdim=1)+1e-3I))
end
width = maximum(f_)-minimum(f_)
normf = (f_.-minimum(f_))/width*K

y = Dict("Regression"=>f_,"Classification"=>sign.(f_),"MultiClass"=>floor.(Int64,normf),"Poisson"=>rand.(Poisson.(2.0*AGP.logistic.(f_))),"NegBinomial"=>rand.(NegativeBinomial.(r,AGP.logistic.(f_))))
n_class = length(unique(y["MultiClass"]))
reg_likelihood = ["GaussianLikelihood","StudentTLikelihood","LaplaceLikelihood","HeteroscedasticLikelihood"]
class_likelihood = ["BayesianSVM","LogisticLikelihood"]
multiclass_likelihood = ["LogisticSoftMaxLikelihood","SoftMaxLikelihood"]
poisson_likelihood = ["PoissonLikelihood"]
negbin_likelihood = ["NegBinomialLikelihood"]
likelihood_types = [reg_likelihood,class_likelihood,multiclass_likelihood,poisson_likelihood,negbin_likelihood]
# likelihood_types = [negbin_likelihood]
likelihood_names = ["Regression","Classification","MultiClass","Poisson","NegBinomial"]
# likelihood_names = ["NegBinomial"]
# inferences = ["GibbsSampling"]#,"NumericalInference"]
inferences = ["AnalyticVI"]#,"GibbsSampling","QuadratureVI"]
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
                                    @test typeof(VGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"("*addiargument(false,inference)*")")))) <: VGP{floattype,eval(Meta.parse(l*"{"*string(floattype)*"}")),eval(Meta.parse(inference*"{"*string(floattype)*","*nlatent(l)*"}")),eval(Meta.parse(nlatent(l)))}
                                    global model = VGP(X,y[l_names],k,eval(Meta.parse(l*"("*addlargument(l)*")")),eval(Meta.parse(inference*"("*addiargument(false,inference)*")")),verbose=2)
                                    @test train!(model,50)
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
