using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
using Distributions
const AGP = AugmentedGaussianProcesses
include("testingtools.jl")






# likelihood_types = [negbin_likelihood]
# likelihood_names = ["NegBinomial"]
# inferences = ["GibbsSampling"]#,"NumericalInference"]
inferences = ["AnalyticVI"]#,"GibbsSampling","QuadratureVI"]
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
