""" Solve any non-conjugate likelihood
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood"""
abstract type NumericalInference{T<:Real} <: Inference{T} end

function NumericalInference(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))
    if integration_technique == :quad
        QuadratureInference{T}(ϵ,nGaussHermite,0,optimizer,false)
    elseif integration_technique == :mcmc
        MCMCIntegrationInference{T}(ϵ,nMC,0,optimizer,false)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

function StochasticNumericalInference(nMinibatch::Integer,integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))
    if integration_technique == :quad
        QuadratureInference{T}(ϵ,nGaussHermite,0,optimizer,false,nMinibatch)
    elseif integration_technique == :mcmc
        MCMCIntegrationInference{T}(ϵ,nMC,0,optimizer,false,nMinibatch)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

function Base.show(io::IO,inference::NumericalInference{T}) where T
    print(io,"($(inference.Stochastic ? "Stochastic numerical" : "Numerical") inference with $(isa(inference,MCMCIntegrationInference) ? "MCMC Integration" : "Quadrature")")
end

function init_inference(inference::TInference,nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {TInference <: AnalyticInference{T} where T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    inference.ρ = nSamples/nSamplesUsed
    inference.HyperParametersUpdated = false
    inference.optimizer_η₁ = [copy(inference.optimizer_η₁[1]) for _ in 1:nLatent]
    inference.optimizer_η₂ = [copy(inference.optimizer_η₂[1]) for _ in 1:nLatent]
    inference.∇η₁ = [zeros(T,nFeatures) for _ in 1:nLatent];
    inference.∇η₂ = [Symmetric(Diagonal(ones(T,nFeatures))) for _ in 1:nLatent]
    inference.∇μE = [zeros(T,nSamplesUsed) for _ in 1:nLatent];
    inference.∇ΣE = [zeros(T,nSamplesUsed) for _ in 1:nLatent]
    return inference
end

function variational_updates!(model::VGP{<:Likelihood,<:NumericalInference})
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates!(model::SVGP{<:Likelihood,<:NumericalInference}) where {L<:Likelihood,T}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    compute_learningrate!(model)
    global_update!(model)
end


function natural_gradient!(model::VGP{<:Likelihood,<:NumericalInference})
    model.inference.∇η₁ .= model.Σ*model.inference.∇μE .- model.η₁
    model.inference.∇η₂ .= Symmetric.(-Diagonal.(model.inference.∇ΣE)+0.5.*model.invKmm .- model.η₂)
end

function natural_gradient!(model::SVGP{<:Likelihood,<:NumericalInference})
    model.inference.∇η₁ .= model.inference.ρ.*transpose.(model.κ).*model.inference.∇μE .- model.η₁
    model.inference.∇η₂ .= Symmetric.(-model.inference.ρ.*transpose.(model.κ).*Diagonal.(model.inference.∇ΣE).*model.κ.+0.5.*model.invKmm .- model.η₂)
end

function global_update!(model::VGP)
    model.η₁ .+= model.inference.∇η₁
    model.η₂ .+= model.inference.∇η₂
    model.Σ .= inv.(model.∇η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end

function expec_μ(model::GP{<:Likelihood,<:NumericalInference},index::Integer)
    return model.inference.∇μE[index]
end

function expec_μ(model::GP{<:Likelihood,<:NumericalInference})
    return model.inference.∇μE
end


function expec_Σ(model::GP{<:Likelihood,<:NumericalInference},index::Integer)
    return model.inference.∇ΣE[index]
end

function expec_Σ(model::GP{<:Likelihood,<:NumericalInference})
    return model.inference.∇ΣE
end

function compute_learningrate!(model::SVGP{L,NumericalInference{T}}) where {L<:Likelihood,T}
 #TODO learningrate_optimizer
end

function global_update!(model::SVGP{L,NumericalInference{T}}) where {L<:Likelihood,T}
    if model.inference.Stochastic
    else
        model.η₁ .= model.inference.∇η₁ .+ model.η₁
        model.η₂ .= Symmetric.(model.inference.∇η₂ .+ model.η₂)
    end
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end

function convert(::Type{T1},x::T2) where {T1<:VGP{<:Likelihood,T3} where {T3<:NumericalInference},T2<:VGP{<:Likelihood,<:AnalyticInference}}
    #TODO Check likelihood is compatibl
    inference = T3(x.inference.ϵ,x.inference.nIter,x.inference.optimizer,defaultn(T3),x.inference.Stochastic,x.inference.nSamples,x.inference.nSamplesUsed,x.inference.MBIndices,x.inference.ρ,x.inference.HyperParametersUpdated,x.inference.∇η₁,x.inference.∇η₂,copy(expec_μ(x)),copy(expec_Σ(x)))
    likelihood =isaugmented(x.likelihood) ? remove_augmentation(x.likelihood) : likelihood
    return T1(x.X,x.y,x.nSample,x.nDim,x.nFeature,x.nLatent,x.IndependentPriors,x.nPrior,x.μ,x.Σ,x.η₁,x.η₂,x.Knn,x.invKnn,x.kernel,likelihood,inference,x.verbose,x.Autotuning,x.atfrequency,x.Trained)
end
