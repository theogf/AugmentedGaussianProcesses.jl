""" Solve any non-conjugate likelihood
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood"""
abstract type NumericalInference{T<:Real} <: Inference{T} end

function NumericalInference(integration_technique::Symbol=:quad,)
if integration_technique == :quad
    QuadratureInference{T}()
elseif integration_technique == :mcmc
    MCMCIntegrationInference{T}()
else
    @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
end
end

function NumericalInference(nSamples::Integer;ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    NumericalInference{T}(ϵ,nMC,0,optimizer,false,nSample,nSample,1:nSample,1.0,true)
end

function NumericalInference(Stochastic::Bool,nSample::Integer,nSampleUsed::Integer,η₁::AbstractVector{<:AbstractVector},η₂::AbstractVector{<:AbstractMatrix};ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    NumericalInference{T}(ϵ,nMC,0,optimizer,Stochastic,nSample,nSampleUsed,1:nSampleUsed,nSample/nSampleUsed,true,copy(η₁),copy(η₂))
end

function NumericalInference(;ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    NumericalInference{Float64}(ϵ,nMC,0,optimizer,false,1,1,[1],1.0,true)
end

function init_inference(inference::NumericalInference{T},nLatent::Integer,nFeatures::Integer,nSamplesUsed::Integer) where {T<:Real}
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
    model.inference.∇η₁ .= model.inference.∇μE .- model.η₁
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
