"""
Solve any non-conjugate likelihood using Variational Inference
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood ad its gradients
"""
abstract type NumericalVI{T<:Real} <: Inference{T} end

include("quadratureVI.jl")
include("MCMCVI.jl")


""" `NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Variational Inference via numerical approximation.

**Argument**

    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mcmc` for MCMC integration see [MCMCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCMCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,false)
    elseif integration_technique == :mcmc
        MCMCIntegrationVI{T}(ϵ,nMC,0,optimizer,false)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

""" `NumericalSVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Stochastic Variational Inference via numerical approximation.

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches
    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mcmc` for MCMC integration see [MCMCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCMCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalSVI(nMinibatch::Integer,integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,false,nMinibatch)
    elseif integration_technique == :mcmc
        MCMCIntegrationVI{T}(ϵ,nMC,0,optimizer,false,nMinibatch)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

function Base.show(io::IO,inference::NumericalVI{T}) where T
    print(io,"$(inference.Stochastic ? "Stochastic numerical" : "Numerical") inference with $(isa(inference,MCMCIntegrationVI) ? "MCMC Integration" : "Quadrature")")
end

function init_inference(inference::NumericalVI{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    inference.ρ = nSamples/nSamplesUsed
    inference.HyperParametersUpdated = true
    inference.optimizer_η₁ = [copy(inference.optimizer_η₁[1]) for _ in 1:nLatent]
    inference.optimizer_η₂ = [copy(inference.optimizer_η₂[1]) for _ in 1:nLatent]
    inference.∇η₁ = [zeros(T,nFeatures) for _ in 1:nLatent];
    inference.∇η₂ = [Symmetric(Diagonal(ones(T,nFeatures))) for _ in 1:nLatent]
    inference.∇μE = [zeros(T,nSamplesUsed) for _ in 1:nLatent];
    inference.∇ΣE = [zeros(T,nSamplesUsed) for _ in 1:nLatent]
    return inference
end

function variational_updates!(model::VGP{<:Likelihood,<:NumericalVI})
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates!(model::SVGP{<:Likelihood,<:NumericalVI}) where {L<:Likelihood,T}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function natural_gradient!(model::VGP{<:Likelihood,<:NumericalVI})
    model.inference.∇η₁ .= model.Σ.*(model.inference.∇μE .- model.invKnn.*model.μ)
    model.inference.∇η₂ .= Symmetric.(Diagonal.(model.inference.∇ΣE).-0.5.*model.invKnn .- model.η₂)
end

function natural_gradient!(model::SVGP{<:Likelihood,<:NumericalVI})
    model.inference.∇η₁ .= model.Σ.*(model.inference.ρ.*transpose.(model.κ).*model.inference.∇μE .- model.invKmm.*model.μ)
    model.inference.∇η₂ .= Symmetric.(model.inference.ρ.*transpose.(model.κ).*Diagonal.(model.inference.∇ΣE).*model.κ.-0.5.*model.invKmm .- model.η₂)
end

function global_update!(model::AbstractGP{<:Likelihood,<:NumericalVI})
    model.η₁ .= model.η₁ .+ update.(model.inference.optimizer_η₁,model.inference.∇η₁)
    for k in 1:model.nLatent
        Δ = update(model.inference.optimizer_η₂[k],model.inference.∇η₂[k])
        α=1.0
        while true
            try
                @assert isposdef(-(model.η₂[k]+α*Δ))
                model.η₂[k] = Symmetric(model.η₂[k]+α*Δ)
                model.η₁[k] .+= update(model.inference.optimizer_η₁[k],model.inference.∇η₁[k])
                break;
            catch e
                if isa(e,AssertionError)
                    println("Error, results not pos def with α=$α")
                    α *= 0.5
                    if α < 1e-6
                        @error "α too small, stopping loop"
                        rethrow()
                    end
                else
                    rethrow()
                end
            end
        end
        if isa(model.inference.optimizer_η₂[k],Adam)
            model.inference.optimizer_η₂[k].α = min(model.inference.optimizer_η₂[k].α*α*2.0,1.0)
            # model.inference.optimizer_η₁[k].α = min(model.inference.optimizer_η₁[k].α*α*2.0,1.0)
        elseif isa(model.inference.optimizer_η₂[k],VanillaGradDescent)
            # model.inference.optimizer_η₂[k].η = min(model.inference.optimizer_η₂[k].η*α*2.0,1.0)
            # model.inference.optimizer_η₁[k].η = min(model.inference.optimizer_η₁[k].η*α*2.0,1.0)
        elseif isa(model.inference.optimizer_η₂[k],ALRSVI)
        elseif isa(model.inference.optimizer_η₂[k],InverseDecay)
        end
    end
    model.Σ .= -0.5.*inv.(model.η₂)
    model.μ .= model.η₁
    # model.μ .= model.Σ.*model.η₁
end

function ELBO(model::AbstractGP{<:Likelihood,<:NumericalVI})
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expec_μ(model::AbstractGP{<:Likelihood,<:NumericalVI},index::Integer)
    return model.inference.∇μE[index]
end

function expec_μ(model::AbstractGP{<:Likelihood,<:NumericalVI})
    return model.inference.∇μE
end


function expec_Σ(model::AbstractGP{<:Likelihood,<:NumericalVI},index::Integer)
    return model.inference.∇ΣE[index]
end

function expec_Σ(model::AbstractGP{<:Likelihood,<:NumericalVI})
    return model.inference.∇ΣE
end

function global_update!(model::SVGP{L,NumericalVI{T}}) where {L<:Likelihood,T}
    if model.inference.Stochastic
    else
        model.η₁ .= model.inference.∇η₁ .+ model.η₁
        model.η₂ .= Symmetric.(model.inference.∇η₂ .+ model.η₂)
    end
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end

function convert(::Type{T1},x::T2) where {T1<:VGP{<:Likelihood,T3} where {T3<:NumericalVI},T2<:VGP{<:Likelihood,<:AnalyticVI}}
    #TODO Check likelihood is compatibl
    inference = T3(x.inference.ϵ,x.inference.nIter,x.inference.optimizer,defaultn(T3),x.inference.Stochastic,x.inference.nSamples,x.inference.nSamplesUsed,x.inference.MBIndices,x.inference.ρ,x.inference.HyperParametersUpdated,x.inference.∇η₁,x.inference.∇η₂,copy(expec_μ(x)),copy(expec_Σ(x)))
    likelihood =isaugmented(x.likelihood) ? remove_augmentation(x.likelihood) : likelihood
    return T1(x.X,x.y,x.nSample,x.nDim,x.nFeature,x.nLatent,x.IndependentPriors,x.nPrior,x.μ,x.Σ,x.η₁,x.η₂,x.Knn,x.invKnn,x.kernel,likelihood,inference,x.verbose,x.Autotuning,x.atfrequency,x.Trained)
end
