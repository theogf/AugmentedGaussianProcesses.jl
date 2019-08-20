"""
Solve any non-conjugate likelihood using Variational Inference
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood ad its gradients
Gradients are computed as in "The Variational Gaussian Approximation
Revisited" by Opper and Archambeau 2009
"""
abstract type NumericalVI{T<:Real} <: Inference{T} end

include("quadratureVI.jl")
include("MCVI.jl")


""" `NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Variational Inference via numerical approximation.

**Argument**

    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Momentum(η=1e-5)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,false)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,0,optimizer,false)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

""" `NumericalSVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Stochastic Variational Inference via numerical approximation.

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches
    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalSVI(nMinibatch::Integer,integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimizer::Optimizer=Momentum(η=1e-5)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,true,nMinibatch)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,0,optimizer,true,nMinibatch)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mc"
    end
end

function Base.show(io::IO,inference::NumericalVI{T}) where T
    print(io,"$(inference.Stochastic ? "Stochastic numerical" : "Numerical") inference by $(isa(inference,MCIntegrationVI) ? "Monte Carlo Integration" : "Quadrature")")
end

function init_inference(inference::NumericalVI{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    inference.ρ = nSamples/nSamplesUsed
    inference.HyperParametersUpdated = true
    inference.optimizer = [copy(inference.optimizer[1]) for _ in 1:nLatent]
    inference.∇η₁ = [zeros(T,nFeatures) for _ in 1:nLatent];
    inference.∇η₂ = [Symmetric(Diagonal(ones(T,nFeatures))) for _ in 1:nLatent]
    inference.ν = [zeros(T,nSamplesUsed) for _ in 1:nLatent];
    inference.λ = [zeros(T,nSamplesUsed) for _ in 1:nLatent]
    return inference
end

∇E_μ(model::AbstractGP{T,L,<:NumericalVI}) where {T,L} = -model.inference.ν
∇E_μ(model::AbstractGP{T,L,<:NumericalVI},i::Int) where {T,L} = -model.inference.ν[i]
∇E_Σ(model::AbstractGP{T,L,<:NumericalVI}) where {T,L} = 0.5.*model.inference.λ
∇E_Σ(model::AbstractGP{T,L,<:NumericalVI},i::Int) where {T,L} = 0.5.*model.inference.λ[i]

function variational_updates!(model::VGP{T,L,<:NumericalVI}) where {T,L}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates!(model::SVGP{T,L,<:NumericalVI}) where {T,L}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function natural_gradient!(model::VGP{T,L,<:NumericalVI}) where {T,L}
    model.inference.∇η₂ .= Symmetric.(Diagonal.(∇E_Σ(model)) .- 0.5.*model.invKnn .- model.η₂)
    model.inference.∇η₁ .= ∇E_μ(model) .- model.invKnn.*(model.μ.-model.μ₀) - 2 .*model.inference.∇η₂.*model.μ
end

function natural_gradient!(model::SVGP{T,L,<:NumericalVI}) where {T,L}
    model.inference.∇η₂ .= Symmetric.(model.inference.ρ.*transpose.(model.κ).*Diagonal.(∇E_Σ(model)).*model.κ.-0.5.*model.invKmm .- model.η₂)
    model.inference.∇η₁ .= model.inference.ρ.*transpose.(model.κ).*∇E_μ(model) .- model.invKmm.*(model.μ.-model.μ₀) - 2 .*model.inference.∇η₂.*model.μ
end

function global_update!(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    # model.η₁ .= model.η₁ .+ update.(model.inference.optimizer_η₁,model.inference.∇η₁)
    for k in 1:model.nLatent
        Δ = update(model.inference.optimizer[k],vcat(model.inference.∇η₁[1],model.inference.∇η₂[k][:]))
        Δ₁ = Δ[1:model.nFeatures]
        Δ₂ = reshape(Δ[model.nFeatures+1:end],model.nFeatures,model.nFeatures)
        # Δ = update(model.inference.optimizer_η₂[k],model.inference.∇η₂[k])
        α=1.0
        # Loop to verify update keeps positive definiteness
        while !isposdef(-(model.η₂[k] + α*Δ₂)) &&  α > 1e-6
            α *= 0.1
        end
        if α <= 1e-6
            @error "α too small, positive definiteness could not be achieved"
        end
        model.η₂[k] = Symmetric(model.η₂[k] + α*Δ₂)
        model.η₁[k] = model.η₁[k] + α*Δ₁
        if isa(model.inference.optimizer[k],Adam)
            model.inference.optimizer[k].α = min(model.inference.optimizer_η₂[k].α * α*2.0,0.1)
        elseif isa(model.inference.optimizer[k],Union{VanillaGradDescent,Momentum,RMSprop})
            model.inference.optimizer[k].η = min(model.inference.optimizer[k].η*α*1.01,0.1)
        elseif isa(model.inference.optimizer[k],ALRSVI)
        elseif isa(model.inference.optimizer[k],InverseDecay)
        end
    end
    model.Σ .= -0.5.*inv.(model.η₂)
    model.μ .= model.Σ.*model.η₁
end

function ELBO(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function convert(::Type{T1},x::T2) where {T1<:VGP{<:Likelihood,T3} where {T3<:NumericalVI},T2<:VGP{<:Real,<:Likelihood,<:AnalyticVI}}
    #TODO Check if likelihood is compatible
    inference = T3(x.inference.ϵ,x.inference.nIter,x.inference.optimizer,defaultn(T3),x.inference.Stochastic,x.inference.nSamples,x.inference.nSamplesUsed,x.inference.MBIndices,x.inference.ρ,x.inference.HyperParametersUpdated,x.inference.∇η₁,x.inference.∇η₂,copy(expec_μ(x)),copy(expec_Σ(x)))
    likelihood =isaugmented(x.likelihood) ? remove_augmentation(x.likelihood) : likelihood
    return T1(x.X,x.y,x.nSample,x.nDim,x.nFeatures,x.nLatent,x.IndependentPriors,x.nPrior,x.μ,x.Σ,x.η₁,x.η₂,x.Knn,x.invKnn,x.kernel,likelihood,inference,x.verbose,x.optimizer,x.atfrequency,x.Trained)
end
