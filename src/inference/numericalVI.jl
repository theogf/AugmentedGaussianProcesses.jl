"""
Solve any non-conjugate likelihood using Variational Inference
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood ad its gradients
Gradients are computed as in "The Variational Gaussian Approximation
Revisited" by Opper and Archambeau 2009
"""
abstract type NumericalVI{T<:Real} <: VariationalInference{T} end

include("quadratureVI.jl")
include("MCVI.jl")


""" `NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimiser=Momentum(0.001))`

General constructor for Variational Inference via numerical approximation.

**Argument**

    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.001)`
"""
function NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimiser=Momentum(1e-3)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,optimiser,false,0)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,optimiser,false,0)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

""" `NumericalSVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer=Momentum(0.001))`

General constructor for Stochastic Variational Inference via numerical approximation.

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches
    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.001)`
"""
function NumericalSVI(nMinibatch::Integer,integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimiser=Momentum(1e-3)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,optimizer,true,nMinibatch)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,optimizer,true,nMinibatch)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mc"
    end
end

function Base.show(io::IO,inference::NumericalVI{T}) where T
    print(io,"$(inference.Stochastic ? "Stochastic numerical" : "Numerical") inference by $(isa(inference,MCIntegrationVI) ? "Monte Carlo Integration" : "Quadrature")")
end

∇E_μ(::Likelihood,i::NVIOptimizer,::AbstractVector) = -i.ν
∇E_Σ(::Likelihood,i::NVIOptimizer,::AbstractVector) = 0.5.*i.λ

function variational_updates!(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    compute_grad_expectations!(model)
    natural_gradient!.(model.likelihood,model.inference,model.inference.vi_opt,[get_y(model)],model.f)
    global_update!(model)
end

function natural_gradient!(l::Likelihood,i::NumericalVI,opt::NVIOptimizer,y::AbstractVector,X::AbstractMatrix,gp::_VGP{T}) where {T,L}
    opt.∇η₂ .= Symmetric(Diagonal(∇E_Σ(l,opt,y)) - 0.5*inv(gp.K) - gp.η₂)
    opt.∇η₁ .= ∇E_μ(l,opt,y) - gp.K \ (gp.μ - gp.μ₀(X)) - 2 * opt.∇η₂ * gp.μ
end

function natural_gradient!(l::Likelihood,i::NumericalVI,opt::NVIOptimizer,y::AbstractVector,Z::AbstractMatrix,gp::_SVGP{T}) where {T,L}
    opt.∇η₂ .= Symmetric(i.ρ*transpose(gp.κ)*Diagonal(∇E_Σ(l,opt,y))*gp.κ - 0.5*inv(gp.K) - gp.η₂)
    opt.∇η₁ .= i.ρ * transpose(gp.κ) * ∇E_μ(l,opt,y) - gp.K \ (gp.μ - gp.μ₀(Z)) - 2 * opt.∇η₂ * gp.μ
end

function global_update!(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    for (gp,opt) in zip(model.f,model.inference.vi_opt)
        Δ = update(opt.optimizer,vcat(opt.∇η₁,vec(LowerTriangular(-2*opt.∇η₂*opt.L))))
        Δ₁ = Δ[1:model.nFeatures]
        Δ₂ = LowerTriangular(reshape(Δ[model.nFeatures+1:end],model.nFeatures,model.nFeatures))
        display(ELBO(model))
        α = 1.0
        # # Loop to verify update keeps positive definiteness
        while !isposdef(Symmetric((opt.L+α*Δ₂)*(opt.L+α*Δ₂)')) &&  α > 1e-7
            α *= 0.1
        end
        if α < 1e-7
            @error "α too small, positive definiteness could not be achieved"
        end
        opt.L = LowerTriangular(opt.L+α*Δ₂)
        gp.η₂ .= Symmetric(-opt.L*opt.L')
        display(gp.η₁)
        display(Δ₁)

        gp.η₁ .+= α*Δ₁
        ## Passed the pos. def. test, now update parameters
        # gp.η₂ = Symmetric(gp.η₂+α*Δ₂)
        # gp.η₁ .+= α*Δ₁

        ## Save the new scaling on the optimizer
        #TODO
        ## Reparametrize to the normal distribution
        global_update!.(model.f)
    end
end

function ELBO(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    return model.inference.ρ*expec_log_likelihood(model.likelihood,model.inference,get_y(model),mean_f(model),diag_cov_f(model)) - GaussianKL(model)
end
