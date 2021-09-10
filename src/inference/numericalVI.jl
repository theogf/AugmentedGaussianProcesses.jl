#= 
Solve any non-conjugate likelihood using Variational Inference
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood ad its gradients
Gradients are computed as in "The Variational Gaussian Approximation
Revisited" by Opper and Archambeau 2009 =#
abstract type NumericalVI{T<:Real} <: VariationalInference{T} end

include("quadratureVI.jl")
include("MCVI.jl")

isnatural(vi::NumericalVI) = vi.NaturalGradient

"""
    NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimiser=Momentum(0.001))

General constructor for Variational Inference via numerical approximation.

## Arguments
-`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

## Keyword arguments
- `ϵ::T` : convergence criteria, which can be user defined
- `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
- `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.001)`
"""
NumericalVI

function NumericalVI(
    integration_technique::Symbol=:quad;
    ϵ::T=1e-5,
    nMC::Integer=1000,
    nGaussHermite::Integer=20,
    optimiser=Momentum(1e-3),
    natural::Bool=true,
) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ, nGaussHermite, optimiser, false, 0.0, 0, natural)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ, nMC, optimiser, false, 0.0, 0, natural)
    else
        throw(
            ErrorException(
                "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc",
            ),
        )
    end
end

"""
    NumericalSVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer=Momentum(0.001))

General constructor for Stochastic Variational Inference via numerical approximation.

## Arguments

- `nMinibatch::Integer` : Number of samples per mini-batches
- `integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

## Keyword arguments
- `ϵ::T` : convergence criteria, which can be user defined
- `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
- `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.001)`
"""
function NumericalSVI(
    nMinibatch::Integer,
    integration_technique::Symbol=:quad;
    ϵ::T=1e-5,
    nMC::Integer=200,
    nGaussHermite::Integer=20,
    optimiser=Momentum(1e-3),
    natural::Bool=true,
) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ, nGaussHermite, optimiser, true, 0.0, nMinibatch, natural)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ, nMC, optimiser, true, 0.0, nMinibatch, natural)
    else
        throw(
            ErrorException(
                "Only possible integration techniques are quadrature : :quad or mcmc integration :mc",
            ),
        )
    end
end

function Base.show(io::IO, inference::NumericalVI)
    return print(
        io,
        "$(isStochastic(inference) ? "Stochastic numerical" : "Numerical") Inference by $(isa(inference, MCIntegrationVI) ? "Monte Carlo Integration" : "Quadrature")",
    )
end

∇E_μ(::AbstractLikelihood, i::NVIOptimizer, ::AbstractVector) = (-i.ν,)
∇E_Σ(::AbstractLikelihood, i::NVIOptimizer, ::AbstractVector) = (0.5 .* i.λ,)

function variational_updates!(model::AbstractGPModel{T,L,<:NumericalVI}) where {T,L}
    grad_expectations!(model)
    classical_gradient!.(
        ∇E_μ(likelihood(model), model.inference.vi_opt[1], []),
        ∇E_Σ(likelihood(model), model.inference.vi_opt[1], []),
        model.inference,
        model.inference.vi_opt,
        Zviews(model),
        model.f,
    )
    if isnatural(model.inference)
        natural_gradient!.(model.f, model.inference.vi_opt)
    end
    return global_update!(model)
end

function classical_gradient!(
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::NumericalVI,
    opt::NVIOptimizer,
    X::AbstractVector,
    gp::VarLatent{T},
) where {T<:Real}
    opt.∇η₂ .= Diagonal(∇E_Σ) - 0.5 * (inv(pr_cov(gp)) - inv(cov(gp)))
    return opt.∇η₁ .= ∇E_μ - pr_cov(gp) \ (mean(gp) - pr_mean(gp, X))
end

function classical_gradient!(
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::NumericalVI,
    opt::NVIOptimizer,
    Z::AbstractVector,
    gp::SparseVarLatent{T},
) where {T<:Real}
    opt.∇η₂ .=
        getρ(i) * transpose(gp.κ) * Diagonal(∇E_Σ) * gp.κ -
        0.5 * (inv(pr_cov(gp)) - inv(cov(gp)))
    return opt.∇η₁ .=
        getρ(i) * transpose(gp.κ) * ∇E_μ - pr_cov(gp) \ (mean(gp) - pr_mean(gp, Z))
end

function natural_gradient!(gp::AbstractLatent, opt::NVIOptimizer)
    opt.∇η₂ .= 2 * cov(gp) * opt.∇η₂ * cov(gp)
    return opt.∇η₁ .= pr_cov(gp) * opt.∇η₁
end

function global_update!(model::AbstractGPModel{T,L,<:NumericalVI}) where {T,L}
    for (gp, opt) in zip(model.f, model.inference.vi_opt)
        Δ1 = Optimise.apply!(opt.optimiser, mean(gp), opt.∇η₁)
        Δ2 = Optimise.apply!(opt.optimiser, cov(gp).data, opt.∇η₂)
        gp.post.μ .+= Δ1
        α = 1.0
        while !isposdef(cov(gp) + α * Symmetric(Δ2)) && α > 1e-8
            α *= 0.5
        end
        if α > 1e-8
            gp.post.Σ .= cov(gp) + α * Symmetric(Δ2)
        else
            @warn "α was too small for update" maxlog = 10
        end
        # global_update!.(model.f)
    end
end

## ELBO

function expec_loglikelihood(
    l::AbstractLikelihood,
    i::NumericalVI,
    y,
    μ::Tuple{<:AbstractVector{T}},
    Σ::Tuple{<:AbstractVector{T}},
) where {T}
    return expec_loglikelihood(l, i, y, first(μ), first(Σ))
end

function ELBO(m::AbstractGPModel{T,L,<:NumericalVI}) where {T,L}
    tot = zero(T)
    tot +=
        getρ(m.inference) *
        expec_loglikelihood(likelihood(m), inference(m), yview(m), mean_f(m), var_f(m))
    tot -= GaussianKL(m)
    tot -= extraKL(m)
    return tot
end
