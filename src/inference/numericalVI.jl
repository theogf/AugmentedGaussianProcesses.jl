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
    NumericalVI(integration_technique::Symbol=:quad; ϵ::T=1e-5, nMC::Integer=1000, nGaussHermite::Integer=20, optimiser=Momentum(0.001))

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

- `batchsize::Integer` : Number of samples per mini-batches
- `integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

## Keyword arguments
- `ϵ::T` : convergence criteria, which can be user defined
- `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
- `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.001)`
"""
function NumericalSVI(
    batchsize::Integer,
    integration_technique::Symbol=:quad;
    ϵ::T=1e-5,
    nMC::Integer=200,
    nGaussHermite::Integer=20,
    optimiser=Momentum(1e-3),
    natural::Bool=true,
) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ, nGaussHermite, optimiser, true, 0.0, batchsize, natural)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ, nMC, optimiser, true, 0.0, batchsize, natural)
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
        "$(is_stochastic(inference) ? "Stochastic numerical" : "Numerical") Inference by $(isa(inference, MCIntegrationVI) ? "Monte Carlo Integration" : "Quadrature")",
    )
end

∇E_μ(::AbstractLikelihood, ::NVIOptimizer, ::AbstractVector, state) = (-state.ν,)
∇E_Σ(::AbstractLikelihood, ::NVIOptimizer, ::AbstractVector, state) = (0.5 .* state.λ,)

function variational_updates(
    model::AbstractGPModel{T,L,<:NumericalVI}, state, y
) where {T,L}
    grad_expectations!(model, state, y)
    classical_gradient!.(
        model.f,
        ∇E_μ(likelihood(model), vi_opt(inference(model)), y, state.opt_state),
        ∇E_Σ(likelihood(model), vi_opt(inference(model)), y, state.opt_state),
        inference(model),
        Zviews(model),
        state.kernel_matrices,
        state.opt_state,
    )
    if isnatural(inference(model))
        natural_gradient!.(model.f, state.kernel_matrices, state.opt_state)
    end
    return global_update!(model)
end

function classical_gradient!(
    gp::VarLatent{T},
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    ::NumericalVI,
    X::AbstractVector,
    kernel_matrices,
    opt_state,
) where {T<:Real}
    K = kernel_matrices.K
    opt_state.∇η₂ .= Diagonal(∇E_Σ) - 0.5 * (inv(K) - inv(cov(gp)))
    opt_state.∇η₁ .= ∇E_μ - K \ (mean(gp) - pr_mean(gp, X))
    return opt_state
end

function classical_gradient!(
    gp::SparseVarLatent{T},
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::NumericalVI,
    Z::AbstractVector,
    kernel_matrices,
    opt_state,
) where {T<:Real}
    K = kernel_matrices.K
    κ = kernel_matrices.κ
    opt_state.∇η₂ .= ρκdiagθκ(ρ(i), κ, ∇E_Σ) - 0.5 * (inv(K) - inv(cov(gp)))
    opt_state.∇η₁ .= ρ(i) * transpose(κ) * ∇E_μ - K \ (mean(gp) - pr_mean(gp, Z))
    return opt_state
end

function natural_gradient!(gp::AbstractLatent, kernel_matrices, opt_state)
    opt_state.∇η₂ .= 2 * cov(gp) * opt_state.∇η₂ * cov(gp)
    opt_state.∇η₁ .= kernel_matrices.K * opt_state.∇η₁
    return opt_state
end

function global_update!(model::AbstractGPModel{T,L,<:NumericalVI}, state) where {T,L}
    opt_state = map(model.f, state.opt_state) do gp, opt_state
        state_μ, Δμ = Optimisers.apply(
            model.inference.optimiser, opt_state.state_μ, mean(gp), opt_state.∇η₁
        )
        state_Σ, ΔΣ = Optimisers.apply(
            model.inference.optimiser, opt_state.state_Σ, cov(gp).data, opt_state.∇η₂
        )
        gp.post.μ .+= Δμ
        α = 1.0
        while !isposdef(cov(gp) + α * Symmetric(ΔΣ)) && α > 1e-8
            α *= 0.5
        end
        if α > 1e-8
            gp.post.Σ .= cov(gp) + α * Symmetric(ΔΣ)
        else
            @warn "α was too small for update" maxlog = 10
        end
        merge(opt_state, (; state_μ, state_Σ))
    end
    return merge(state, (; opt_state))
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

function ELBO(m::AbstractGPModel{T,L,<:NumericalVI}, state, y) where {T,L}
    tot = zero(T)
    tot +=
        ρ(m) * expec_loglikelihood(
            likelihood(m),
            inference(m),
            y,
            mean_f(m, state.kernel_matrices),
            var_f(m, state.kernel_matrices),
        )
    tot -= GaussianKL(m, state)
    tot -= extraKL(m, state)
    return tot
end
