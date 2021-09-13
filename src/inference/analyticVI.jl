mutable struct AnalyticVI{T,O<:AVIOptimizer} <: VariationalInference{T}
    ϵ::T # Convergence criteria
    n_iter::Int # Number of steps performed
    stoch::Bool # Flag for stochastic optimization
    batchsize::Int # Size of mini-batches
    ρ::T # Scaling coeff. for stoch. opt.
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    vi_opt::O # Local optimizer for the variational parameters

    function AnalyticVI{T}(ϵ::T, optimiser, batchsize::Int, stoch::Bool) where {T}
        vi_opt = AVIOptimizer(optimiser)
        return new{T,typeof(vi_opt)}(ϵ, 0, stoch, batchsize, one(T), true, vi_opt)
    end
end

"""
    AnalyticVI(;ϵ::T=1e-5)

Variational Inference solver for conjugate or conditionally conjugate likelihoods (non-gaussian are made conjugate via augmentation)
All data is used at each iteration (use [`AnalyticSVI`](@ref) for updates using minibatches)

## Keywords arguments
- `ϵ::Real` : convergence criteria
"""
AnalyticVI

"""
    AnalyticSVI(nMinibatch::Int; ϵ::T=1e-5, optimiser=RobbinsMonro())

Stochastic Variational Inference solver for conjugate or conditionally conjugate likelihoods (non-gaussian are made conjugate via augmentation).
See [`AnalyticVI`](@ref) for reference

## Arguments

- `nMinibatch::Integer` : Number of samples per mini-batches

## Keywords arguments

- `ϵ::T` : convergence criteria
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `RobbinsMonro()` (ρ=(τ+iter)^-κ)
"""
AnalyticSVI

function AnalyticVI(; ϵ::T=1e-5) where {T<:Real}
    return AnalyticVI{T}(ϵ, Descent(1.0), 0, false)
end

function AnalyticSVI(
    nMinibatch::Integer; ϵ::T=1e-5, optimiser=RobbinsMonro()
) where {T<:Real}
    return AnalyticVI{T}(ϵ, optimiser, nMinibatch, true)
end

function Base.show(io::IO, inference::AnalyticVI)
    return print(
        io, "Analytic$(is_stochastic(inference) ? " Stochastic" : "") Variational Inference"
    )
end

### Generic method for variational updates using analytical formulas ###
# Single output version
@traitfn function variational_updates(
    m::TGP, state, y
) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
    local_vars = local_updates!(
        state.local_vars,
        likelihood(m),
        y,
        mean_f(m, state.kernel_matrices),
        var_f(m, state.kernel_matrices),
    )
    natural_gradient!.(
        m.f,
        ∇E_μ(m, y, local_vars),
        ∇E_Σ(m, y, local_vars),
        ρ(inference(m)),
        get_opt(inference(m)),
        Zviews(m),
        state.kernel_matrices,
    )
    state = global_update!(m, state)
    return merge(state, (; local_vars))
end
# Multioutput version
@traitfn function variational_updates(
    m::TGP, state, y
) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};IsMultiOutput{TGP}}
    local_vars =
        local_updates!.(
            state.local_vars,
            likelihood(m),
            y,
            mean_f(m, state.kernel_matrices),
            var_f(m, state.kernel_matrices),
        ) # Compute the local updates given the expectations of f
    natural_gradient!.(
        m.f,
        ∇E_μ(m, y, local_vars),
        ∇E_Σ(m, y, local_vars),
        ρ(inference(m)),
        get_opt(inference(m)),
        Zviews(m),
        state.kernel_matrices,
        state.vi_opt_state,
    ) # Compute the natural gradients of u given the weighted sum of the gradient of f
    state = global_update!(m, state) # Update η₁ and η₂
    return merge(state, (; local_vars))
end

# Wrappers for tuple of 1 element,
# when multiple f are needed, these methods can be simply overloaded 
function local_updates!(
    local_vars,
    l::AbstractLikelihood,
    y,
    μ::Tuple{<:AbstractVector{T}},
    diagΣ::Tuple{<:AbstractVector{T}},
) where {T}
    return local_updates!(local_vars, l, y, first(μ), first(diagΣ))
end

function expec_loglikelihood(
    l::AbstractLikelihood,
    i::AnalyticVI,
    y,
    μ::Tuple{<:AbstractVector{T}},
    diagΣ::Tuple{<:AbstractVector{T}},
    state,
) where {T}
    return expec_loglikelihood(l, i, y, first(μ), first(diagΣ), state)
end

# Coordinate ascent updates on the natural parameters ##
function natural_gradient!(
    gp::Union{VarLatent{T},TVarLatent{T}},
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    ::Real,
    ::AVIOptimizer,
    X::AbstractVector,
    kernel_matrices,
    vi_opt_state,
) where {T}
    K = kernel_matrices.K
    gp.post.η₁ .= ∇E_μ .+ K \ pr_mean(gp, X)
    gp.post.η₂ .= -Symmetric(Diagonal(∇E_Σ) + 0.5 * inv(K))
    return gp
end

# Computation of the natural gradient for the natural parameters
function natural_gradient!(
    gp::SparseVarLatent{T},
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    ρ::Real,
    ::AVIOptimizer,
    Z::AbstractVector,
    kernel_matrices,
    vi_opt_state,
) where {T}
    K, κ = kernel_matrices.K, kernel_matrices.κ
    vi_opt_state.∇η₁ .= ∇η₁(∇E_μ, ρ, κ, K, pr_mean(gp, Z), nat1(gp))
    vi_opt_state.∇η₂ .= ∇η₂(∇E_Σ, ρ, κ, K, nat2(gp))
    return gp
end

# Gradient of on the first natural parameter η₁ = Σ⁻¹μ
function ∇η₁(
    ∇μ::AbstractVector{T},
    ρ::Real,
    κ::AbstractMatrix{T},
    K::Cholesky{T,Matrix{T}},
    μ₀::AbstractVector,
    η₁::AbstractVector{T},
) where {T<:Real}
    return transpose(κ) * (ρ * ∇μ) + (K \ μ₀) - η₁
end

# Gradient of on the second natural parameter η₂ = -0.5Σ⁻¹
function ∇η₂(
    θ::AbstractVector{T},
    ρ::Real,
    κ::AbstractMatrix{<:Real},
    K::Cholesky{T,Matrix{T}},
    η₂::Symmetric{T,Matrix{T}},
) where {T<:Real}
    return -(ρκdiagθκ(ρ, κ, θ) + 0.5 * inv(K)) - η₂
end

# Natural gradient for the ONLINE model (OSVGP) #
function natural_gradient!(
    gp::OnlineVarLatent{T},
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    ::Real,
    ::AVIOptimizer,
    Z::AbstractVector,
    kernel_matrices,
    vi_opt_state,
) where {T}
    K = kernel_matrices.K
    κ = kernel_matrices.κ
    κₐ = kernel_matrices.κₐ
    previous_gp = vi_opt_state.previous_gp
    prevη₁ = previous_gp.η₁
    invDₐ = previous_gp.invDₐ
    gp.post.η₁ = K \ pr_mean(gp, Z) + transpose(κ) * ∇E_μ + transpose(κₐ) * prevη₁
    gp.post.η₂ =
        -Symmetric(ρκdiagθκ(1.0, κ, ∇E_Σ) + 0.5 * transpose(κₐ) * invDₐ * κₐ + 0.5 * inv(K))
    return gp
end

# Once the natural parameters have been updated we need to convert back to μ and Σ
@traitfn function global_update!(
    model::TGP, state
) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};IsFull{TGP}}
    global_update!.(model.f)
    return state
end

global_update!(gp::VarLatent, ::AVIOptimizer, ::AnalyticVI) = global_update!(gp)

function global_update!(model::OnlineSVGP, state)
    global_update!.(model.f)
    return state
end

#Update of the natural parameters and conversion from natural to standard distribution parameters
@traitfn function global_update!(
    model::TGP, state
) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};!IsFull{TGP}}
    state = global_update!.(model.f, inference(model).vi_opt, inference(model), state)
    return state
end

function global_update!(gp::SparseVarLatent, opt::AVIOptimizer, i::AnalyticVI, state)
    if is_stochastic(i)
        if haskey(state, :vi_opt_state)
            Δ₁, state_η₁ = Optimisers.apply(
                opt.optimiser, vi_opt_state.state_η₁, nat1(gp), vi_opt_state.∇η₁
            )
            Δ₂, state_η₂ = Optimisers.apply(
                opt.optimiser, vi_opt_state.state_η₂, nat2(gp).data, vi_opt_state.∇η₂
            )
        end
        gp.post.η₁ .+= Δ₁
        gp.post.η₂ .= Symmetric(Δ₂) + nat2(gp)
        vi_opt_state = merge(vi_opt_state, (; state_η₁, state_η₂))
    else
        gp.post.η₁ .+= vi_opt_state.∇η₁
        gp.post.η₂ .= Symmetric(vi_opt_state.∇η₂ + nat2(gp))
    end
    global_update!(gp)
    return vi_opt_state
end

function global_update!(gp::OnlineVarLatent, ::Any, ::Any)
    return global_update!(gp)
end

# Computation of the ELBO for all model
# There are 4 parts : the (augmented) log-likelihood, the Gaussian KL divergence
# the augmented variable KL divergence, some eventual additional part (like in the online case)
@traitfn function ELBO(
    model::TGP, y, state
) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
    tot = zero(T)
    tot +=
        ρ(inference(model)) * expec_loglikelihood(
            likelihood(model),
            inference(model),
            y,
            mean_f(model, state.kernel_matrices),
            var_f(model, state.kernel_matrices),
            state,
        )
    tot -= GaussianKL(model, state)
    tot -= Zygote.@ignore(
        ρ(inference(model)) * AugmentedKL(likelihood(model), yview(model))
    )
    return tot -= extraKL(model)
end

# Multi-output version
@traitfn function ELBO(
    model::TGP, state
) where {T,L,TGP<:AbstractGPModel{T,L,<:AnalyticVI};IsMultiOutput{TGP}}
    tot = zero(T)
    tot += sum(
        ρ(inference(model)) .*
        expec_loglikelihood.(
            likelihood(model),
            inference(model),
            yview(model),
            mean_f(model),
            var_f(model),
            state.local_vars,
        ),
    )
    tot -= GaussianKL(model)
    tot -= Zygote.@ignore(
        sum(ρ(inference(model)) .* AugmentedKL.(likelihood(model), yview(model)))
    )
    return tot
end
