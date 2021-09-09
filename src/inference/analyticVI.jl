mutable struct AnalyticVI{T,N,Tx,Ty} <: VariationalInference{T}
    ϵ::T # Convergence criteria
    nIter::Integer # Number of steps performed
    stoch::Bool # Flag for stochastic optimization
    nSamples::Int # Total number of samples
    nMinibatch::Int # Size of mini-batches
    ρ::T # Scaling coeff. for stoch. opt.
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    vi_opt::NTuple{N,AVIOptimizer} # Local optimizers for the variational parameters
    MBIndices::Vector{Int} # Indices of the minibatch

    function AnalyticVI{T}(ϵ::T, optimiser, nMinibatch::Int, Stochastic::Bool) where {T}
        return new{T,1,Vector{T},Vector{T}}(
            ϵ, 0, Stochastic, 0, nMinibatch, one(T), true, (AVIOptimizer{T}(0, optimiser),)
        )
    end
    function AnalyticVI{T}(
        ϵ::Real,
        Stochastic::Bool,
        nFeatures::Vector{<:Int},
        nSamples::Int,
        nMinibatch::Int,
        nLatent::Int,
        optimiser,
        xview::Tx,
        yview::Ty,
    ) where {T,Tx,Ty}
        vi_opts = ntuple(i -> AVIOptimizer{T}(nFeatures[i], optimiser), nLatent)
        return new{T,nLatent,Tx,Ty}(
            ϵ,
            0,
            Stochastic,
            nSamples,
            nMinibatch,
            convert(T, nSamples / nMinibatch),
            true,
            vi_opts,
            1:nMinibatch,
            xview,
            yview,
        )
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
        io, "Analytic$(isStochastic(inference) ? " Stochastic" : "") Variational Inference"
    )
end

function tuple_inference(
    i::AnalyticVI{T},
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Int,
    nMinibatch::Int,
    xview,
    yview,
) where {T}
    return AnalyticVI{T}(
        conv_crit(i),
        isStochastic(i),
        nFeatures,
        nSamples,
        nMinibatch,
        nLatent,
        i.vi_opt[1].optimiser,
        xview,
        yview,
    )
end

### Generic method for variational updates using analytical formulas ###
# Single output version
@traitfn function variational_updates(
    local_vars, m::TGP, kernel_matrices, y
) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
    local_vars = local_updates!(
        local_vars, likelihood(m), y, mean_f(m, kernel_matrices), var_f(m, kernel_matrices),
    )
    natural_gradient!.(
        ∇E_μ(m, y, local_vars),
        ∇E_Σ(m, y, local_vars),
        getρ(inference(m)),
        get_opt(inference(m)),
        Zviews(m),
        kernel_matrices,
        m.f,
    )
    global_update!(m)
    return local_vars
end
# Multioutput version
@traitfn function variational_updates(
    local_vars, m::TGP, kernel_matrices, y
) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};IsMultiOutput{TGP}}
    local_vars = local_updates!.(likelihood(m), y, mean_f(m, kernel_matrices), var_f(m, kernel_matrices)) # Compute the local updates given the expectations of f
    natural_gradient!.(
        ∇E_μ(m, y, local_vars), ∇E_Σ(m, y, local_vars), getρ(inference(m)), get_opt(inference(m)), Zviews(m), m.f
    ) # Compute the natural gradients of u given the weighted sum of the gradient of f
    return global_update!(m) # Update η₁ and η₂
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
) where {T}
    return expec_loglikelihood(l, i, y, first(μ), first(diagΣ))
end

# Coordinate ascent updates on the natural parameters ##
function natural_gradient!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    ::Real,
    ::AVIOptimizer,
    X::AbstractVector,
    kernel_matrices,
    gp::Union{VarLatent{T},TVarLatent{T}},
) where {T}
    gp.post.η₁ .= ∇E_μ .+ kernel_matrices.K \ pr_mean(gp, X)
    return gp.post.η₂ .= -Symmetric(Diagonal(∇E_Σ) + 0.5 * inv(kernel_matrices.K))
end

# Computation of the natural gradient for the natural parameters
function natural_gradient!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    ρ::Real,
    opt::AVIOptimizer,
    Z::AbstractVector,
    kernel_matrices,
    gp::SparseVarLatent{T},
) where {T}
    opt.∇η₁ .= ∇η₁(∇E_μ, ρ, kernel_matrices.κ, kernel_matrices.K, pr_mean(gp, Z), nat1(gp))
    return opt.∇η₂ .= ∇η₂(∇E_Σ, ρ, kernel_matrices.κ, kernel_matrices.K, nat2(gp))
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
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    ::Real,
    ::AVIOptimizer,
    Z::AbstractVector,
    kernel_matrices,
    gp::OnlineVarLatent{T},
) where {T}
    gp.post.η₁ =
        kernel_matrices.K \ pr_mean(gp, Z) + transpose(kernel_matrices.κ) * ∇E_μ + transpose(kernel_matrices.κₐ) * gp.prevη₁
    return gp.post.η₂ =
        -Symmetric(
            ρκdiagθκ(1.0, kernel_matrices.κ, ∇E_Σ) +
            0.5 * transpose(kernel_matrices.κₐ) * gp.invDₐ * kernel_matrices.κₐ +
            0.5 * inv(kernel_matrices.K),
        )
end

# Once the natural parameters have been updated we need to convert back to μ and Σ
@traitfn function global_update!(
    model::TGP
) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};IsFull{TGP}}
    return global_update!.(model.f)
end

global_update!(gp::VarLatent, ::AVIOptimizer, ::AnalyticVI) = global_update!(gp)

global_update!(model::OnlineSVGP) = global_update!.(model.f)
global_update!(gp::OnlineVarLatent, ::Any, ::Any) = global_update!(gp)

#Update of the natural parameters and conversion from natural to standard distribution parameters
@traitfn function global_update!(
    model::TGP
) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};!IsFull{TGP}}
    return global_update!.(model.f, inference(model).vi_opt, inference(model))
end

function global_update!(gp::SparseVarLatent, opt::AVIOptimizer, i::AnalyticVI)
    if isStochastic(i)
        Δ₁ = Optimisers.apply(opt.optimiser, opt.η₁_state, nat1(gp), opt.∇η₁)
        Δ₂ = Optimisers.apply(opt.optimiser, opt.η₂_state, nat2(gp).data, opt.∇η₂)
        gp.post.η₁ .+= Δ₁
        gp.post.η₂ .= Symmetric(Δ₂) + nat2(gp)
    else
        gp.post.η₁ .+= opt.∇η₁
        gp.post.η₂ .= Symmetric(opt.∇η₂ + nat2(gp))
    end
    return global_update!(gp)
end

# Computation of the ELBO for all model
# There are 4 parts : the (augmented) log-likelihood, the Gaussian KL divergence
# the augmented variable KL divergence, some eventual additional part (like in the online case)
@traitfn function ELBO(
    model::TGP
) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
    tot = zero(T)
    tot +=
        getρ(inference(model)) * expec_loglikelihood(
            likelihood(model), inference(model), yview(model), mean_f(model), var_f(model)
        )
    tot -= GaussianKL(model)
    tot -= Zygote.@ignore(
        getρ(inference(model)) * AugmentedKL(likelihood(model), yview(model))
    )
    return tot -= extraKL(model)
end

# Multi-output version
@traitfn function ELBO(
    model::TGP
) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};IsMultiOutput{TGP}}
    tot = zero(T)
    tot += sum(
        getρ(inference(model)) .*
        expec_loglikelihood.(
            likelihood(model), inference(model), yview(model), mean_f(model), var_f(model)
        ),
    )
    tot -= GaussianKL(model)
    tot -= Zygote.@ignore(
        sum(getρ(inference(model)) .* AugmentedKL.(likelihood(model), yview(model)))
    )
    return tot
end
