"""
    AnalyticVI(;ϵ::T=1e-5)

Variational Inference solver for conjugate or conditionally conjugate likelihoods (non-gaussian are made conjugate via augmentation)
All data is used at each iteration (use AnalyticSVI for Stochastic updates)

**Keywords arguments**
    - `ϵ::T` : convergence criteria
"""
mutable struct AnalyticVI{T,N} <: VariationalInference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Vector{Int64}
    nMinibatch::Vector{Int64} #Size of mini-batches
    ρ::Vector{T} #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,AVIOptimizer}
    MBIndices::Vector{Vector{Int64}} #Indices of the minibatch
    xview::Vector
    yview::Vector

    function AnalyticVI{T}(
        ϵ::T,
        optimiser,
        nMinibatch::Int,
        Stochastic::Bool,
    ) where {T}
        return new{T,1}(
            ϵ,
            0,
            Stochastic,
            [0],
            [nMinibatch],
            [one(T)],
            true,
            (AVIOptimizer{T}(0, optimiser),),
        )
    end
    function AnalyticVI{T,1}(
        ϵ::T,
        Stochastic::Bool,
        nFeatures::Vector{<:Int},
        nSamples::Vector{<:Int},
        nMinibatch::Vector{<:Int},
        nLatent::Int,
        optimiser,
    ) where {T}
        vi_opts = ntuple(i -> AVIOptimizer{T}(nFeatures[i], optimiser), nLatent)
        new{T,nLatent}(
            ϵ,
            0,
            Stochastic,
            nSamples,
            nMinibatch,
            T.(nSamples ./ nMinibatch),
            true,
            vi_opts,
            range.(1, nMinibatch, step = 1),
        )
    end
end


function AnalyticVI(; ϵ::T = 1e-5) where {T<:Real}
    AnalyticVI{T}(ϵ, Descent(1.0), 0, false)
end

"""
    AnalyticSVI(nMinibatch::Integer;ϵ::T=1e-5,optimiser=RobbinsMonro())

Stochastic Variational Inference solver for conjugate or conditionally conjugate likelihoods (non-gaussian are made conjugate via augmentation)


- `nMinibatch::Integer` : Number of samples per mini-batches

**Keywords arguments**

    - `ϵ::T` : convergence criteria
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `RobbinsMonro()` (ρ=(τ+iter)^-κ)
"""
function AnalyticSVI(
    nMinibatch::Integer;
    ϵ::T = 1e-5,
    optimiser = RobbinsMonro(),
) where {T<:Real}
    AnalyticVI{T}(ϵ, optimiser, nMinibatch, true)
end

function Base.show(io::IO, inference::AnalyticVI{T}) where {T}
    print(
        io,
        "Analytic$(inference.Stochastic ? " Stochastic" : "") Variational Inference",
    )
end


## Initialize the final version of the inference object ##
function tuple_inference(
    i::TInf,
    nLatent::Integer,
    nFeatures::Vector{<:Integer},
    nSamples::Vector{<:Integer},
    nMinibatch::Vector{<:Integer},
) where {TInf<:AnalyticVI}
    return TInf(
        i.ϵ,
        i.Stochastic,
        nFeatures,
        nSamples,
        nMinibatch,
        nLatent,
        i.vi_opt[1].optimiser,
    )
end


## Generic method for variational updates using analytical formulas ##
@traitfn function variational_updates!(m::TGP) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
    local_updates!(
        m.likelihood,
        get_y(m),
        mean_f(m),
        diag_cov_f(m),
    )
    natural_gradient!.(
        ∇E_μ(m),
        ∇E_Σ(m),
        m.inference.ρ,
        m.inference.vi_opt,
        get_Z(m),
        m.f,
    )
    global_update!(m)
end

@traitfn function variational_updates!(m::TGP) where {T, L, TGP<:AbstractGP{T,L,<:AnalyticVI}; IsMultiOutput{TGP}}
    local_updates!.(
        m.likelihood,
        get_y(m),
        mean_f(m),
        diag_cov_f(m),
    ) # Compute the local updates given the expectations of f
    natural_gradient!.(
        ∇E_μ(m),
        ∇E_Σ(m),
        getρ(m.inference),
        m.inference.vi_opt,
        get_Z(m),
        m.f,
    ) # Compute the natural gradients of u given the weighted sum of the gradient of f
    global_update!(m) # Update η₁ and η₂
end

## Wrappers for tuple of 1 element
local_updates!(
    l::Likelihood,
    y,
    μ::Tuple{<:AbstractVector{T}},
    Σ::Tuple{<:AbstractVector{T}},
) where {T} = local_updates!(l, y, first(μ), first(Σ))

expec_log_likelihood(
    l::Likelihood,
    i::AnalyticVI,
    y,
    μ::Tuple{<:AbstractVector{T}},
    Σ::Tuple{<:AbstractVector{T}},
) where {T} = expec_log_likelihood(l, i, y, first(μ), first(Σ))

## Coordinate ascent updates on the natural parameters ##
function natural_gradient!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    ρ::Real,
    opt::AVIOptimizer,
    X::AbstractMatrix,
    gp::_VGP{T},
) where {T,L}
    gp.η₁ .= ∇E_μ .+ gp.K \ gp.μ₀(X)
    gp.η₂ .= - Symmetric(Diagonal(∇E_Σ) + 0.5 * inv(gp.K).mat)
end

#Computation of the natural gradient for the natural parameters
function natural_gradient!(
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    ρ::Real,
    opt::AVIOptimizer,
    Z::AbstractMatrix,
    gp::_SVGP{T},
) where {T<:Real}
    opt.∇η₁ .= ∇η₁(∇E_μ, ρ, gp.κ, gp.K, gp.μ₀(Z), gp.η₁)
    opt.∇η₂ .= ∇η₂(∇E_Σ, ρ, gp.κ, gp.K, gp.η₂)
end

function ∇η₁(
    ∇μ::AbstractVector{T},
    ρ::Real,
    κ::AbstractMatrix{T},
    K::PDMat{T,Matrix{T}},
    μ₀::AbstractVector,
    η₁::AbstractVector{T},
) where {T<:Real}
    transpose(κ) * (ρ * ∇μ) + (K \ μ₀) - η₁
end

function ∇η₂(
    θ::AbstractVector{T},
    ρ::Real,
    κ::AbstractMatrix{<:Real},
    K::PDMat{T,Matrix{T}},
    η₂::Symmetric{T,Matrix{T}},
) where {T<:Real}
    -(ρκdiagθκ(ρ, κ, θ) + 0.5 * inv(K).mat) - η₂
end

## Natural gradient for the ONLINE model (OSVGP) ##
function natural_gradient!(
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    ρ::Real,
    opt::AVIOptimizer,
    Z::AbstractMatrix,
    gp::_OSVGP{T},
) where {T}
    gp.η₁ =
        gp.K \ gp.μ₀(Z) + transpose(gp.κ) * ∇E_μ + transpose(gp.κₐ) * gp.prevη₁
    gp.η₂ =
        -Symmetric(
            ρκdiagθκ(1.0, gp.κ, ∇E_Σ) +
            0.5 * transpose(gp.κₐ) * gp.invDₐ * gp.κₐ +
            0.5 * inv(gp.K),
        )
end

@traitfn function global_update!(model::TGP) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};IsFull{TGP}}
    global_update!.(model.f)
end

global_update!(gp::_VGP, opt::AVIOptimizer, i::AnalyticVI) = global_update!(gp)

global_update!(model::OnlineSVGP) = global_update!.(model.f)
global_update!(gp::_OSVGP, opt, i) = global_update!(gp)

#Update of the natural parameters and conversion from natural to standard distribution parameters
@traitfn function global_update!(model::TGP) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};!IsFull{TGP}}
    global_update!.(model.f, model.inference.vi_opt, model.inference)
end

function global_update!(gp::_SVGP, opt::AVIOptimizer, i::AnalyticVI)
    if isStochastic(i)
        Δ = Optimise.apply!(opt.optimiser, gp.η₁, vcat(opt.∇η₁, opt.∇η₂[:]))
        gp.η₁ .+= Δ[1:gp.dim]
        gp.η₂ .= Symmetric(gp.η₂ + reshape(Δ[(gp.dim+1):end], gp.dim, gp.dim))
    else
        gp.η₁ .+= opt.∇η₁
        gp.η₂ .= Symmetric(opt.∇η₂ + gp.η₂)
    end
    global_update!(gp)
end


@traitfn function ELBO(model::TGP) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};!IsMultiOutput{TGP}}
    tot = zero(T)
    tot +=
        getρ(model.inference) * expec_log_likelihood(
            model.likelihood,
            model.inference,
            get_y(model),
            mean_f(model),
            diag_cov_f(model),
        )
    tot -= GaussianKL(model)
    tot -= getρ(model.inference) * AugmentedKL(model.likelihood, get_y(model))
    tot -= extraKL(model)
end

@traitfn function ELBO(model::TGP) where {T,L,TGP<:AbstractGP{T,L,<:AnalyticVI};IsMultiOutput{TGP}}
    tot = zero(T)
    tot += sum(model.inference.ρ .* expec_log_likelihood.(model.likelihood,model.inference,get_y(model),mean_f(model),diag_cov_f(model)))
    tot -= GaussianKL(model)
    tot -= sum(model.inference.ρ .* AugmentedKL.(model.likelihood,get_y(model)))
end
