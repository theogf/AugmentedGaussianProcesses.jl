"""
**AnalyticVI**

Variational Inference solver for conjugate or conditionally conjugate likelihoods (non-gaussian are made conjugate via augmentation)
All data is used at each iteration (use AnalyticSVI for Stochastic updates)

```julia
AnalyticVI(;ϵ::T=1e-5)
```
**Keywords arguments**

    - `ϵ::T` : convergence criteria
"""
mutable struct AnalyticVI{T,N} <: VariationalInference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nMinibatch::Int64 #Size of mini-batches
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,AVIOptimizer}
    MBIndices::Vector{Int64} #Indices of the minibatch
    xview::SubArray{T,2,Matrix{T}}
    yview::AbstractVector

    function AnalyticVI{T}(ϵ::T,optimizer::Optimizer,nMinibatch::Int,Stochastic::Bool) where {T}
        return new{T,1}(ϵ,0,Stochastic,0,nMinibatch,1.0,true,(AVIOptimizer{T}(0,optimizer),))
    end
    function AnalyticVI{T,1}(ϵ::T,Stochastic::Bool,nFeatures::Int,nSamples::Int,nMinibatch::Int,nLatent::Int,optimizer::Optimizer) where {T}
        vi_opts = ntuple(_->AVIOptimizer{T}(nFeatures,optimizer),nLatent)
        new{T,nLatent}(ϵ,0,Stochastic,nSamples,nMinibatch,nSamples/nMinibatch,true,vi_opts,collect(1:nMinibatch))
    end
end


function AnalyticVI(;ϵ::T=1e-5) where {T<:Real}
    AnalyticVI{Float64}(ϵ,VanillaGradDescent(η=1.0),0,false)
end

"""
**AnalyticSVI**
Stochastic Variational Inference solver for conjugate or conditionally conjugate likelihoods (non-gaussian are made conjugate via augmentation)

```julia
AnalyticSVI(nMinibatch::Integer;ϵ::T=1e-5,optimizer::Optimizer=InverseDecay())
```
    - `nMinibatch::Integer` : Number of samples per mini-batches

**Keywords arguments**

    - `ϵ::T` : convergence criteria
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl) package. Default is `InverseDecay()` (ρ=(τ+iter)^-κ)
"""
function AnalyticSVI(nMinibatch::Integer;ϵ::T=1e-5,optimizer::Optimizer=InverseDecay()) where {T<:Real}
    AnalyticVI{T}(ϵ,optimizer,nMinibatch,true)
end

function Base.show(io::IO,inference::AnalyticVI{T}) where T
    print(io,"Analytic$(inference.Stochastic ? " Stochastic" : "") Variational Inference")
end


## Initialize the final version of the inference object ##
function tuple_inference(i::TInf,nLatent::Integer,nFeatures::Integer,nSamples::Integer,nMinibatch::Integer) where {TInf <: AnalyticVI}
    return TInf(i.ϵ,i.Stochastic,nFeatures,nSamples,nMinibatch,nLatent,i.vi_opt[1].optimizer)
end


## Generic method for variational updates using analytical formulas ##
function variational_updates!(model::AbstractGP{T,L,<:AnalyticVI}) where {T,L}
    local_updates!(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model))
    natural_gradient!.(
        ∇E_μ(model.likelihood,model.inference.vi_opt[1],get_y(model)),
        ∇E_Σ(model.likelihood,model.inference.vi_opt[1],get_y(model)),
        model.inference,model.inference.vi_opt,get_Z(model),model.f)
    global_update!(model)
end

function variational_updates!(model::MOSVGP{T,L,<:AnalyticVI}) where {T,L}
    local_updates!.(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model)) # Compute the local updates given the expectations of f
    # for i in 1:model.nLatent
    #     local_updates!.(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model)) # Compute the local updates given the expectations of f
    #     natural_gradient!(∇E_μ(model)[i],∇E_Σ(model)[i],model.inference,model.inference.vi_opt[i],get_Z(model)[i],model.f[i])
    #     global_update!(model) # Update η₁ and η₂
    # end
    natural_gradient!.(∇E_μ(model), ∇E_Σ(model), model.inference,
        model.inference.vi_opt, get_Z(model), model.f) # Compute the natural gradients of u given the weighted sum of the gradient of f
    global_update!(model) # Update η₁ and η₂
end

## Wrappers for tuple of 1 element
local_updates!(l::Likelihood,y,μ::Tuple{<:AbstractVector{T}},Σ::Tuple{<:AbstractVector{T}}) where {T} = local_updates!(l,y,first(μ),first(Σ))
expec_log_likelihood(l::Likelihood,i::AnalyticVI,y,μ::Tuple{<:AbstractVector{T}},Σ::Tuple{<:AbstractVector{T}}) where {T} = expec_log_likelihood(l,i,y,first(μ),first(Σ))

## Coordinate ascent updates on the natural parameters ##
function natural_gradient!(∇E_μ::AbstractVector,∇E_Σ::AbstractVector,i::AnalyticVI,opt::AVIOptimizer,X::AbstractMatrix,gp::_VGP{T}) where {T,L}
    gp.η₁ .= ∇E_μ .+ gp.K \ gp.μ₀(X)
    gp.η₂ .= -Symmetric(Diagonal(∇E_Σ) + 0.5 * inv(gp.K).mat)
end

#Computation of the natural gradient for the natural parameters
function natural_gradient!(∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::AnalyticVI,opt::AVIOptimizer,Z::AbstractMatrix,gp::_SVGP{T}) where {T<:Real,L}
    opt.∇η₁ .= ∇η₁(∇E_μ, i.ρ, gp.κ, gp.K, gp.μ₀(Z), gp.η₁)
    opt.∇η₂ .= ∇η₂(∇E_Σ, i.ρ, gp.κ, gp.K, gp.η₂)
end

function ∇η₁(∇μ::AbstractVector{T},ρ::Real,κ::AbstractMatrix{T},K::PDMat{T,Matrix{T}},μ₀::AbstractVector,η₁::AbstractVector{T}) where {T <: Real}
    transpose(κ)*(ρ * ∇μ) + (K \ μ₀) - η₁
end

function ∇η₂(θ::AbstractVector{T},ρ::Real,κ::AbstractMatrix{<:Real},K::PDMat{T,Matrix{T}},η₂::Symmetric{T,Matrix{T}}) where {T<:Real}
    -(ρκdiagθκ(ρ, κ, θ) + 0.5 * inv(K).mat) - η₂
end

function natural_gradient!(∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::AnalyticVI,opt::AVIOptimizer,Z::AbstractMatrix,gp::_OSVGP{T}) where {T}
    gp.η₁ .= gp.K \ gp.μ₀(Z) + transpose(gp.κ)*∇E_μ + transpose(gp.κₐ)*gp.prevη₁
    gp.η₂ .= -Symmetric(ρκdiagθκ(1.0,gp.κ,∇E_Σ)+0.5*transpose(gp.κₐ)*gp.invDₐ*gp.κₐ+0.5*inv(gp.K))
end

global_update!(model::VGP{T,L,<:AnalyticVI}) where {T,L} = global_update!.(model.f)

global_update!(gp::_VGP,opt::AVIOptimizer,i::AnalyticVI) = global_update!(gp)

global_update!(model::OnlineSVGP) = global_update!.(model.f)
global_update!(gp::_OSVGP,opt,i) = global_update!(gp)

#Update of the natural parameters and conversion from natural to standard distribution parameters
function global_update!(model::Union{SVGP{T,L,TInf},MOSVGP{T,L,TInf}}) where {T,L,TInf<:AnalyticVI}
    global_update!.(model.f,model.inference.vi_opt,model.inference)
end

function global_update!(gp::_SVGP,opt::AVIOptimizer,i::AnalyticVI)
    if isstochastic(i)
        Δ = GradDescent.update(opt.optimizer,vcat(opt.∇η₁,opt.∇η₂[:]))
        gp.η₁ .+= Δ[1:gp.dim]
        gp.η₂ .= Symmetric(gp.η₂ + reshape(Δ[(gp.dim+1):end],gp.dim,gp.dim))
    else
        gp.η₁ .+= opt.∇η₁
        gp.η₂ .= Symmetric(opt.∇η₂ + gp.η₂)
    end
    global_update!(gp)
end


function ELBO(model::AbstractGP{T,L,<:AnalyticVI}) where {T,L}
    tot = zero(T)
    tot += model.inference.ρ*expec_log_likelihood(model.likelihood,model.inference,get_y(model),mean_f(model),diag_cov_f(model))
    tot -= GaussianKL(model)
    tot -= model.inference.ρ*AugmentedKL(model.likelihood,get_y(model))
end
