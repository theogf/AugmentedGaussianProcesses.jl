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





mutable struct AnalyticVI{T<:Real,N} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nMinibatch::Int64 #Size of mini-batches
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,AVIOptimizer}
    MBIndices::Vector{Int64} #Indices of the minibatch
    xview::SubArray{T,2,Matrix{T}}#,Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    yview::SubArray

    function AnalyticVI{T}(ϵ::T,optimizer::Optimizer,Stochastic::Bool) where T
        return new{T,1}(ϵ,0,Stochastic,0,0,1.0,false,MBIndices,ρ,true,(AVIOptimizer(0,optimizer)))
    end
    function AnalyticVI{T}(ϵ::T,Stochastic::Bool,nFeatures::Int,nSamples::Int,nMinibatch::Int,nLatent::Int,optimizer::Optimizer)
        vi_opts = ntuple(_->AVIOptimizer{T}(nFeatures,optimizer))
        new{T,nLatent}(ϵ,0,Stochastic,nSamples,nMinibatch,nSamples/nMinibatch,true,vi_opts,collect(1:nMinibatch))
    end
end


function AnalyticVI(;ϵ::T=1e-5) where {T<:Real}
    AnalyticVI{Float64}(ϵ,VanillaGradDescent(η=1.0),false)
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
    AnalyticVI{T}(ϵ,optimizer,true)
end

function Base.show(io::IO,inference::AnalyticVI{T}) where T
    print(io,"Analytic$(inference.Stochastic ? " Stochastic" : "") Variational Inference")
end


"""Initialize the final version of the inference object"""
function tuple_inference(i::TInf,nLatent::Integer,nFeatures::Integer,nSamples::Integer,nMinibatch::Integer) where {TInf <: AnalyticVI{T} where T
    return TInf(i.ϵ,i.Stochastic,nFeatures,nSamples,nMinibatch,nLatent,i.vi_opt[1].optimizer)
end

"""Generic method for variational updates using analytical formulas"""
function variational_updates!(model::AbstractGP{T,L,AnalyticVI{T}}) where {T,L}
    local_updates!(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model))
    natural_gradient!.(model.likelihood,model.inference,model.inference.vi_opt,[get_y(model)],model.f)
    global_update!(model)
end

#Coordinate ascent updates on the natural parameters
function natural_gradient!(l::Likelihood,i::AnalyticVI,opt::AVIOptimizer,y::AbstractVector,gp::_VGP{T}) where {T,L}
    gp.η₁ .= ∇E_μ(l,opt,y) .+ gp.K \ gp.μ₀
    gp.η₂ .= -Symmetric(Diagonal{T}(∇E_Σ(l,opt,y)).+0.5.*inv(gp.K))
end

#Computation of the natural gradient for the natural parameters
function natural_gradient!(l::Likelihood,i::AnalyticVI,opt::AVIOptimizer,y::AbstractVector,gp::_SVGP{T}) where {T,L}
    opt.∇η₁ .= ∇η₁(∇E_μ(l,opt,y),i.ρ,gp.κ,gp.K,gp.μ₀,gp.η₁)
    opt.∇η₂ .= ∇η₂(∇E_Σ(l,opt,y),i.ρ,gp.κ,gp.K,gp.η₂)
end

function ∇η₁(∇μ::AbstractVector{T},ρ::Real,κ::AbstractMatrix{T},K::PDMat{T,Matrix{T}},μ₀::PriorMean,η₁::AbstractVector{T}) where {T <: Real}
    transpose(κ)*(ρ*∇μ) + (K \ μ₀) - η₁
end

function ∇η₂(θ::AbstractVector{T},ρ::Real,κ::AbstractMatrix{<:Real},K::PDMat{T,Matrix{T}},η₂::Symmetric{T,Matrix{T}}) where {T<:Real}
    -(ρκdiagθκ(ρ,κ,θ)+0.5.*inv(K)) - η₂
end

function global_update!(model::VGP{T,L,<:AnalyticVI}) where {T,L}
    global_update!.(model.f)
end

#Conversion from natural to standard distribution parameters
function global_update!(gp::Abstract_GP) where {T,L}
    gp.Σ .= -0.5.*inv(gp.η₂)
    mul!(gp.μ,gp.Σ,gp.η₁)
end

#Update of the natural parameters and conversion from natural to standard distribution parameters
function global_update!(model::SVGP{T,L,<:AnalyticVI}) where {T,L}
    if model.inference.Stochastic
        for (f,vi_opt) in zip(model.f,model.inference.vi_opt)
            Δ = GradDescent.update(vi_opt.optimizer,vcat(vi_opt.∇η₁,vi_opt.∇η₂[:]))
            f.η₁ .+= Δ[1:model.nFeatures]
            f.η₂ .= Symmetric(f.η₂ + reshape(Δ[(model.nFeatures+1):end],model.nFeatures,model.nFeatures))
        end
    else
        for f,vi_opt in zip(model.f,model.inference.vi_opt)
            f.η₁ .+= vi_opt.∇η₁
            f.η₂ .= Symmetric(vi_opt.∇η₂ .+ f.η₂)
        end
    end
    global_update!.(model.f)
end
