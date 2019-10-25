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
mutable struct AnalyticVI{T<:Real} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer::Optimizer #Learning rate for stochastic updates
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::Vector{Int64} #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T} #Stored as a matrix since symmetric sums do not help for the moment WARNING
    x::SubArray{T,2,Matrix{T}}#,Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    y::SubArray
    function AnalyticVI{T}(ϵ::T,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
    function AnalyticVI{T}(ϵ::T,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool,∇η₁::AbstractVector,
    ∇η₂::AbstractMatrix) where T
        return new{T}(ϵ,nIter,optimizer,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag,∇η₁,∇η₂)
    end
end


function AnalyticVI(;ϵ::T=1e-5) where {T<:Real}
    AnalyticVI{Float64}(ϵ,0,VanillaGradDescent(η=1.0),false,1,1,[1],1.0,true)
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
    AnalyticVI{T}(ϵ,0,optimizer,true,1,nMinibatch,1:nMinibatch,1.0,true)
end

function Base.show(io::IO,inference::AnalyticVI{T}) where T
    print(io,"Analytic$(inference.Stochastic ? " Stochastic" : "") Variational Inference")
end


"""Initialize the final version of the inference object"""
function init_inference(inference::AnalyticVI{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    inference.ρ = nSamples/nSamplesUsed
    inference.∇η₁ = zeros(T,nFeatures);
    inference.∇η₂ = Matrix{T}(undef,nFeatures,nFeatures)
    return inference
end

"""Generic method for variational updates using analytical formulas"""
function variational_updates!(model::AbstractGP{T,L,AnalyticVI{T}}) where {T,L}
    local_updates!(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model))
    natural_gradient!(get_y(model),model.likelihood,model.inference,model.f...)
    global_update!(model)
end

#Coordinate ascent updates on the natural parameters
function natural_gradient!(y::AbstractVector,l::Likelihood,i::AnalyticVI,gp::_VGP{T}) where {T,L}
    gp.η₁ .= ∇E_μ(gp,l,y) .+ gp.K \ gp.μ₀
    gp.η₂ .= -Symmetric(Diagonal{T}(∇E_Σ(gp,l,y)).+0.5.*inv(gp.K))
end

function natural_gradient!(y::AbstractVector,l::Likelihood,i::AnalyticVI,gps::Vararg{_VGP{T},N}) where {T,N}

end
#Computation of the natural gradient for the natural parameters

function natural_gradient!(y::AbstractVector,l::Likelihood,i::AnalyticVI,gp::_SVGP{T}) where {T,L}
    i.∇η₁ .= ∇η₁(∇E_μ(gp,l,y),i.ρ,gp.κ,gp.K,gp.μ₀,gp.η₁)
    i.∇η₂ .= ∇η₂(∇E_Σ(gp,l,y),i.ρ,gp.κ,gp.K,gp.η₂)
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
        for f in model.f
            Δ = GradDescent.update(model.inference.optimizer,vcat(model.inference.∇η₁,model.inference.∇η₂[:]))
            f.η₁ .+= Δ[1:model.nFeatures]
            f.η₂ .= Symmetric(f.η₂ + reshape(Δ[(model.nFeatures+1):end],model.nFeatures,model.nFeatures))
            # model.η₁ .= model.η₁ .+ GradDescent.update.(model.inference.optimizer_η₁,model.inference.∇η₁)
            # model.η₂ .= Symmetric.(model.η₂ .+ GradDescent.update.(model.inference.optimizer_η₂,model.inference.∇η₂))
        end
    else
        for f in model.f
            f.η₁ .+= model.inference.∇η₁
            f.η₂ .= Symmetric(model.inference.∇η₂ .+ f.η₂)
        end
    end
    global_update!.(model.f)
end
