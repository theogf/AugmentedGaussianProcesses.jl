"""
**Generic Likelihood**

Template file for likelihood creation

```julia
GenericLikelihood()
```
See all functions you need to implement
---


"""
struct GenericLikelihood{T<:Real} <: Likelihood{T}
    b::T
    c²::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function GenericLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function GenericLikelihood{T}(θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
        new{T}(θ)
    end
end

function GenericLikelihood()
    GenericLikelihood{Float64}()
end

function init_likelihood(likelihood::GenericLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        GenericLikelihood{T}([zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        GenericLikelihood{T}()
    end
end

function C(l::GenericLikelihood{T}) where {T}
    zero(T)
end

function g(l::GenericLikelihood,y::AbstractVector{T}) where {T}
    zero(T)
end

function α(l::GenericLikelihood,y::AbstractVector{T}) where {T}
    zero(T)
end

function β(l::GenericLikelihood,y::AbstractVector{T}) where {T}
    zero(T)
end

function γ(l::GenericLikelihood,y::AbstractVector{T}) where {T}
    zero(y)
end

function φ(l::GenericLikelihood,r::T) where {T}
    zero(r)
end

function ∇φ(l::GenericLikelihood,r::T) where {T}
    zero(r)
end

function pdf(l::GenericLikelihood,y::Real,f::Real)

end

function Base.show(io::IO,model::GenericLikelihood{T}) where T
    print(io,"Template Likelihood")
end

function compute_proba(l::GenericLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        pred[i]  = 0.0
    end
    return pred
end

### Local Updates Section ###

function local_updates!(model::VGP{T,<:GenericLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c² .= broadcast((y,μ,Σ)->α.(model.likelihood,y)-β.(model.likelihood,y).*μ+γ.(model.likelihood,y).*(abs2.(μ)+Σ),model.inference.y,model.μ,diag.(model.Σ))
    model.likelihood.θ .= broadcast(c²->-∇φ.(model.likelihood,c²)./φ.(model.likelihood,c²),model.likelihood.c²)
end

function local_updates!(model::SVGP{T,<:GenericLikelihood,<:AnalyticVI}) where {T}
end

function sample_local!(model::VGP{T,<:GenericLikelihood,<:GibbsSampling}) where {T}
    return nothing
end

### Natural Gradient Section ###

@inline ∇E_μ(model::AbstractGP{T,<:GenericLikelihood,<:GibbsorVI}) where {T} = broadcast((y,θ)->g(model.likelihood,y)+θ.*β(model.likelihood,y),model.inference.y,model.likelihood.θ)
@inline ∇E_μ(model::AbstractGP{T,<:GenericLikelihood,<:GibbsorVI},i::Int) where {T} =  g(model.likelihood,model.inference.y[i])+model.likelihood.θ[i].*β(model.likelihood,model.inference.y[i])
@inline ∇E_Σ(model::AbstractGP{T,<:GenericLikelihood,<:GibbsorVI}) where {T} =
broadcast((y,θ)->θ.*γ(model.likelihood,y),model.inference.y,model.likelihood.θ)
@inline ∇E_Σ(model::AbstractGP{T,<:GenericLikelihood,<:GibbsorVI},i::Int) where {T} = model.likelihood.θ[i].*γ(model.likelihood,model.inference.y[i])

### ELBO Section ###

function ELBO(model::AbstractGP{T,<:GenericLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model) - AugmentedKL(model)
end

function expecLogLikelihood(model::VGP{T,<:GenericLikelihood,<:AnalyticVI}) where {T}
    tot = model.nLatent*model.nSamples*log(C(model.likelihood))
    tot += sum(broadcast((y,μ)->dot(g(model.likelihood,y),μ),model.inference.y,model.μ)
    tot += -sum(broadcast((θ,y,μ,Σ)->dot(θ,α(model.likelihood,y))
                                    - dot(θ,β(model.likelihood,y).*μ)
                                    + dot(θ,γ(model.likelihood,y).*(abs2.(μ)+Σ)),
                                    model.likelihood.θ,model.inference.y,
                                    model.μ,diag.(model.Σ)))
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:GenericLikelihood,<:AnalyticVI}) where {T}
    tot = 0.0
    return model.inference.ρ*tot
end

function AugmentedKL(model::AbstractGP{T,<:GenericLikelihood,<:AnalyticVI}) where {T}
    model.inference.ρ*sum(broadcast((c²,θ)->-sum(c².*θ)-sum(log,φ.(model.likelihood,c²)),model.likelihood.c²,model.likelihood,θ))
end

### Gradient Section ###

function gradpdf(::GenericLikelihood,y::Int,f::T) where {T<:Real}
end

function hessiandiagpdf(::GenericLikelihood,y::Int,f::T) where {T<:Real}
end
