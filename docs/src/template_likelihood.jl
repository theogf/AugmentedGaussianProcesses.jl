"""
**Template Likelihood**

Template file for likelihood creation

```julia
TemplateLikelihood()
```
See all functions you need to implement
---


"""
struct TemplateLikelihood{T<:Real} <: Likelihood{T}
    θ::LatentArray{AbstractVector{T}}
    function TemplateLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function TemplateLikelihood{T}(θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
        new{T}(θ)
    end
end

function TemplateLikelihood()
    TemplateLikelihood{Float64}()
end

function init_likelihood(likelihood::TemplateLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        TemplateLikelihood{T}([zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        TemplateLikelihood{T}()
    end
end

function pdf(l::TemplateLikelihood,y::Real,f::Real)

end

function Base.show(io::IO,model::TemplateLikelihood{T}) where T
    print(io,"Template Likelihood")
end

function compute_proba(l::TemplateLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        pred[i]  = 0.0
    end
    return pred
end

### Local Updates Section ###

function local_updates!(model::VGP{T,<:TemplateLikelihood,<:AnalyticVI}) where {T}
end

function local_updates!(model::SVGP{T,<:TemplateLikelihood,<:AnalyticVI}) where {T}
end

function sample_local!(model::VGP{T,<:TemplateLikelihood,<:GibbsSampling}) where {T}
    return nothing
end

### Natural Gradient Section ###

function expec_μ(model::VGP{T,<:TemplateLikelihood,<:AnalyticVI},index::Integer) where {T}
end

function ∇μ(model::VGP{T,<:TemplateLikelihood}) where {T}
end

function expec_μ(model::SVGP{T,<:TemplateLikelihood,<:AnalyticVI},index::Integer) where {T}
end

function ∇μ(model::SVGP{T,<:TemplateLikelihood}) where {T}
end

function expec_Σ(model::AbstractGP{T,<:TemplateLikelihood,<:AnalyticVI},index::Integer) where {T}
    return model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{T,<:TemplateLikelihood}) where {T}
    return model.likelihood.θ
end

### ELBO Section ###

function ELBO(model::AbstractGP{T,<:TemplateLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{T,<:TemplateLikelihood,<:AnalyticVI}) where {T}
    tot = 0.0
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:TemplateLikelihood,<:AnalyticVI}) wher {T}
    tot = 0.0
    return model.inference.ρ*tot
end


### Gradient Section ###

function gradpdf(::TemplateLikelihood,y::Int,f::T) where {T<:Real}
end

function hessiandiagpdf(::TemplateLikelihood,y::Int,f::T) where {T<:Real}
end
