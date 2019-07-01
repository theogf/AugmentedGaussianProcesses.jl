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

function local_updates!(model::VGP{<:TemplateLikelihood,<:AnalyticVI})
end

function local_updates!(model::SVGP{<:TemplateLikelihood,<:AnalyticVI})
end

function sample_local!(model::VGP{<:TemplateLikelihood,<:GibbsSampling})
    return nothing
end

### Natural Gradient Section ###

function expec_μ(model::VGP{<:TemplateLikelihood,<:AnalyticVI},index::Integer)
end

function ∇μ(model::VGP{<:TemplateLikelihood})
end

function expec_μ(model::SVGP{<:TemplateLikelihood,<:AnalyticVI},index::Integer)
end

function ∇μ(model::SVGP{<:TemplateLikelihood})
end

function expec_Σ(model::AbstractGP{<:TemplateLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{<:TemplateLikelihood})
    return model.likelihood.θ
end

### ELBO Section ###

function ELBO(model::AbstractGP{<:TemplateLikelihood,<:AnalyticVI})
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{<:TemplateLikelihood,<:AnalyticVI})
    tot = 0.0
    return tot
end

function expecLogLikelihood(model::SVGP{<:TemplateLikelihood,<:AnalyticVI})
    tot = 0.0
    return model.inference.ρ*tot
end


### Gradient Section ###

function gradpdf(::TemplateLikelihood,y::Int,f::T) where {T<:Real}
end

function hessiandiagpdf(::TemplateLikelihood,y::Int,f::T) where {T<:Real}
end
