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

function init_likelihood(likelihood::TemplateLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Integer) where T
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

function local_updates!(model::SVGP{<:LogisticLikelihood,<:AnalyticVI})
end

function sample_local!(model::VGP{<:LogisticLikelihood,<:GibbsSampling})
    return nothing
end

### Natural Gradient Section ###

function expec_μ(model::VGP{<:LogisticLikelihood,<:AnalyticVI},index::Integer)
end

function ∇μ(model::VGP{<:LogisticLikelihood})
end

function expec_μ(model::SVGP{<:LogisticLikelihood,<:AnalyticVI},index::Integer)
end

function ∇μ(model::SVGP{<:LogisticLikelihood})
end

function expec_Σ(model::AbstractGP{<:LogisticLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{<:LogisticLikelihood})
    return model.likelihood.θ
end

### ELBO Section ###

function ELBO(model::AbstractGP{<:LogisticLikelihood,<:AnalyticVI})
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{<:LogisticLikelihood,<:AnalyticVI})
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((μ,y,θ,Σ)->0.5.*(sum(μ.*y)-dot(θ,Σ+abs2.(μ))),
                        model.μ,model.y,model.likelihood.θ,diag.(model.Σ)))
    return tot
end

function expecLogLikelihood(model::SVGP{<:LogisticLikelihood,<:AnalyticVI})
    tot = -model.nLatent*(0.5*model.inference.nSamplesUsed*logtwo)
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y[model.inference.MBIndices])-dot(θ,K̃+κΣκ+abs2.(κμ))),
                        model.κ.*model.μ,model.y,model.likelihood.θ,opt_diag.(model.κ.*model.Σ,model.κ),model.K̃))
    return model.inference.ρ*tot
end

function PolyaGammaKL(model::AbstractGP{<:LogisticLikelihood})
    model.inference.ρ*sum(broadcast(PolyaGammaKL,[ones(length(model.likelihood.c[1]))],model.likelihood.c,model.likelihood.θ))
end

### Gradient Section ###

function gradpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=y*f
    σ*(one(T)-σ)
end

function hessiandiagpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=y*f
    σ*(one(T)-2σ + abs2(σ))
end
