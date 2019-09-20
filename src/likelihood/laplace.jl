"""
**Laplace likelihood**

Laplace likelihood for regression: ``\\frac{1}{2\\beta}\\exp\\left(-\\frac{|y-f|}{\\beta}\\right)``
see [wiki page](https://en.wikipedia.org/wiki/Laplace_distribution)

```julia
LaplaceLikelihood(β::T=1.0)  #  Laplace likelihood with scale β
```

---

For the analytical solution, it is augmented via:
```math
p(y|f,\\omega) = \\mathcal{N}(y|f,\\omega^{-1})
```
where ``\\omega \\sim \\text{Exp}\\left(\\omega \\mid \\frac{1}{2 \\beta^2}\\right)``, and Exp is the [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
We approximate ``q(\\omega) = \\mathcal{GIG}\\left(\\omega \\mid a,b,p\\right)
"""
struct LaplaceLikelihood{T<:Real} <: RegressionLikelihood{T}
    β::LatentArray{T}
    a::LatentArray{T}
    p::LatentArray{T}
    b::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}} #Expected value of ω
    function LaplaceLikelihood{T}(β::AbstractVector{T}) where {T<:Real}
        new{T}(β,β.^-2,0.5*ones(size(β)))
    end
    function LaplaceLikelihood{T}(β::AbstractVector{T},b::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(β,β.^(-2),0.5*ones(size(β)),b,θ)
    end
end

function LaplaceLikelihood(β::T=1.0) where {T<:Real}
    LaplaceLikelihood{T}([β])
end

function init_likelihood(likelihood::LaplaceLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        LaplaceLikelihood{T}(
        likelihood.β,
        [abs2.(T.(rand(T,nSamplesUsed))) for _ in 1:nLatent],
        [zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        LaplaceLikelihood{T}(likelihood.β)
    end
end

function pdf(l::LaplaceLikelihood,y::Real,f::Real)
    Distributions.pdf(Laplace(f,l.β[1]),y) #WARNING multioutput invalid
end

function Base.show(io::IO,model::LaplaceLikelihood{T}) where T
    print(io,"Laplace likelihood")
end

function compute_proba(l::LaplaceLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    return μ,max.(σ²,0.0).+ 2*l.β[1]^2
end

## Local Updates ##

function local_updates!(model::Union{VGP{T,<:LaplaceLikelihood,<:AnalyticVI},VStP{T,<:LaplaceLikelihood,<:AnalyticVI}}) where {T}
    model.likelihood.b .= broadcast((Σ,μ,y)->(Σ+abs2.(μ-y)),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast((a,b)->sqrt(a)./sqrt.(b),model.likelihood.a,model.likelihood.b)
end

function local_updates!(model::SVGP{T,<:LaplaceLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.b .= broadcast((K̃,κ,Σ,μ,y)->(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y)),model.K̃,model.κ,model.Σ,model.μ,model.inference.y)
    model.likelihood.θ .= broadcast((a,b)->sqrt(a)./sqrt.(b),model.likelihood.a,model.likelihood.b)
end

function sample_local!(model::VGP{T,<:LaplaceLikelihood,<:GibbsSampling}) where {T}
    model.likelihood.b .= broadcast((μ,y,β)->rand.(GeneralizedInverseGaussian.(1/β^2,abs2.(μ-y),0.5)),model.μ,model.inference.y,model.likelihood.β) #Sample from ω
    model.likelihood.θ .= broadcast(b->1.0./b,model.likelihood.b) #Return inverse of ω
    return nothing
end

@inline ∇E_μ(model::AbstractGP{T,<:LaplaceLikelihood,<:GibbsorVI}) where {T} =  hadamard.(model.likelihood.θ,model.inference.y)
@inline ∇E_μ(model::AbstractGP{T,<:LaplaceLikelihood,<:GibbsorVI},i::Int) where {T} =  model.likelihood.θ[i].*model.inference.y[i]

@inline ∇E_Σ(model::AbstractGP{T,<:LaplaceLikelihood,<:GibbsorVI}) where {T} = 0.5*model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:LaplaceLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*model.likelihood.θ[i]


## ELBO ##
function ELBO(model::AbstractGP{T,<:LaplaceLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GIGExpKL(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{T,LaplaceLikelihood{T},AnalyticVI{T}}) where {T}
    tot = -0.5*model.nLatent*model.nSample*log(twoπ)
    tot += 0.5.*sum(broadcast(θ->sum(log.(θ)),model.likelihood.θ))
    tot += -0.5.*sum(broadcast((θ,Σ,μ,y)->dot(θ,(Σ+abs2.(μ)-2.0*μ.*y-abs2.(y))),model.likelihood.θ,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{T,LaplaceLikelihood{T},AnalyticVI{T}}) where {T}
    tot = -0.5*model.nLatent*model.inference.nSamplesUsed*log(twoπ)
    tot += 0.5.*sum(broadcast(θ->sum(log.(θ)),model.likelihood.θ))
    tot += -0.5.*sum(broadcast((θ,K̃,κ,Σ,μ,y)->dot(θ,(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)-2.0*(κ*μ).*y-abs2.(y))),model.likelihood.θ,model.K̃,model.κ,model.Σ,model.μ,model.inference.y))
    return model.inference.ρ*tot
end

function GIGExpKL(model::AbstractGP{T,<:LaplaceLikelihood}) where {T}
    GIGEntropy(model)-expecExponentialGIG(model)
end

function GIGEntropy(model::AbstractGP{T,<:LaplaceLikelihood}) where {T}
    model.inference.ρ*sum(broadcast(GIGEntropy,model.likelihood.a,model.likelihood.b,model.likelihood.p))
end

function expecExponentialGIG(model::AbstractGP{T,<:LaplaceLikelihood}) where {T}
    sum(broadcast((β,a,b)->sum(-log(2*β^2).-0.5*(a.*sqrt.(b)+b.*sqrt(a))./(a.*b*β^2)),model.likelihood.β,model.likelihood.a,model.likelihood.b))
end

## PDF and Log PDF Gradients ##

function grad_quad(likelihood::LaplaceLikelihood{T},y::Real,μ::Real,σ²::Real,inference::Inference) where {T<:Real}
    nodes = inference.nodes*sqrt2*sqrt(σ²) .+ μ
    Edlogpdf = dot(inference.weights,grad_log_pdf.(likelihood,y,nodes))
    Ed²logpdf =  (1/sqrt(twoπ*σ²))/(likelihood.β[1]^2)
    return -Edlogpdf::T, Ed²logpdf::T
end


@inline grad_log_pdf(l::LaplaceLikelihood{T},y::Real,f::Real) where {T<:Real} = sign(y-f)./l.β[1]

function gradpdf(l::LaplaceLikelihood{T},y::Real,f::Real) where {T<:Real}
    grad_log_pdf_μ(l,y,f)*pdf(l,y,f)
end

function hessiandiagpdf(l::LaplaceLikelihood{T},y::Real,f::Real) where {T<:Real}
    pdf(l,y,f)/(l.β[1]^2)
end

@inline hessian_log_pdf(l::LaplaceLikelihood{T},y::Real,f::Real) where {T<:Real} = zero(T)
