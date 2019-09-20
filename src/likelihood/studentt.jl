"""
**Student-T likelihood**

Student-t likelihood for regression: ``\\frac{\\Gamma((\\nu+1)/2)}{\\sqrt{\\nu\\pi}\\sigma\\Gamma(\\nu/2)}\\left(1+(y-f)^2/(\\sigma^2\\nu)\\right)^{(-(\\nu+1)/2)}``
see [wiki page](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

```julia
StudentTLikelihood(ν::T,σ::Real=one(T)) #ν is the number of degrees of freedom
#σ is the variance for local scale of the data.
```

---

For the analytical solution, it is augmented via:
```math
p(y|f,\\omega) = \\mathcal{N}(y|f,\\sigma^2\\omega)
```
Where ``\\omega \\sim \\mathcal{IG}(\\frac{\\nu}{2},\\frac{\\nu}{2})`` where ``\\mathcal{IG}`` is the inverse gamma distribution
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
struct StudentTLikelihood{T<:Real} <: RegressionLikelihood{T}
    ν::T
    α::T
    σ::T
    β::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function StudentTLikelihood{T}(ν::T,σ::T=one(T)) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0,σ)
    end
    function StudentTLikelihood{T}(ν::T,σ::T,β::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0,σ,β,θ)
    end
end

function StudentTLikelihood(ν::T,σ::T=one(T)) where {T<:Real}
    StudentTLikelihood{T}(ν,σ)
end

function init_likelihood(likelihood::StudentTLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        StudentTLikelihood{T}(
        likelihood.ν,likelihood.σ,
        [abs2.(T.(rand(T,nSamplesUsed))) for _ in 1:nLatent],
        [zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        StudentTLikelihood{T}(likelihood.ν,likelihood.σ)
    end
end

function pdf(l::StudentTLikelihood,y::Real,f::Real)
    tdistpdf(l.ν,(y-f)/l.σ)
end

function Base.show(io::IO,model::StudentTLikelihood{T}) where T
    print(io,"Student-t likelihood")
end

function compute_proba(l::StudentTLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    return μ,max.(σ²,0.0).+0.5*l.ν*l.σ^2/(0.5*l.ν-1)
end

## Local Updates ##

function local_updates!(model::VGP{T,<:StudentTLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.σ^2*model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast(β->model.likelihood.α./β,model.likelihood.β)
end

function local_updates!(model::SVGP{T,<:StudentTLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.β .= broadcast((K̃,κ,Σ,μ,y)->0.5*(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y).+model.likelihood.σ^2*model.likelihood.ν),model.K̃,model.κ,model.Σ,model.μ,model.inference.y)
    model.likelihood.θ .= broadcast(β->model.likelihood.α./β,model.likelihood.β)
end

function sample_local!(model::VGP{T,<:StudentTLikelihood,<:GibbsSampling}) where {T}
    model.likelihood.β .= broadcast((μ::AbstractVector{<:Real},y)->rand.(InverseGamma.(model.likelihood.α,0.5*(abs2.(μ-y).+model.likelihood.σ^2*model.likelihood.ν))),model.μ,model.y) #Sample from ω
    model.likelihood.θ .= broadcast(β->1.0./β,model.likelihood.β) #Return 1/ω
    return nothing
end

## Global Gradients ##

@inline ∇E_μ(model::AbstractGP{T,<:StudentTLikelihood,<:GibbsorVI}) where {T} = hadamard.(model.likelihood.θ,model.inference.y)
@inline ∇E_μ(model::AbstractGP{T,<:StudentTLikelihood,<:GibbsorVI},i::Int) where {T} = hadamard(model.likelihood.θ[i],model.inference.y[i])

@inline ∇E_Σ(model::AbstractGP{T,<:StudentTLikelihood,<:GibbsorVI}) where {T} = 0.5.*model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:StudentTLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5.*model.likelihood.θ[i]

## ELBO Section ##

function ELBO(model::AbstractGP{T,<:StudentTLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - InverseGammaKL(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{T,<:StudentTLikelihood,<:AnalyticVI}) where {T}
    # model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.σ^2*model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    # model.likelihood.θ .= broadcast(β->model.likelihood.α./β,model.likelihood.β)
    tot = -0.5*model.nLatent*model.nSample*(log(twoπ*model.likelihood.σ^2))
    tot += -sum(broadcast((α,β)->sum(log.(β).-digamma(α)), model.likelihood.α,model.likelihood.β))
    tot += -0.5.*sum(broadcast((θ,Σ,μ,y)->dot(θ,Σ)+dot(θ,abs2.(μ))-2.0*dot(θ,μ.*y)+dot(θ,abs2.(y)),model.likelihood.θ,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:StudentTLikelihood,<:AnalyticVI}) where {T}
    tot = -0.5*model.nLatent*model.inference.nSamplesUsed*(log(twoπ*model.likelihood.σ^2))
    tot += -sum(broadcast(β->sum(digamma(model.likelihood.α).-log.(β)),model.likelihood.β))
    tot += -0.5.*sum(broadcast((θ,K̃,κ,Σ,κμ,y)->dot(θ,(K̃+opt_diag(κ*Σ,κ)+abs2.(κμ)-2.0*(κμ).*y+abs2.(y))),model.likelihood.θ,model.K̃,model.κ,model.Σ,model.κ.*model.μ,model.inference.y))
    return model.inference.ρ*tot
end

function InverseGammaKL(model::AbstractGP{T,<:StudentTLikelihood}) where {T}
    α_p = model.likelihood.ν/2; β_p= α_p*model.likelihood.σ^2
    model.inference.ρ*sum(broadcast(InverseGammaKL,model.likelihood.α,model.likelihood.β,α_p,β_p))
end

## PDF and Log PDF Gradients ## (verified gradients)

function grad_log_pdf(l::StudentTLikelihood{T},y::Real,f::Real) where {T<:Real}
    (one(T)+l.ν) * (y-f) / ((f-y)^2 + l.σ^2*l.ν)
end

function hessian_log_pdf(l::StudentTLikelihood{T},y::Real,f::Real) where {T<:Real}
    v = l.ν * l.σ^2; Δ² = (f-y)^2
    (one(T)+l.ν) * (-v + Δ²) / (v+Δ²)^2
end
