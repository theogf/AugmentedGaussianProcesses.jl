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
    c::Vector{T}
    θ::Vector{T}
    function StudentTLikelihood{T}(ν::T,σ::T=one(T)) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0,σ)
    end
    function StudentTLikelihood{T}(ν::T,σ::T,c::AbstractVector{T},θ::AbstractVector{T}) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0,σ,c,θ)
    end
end

function StudentTLikelihood(ν::T,σ::T=one(T)) where {T<:Real}
    StudentTLikelihood{T}(ν,σ)
end

function init_likelihood(likelihood::StudentTLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        StudentTLikelihood{T}(
        likelihood.ν,likelihood.σ,
        rand(T,nSamplesUsed),
        zeros(T,nSamplesUsed))
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

function local_updates!(l::StudentTLikelihood{T},y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    l.c .= 0.5*(diag_cov+abs2.(μ-y).+l.σ^2*l.ν)
    l.θ = l.α./l.c
end

function sample_local!(l::StudentTLikelihood{T},y::AbstractVector,f::AbstractVector) where {T}
    l.c .= rand.(InverseGamma.(l.α,0.5*(abs2.(μ-y).+l.σ^2*l.ν)))
    l.θ .= inv.(l.c)
    return nothing
end

## Global Gradients ##

@inline ∇E_μ(l::StudentTLikelihood{T},::AVIOptimizer,y::AbstractVector) where {T} = l.θ.*y
@inline ∇E_Σ(l::StudentTLikelihood{T},::AVIOptimizer,y::AbstractVector) where {T} = 0.5.*l.θ

## ELBO Section ##

function ELBO(model::AbstractGP{T,<:StudentTLikelihood,<:AnalyticVI}) where {T}
    return model.inference.ρ*(expec_logpdf(model.likelihood, get_y(model), mean_f(model), diag_cov_f(model)) - InverseGammaKL(model.likelihood)) - GaussianKL(model)
end

function expec_logpdf(l::StudentTLikelihood{T},y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    tot = -0.5*length(y)*(log(twoπ*l.σ^2))
    tot += -sum(log.(l.c).-digamma(l.α))
    tot += -0.5*(dot(l.θ,diag_cov)+dot(θ,abs2.(μ))-2.0*dot(l.θ,μ.*y)+dot(l.θ,abs2.(y)))
    return tot
end

function InverseGammaKL(l::StudentTLikelihood{T}) where {T}
    α_p = l.ν/2; β_p= α_p*l.σ^2
    InverseGammaKL(l.α,l.c,α_p,β_p)
end

## PDF and Log PDF Gradients ## (verified gradients)

function grad_log_pdf(l::StudentTLikelihood{T},y::Real,f::Real) where {T<:Real}
    (one(T)+l.ν) * (y-f) / ((f-y)^2 + l.σ^2*l.ν)
end

function hessian_log_pdf(l::StudentTLikelihood{T},y::Real,f::Real) where {T<:Real}
    v = l.ν * l.σ^2; Δ² = (f-y)^2
    (one(T)+l.ν) * (-v + Δ²) / (v+Δ²)^2
end
