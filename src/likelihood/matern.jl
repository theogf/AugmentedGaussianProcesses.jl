abstract type MaternLikelihood{T<:Real} <: RegressionLikelihood{T} end


"""
**Matern 3/2 likelihood**

Matern 3/2 likelihood for regression: ````
see [wiki page](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)

```julia
Matern3_2Likelihood(ρ::T) #ρ is the lengthscale
```

---

For the analytical solution, it is augmented via:
```math
p(y|f,\\omega) = \\mathcal{N}(y|f,\\sigma^2\\omega)
```
Where ``\\omega \\sim \\mathcal{IG}(\\frac{\\nu}{2},\\frac{\\nu}{2})`` where ``\\mathcal{IG}`` is the inverse gamma distribution
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
struct Matern3_2Likelihood{T<:Real} <: MaternLikelihood{T}
    ρ::T
    c²::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function Matern3_2Likelihood{T}(ρ::T) where {T<:Real}
        new{T}(ρ)
    end
    function Matern3_2Likelihood{T}(ρ::T,c²::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(ρ,c²,θ)
    end
end

function Matern3_2Likelihood(ρ::T=1.0) where {T<:Real}
    Matern3_2Likelihood{T}(ρ)
end

function init_likelihood(likelihood::Matern3_2Likelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        Matern3_2Likelihood{T}(
        likelihood.ρ,
        [abs2.(T.(rand(T,nSamplesUsed))) for _ in 1:nLatent],
        [zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        Matern3_2Likelihood{T}(likelihood.ρ)
    end
end

function pdf(l::Matern3_2Likelihood{T},y::Real,f::Real) where {T}
    u = sqrt(3)*abs(y-f)/l.ρ
    4*l.ρ/sqrt(3)*(one(T)+u)*exp(-u)
end

function logpdf(l::Matern3_2Likelihood{T},y::Real,f::Real) where {T}
    u = sqrt(3)*abs(y-f)/l.ρ
    log(4*l.ρ/sqrt(3)) + log(one(T)+u) - u
end

function Base.show(io::IO,model::Matern3_2Likelihood{T}) where T
    print(io,"Matern 3/2 likelihood")
end

function compute_proba(l::Matern3_2Likelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    return μ,max.(σ²,0.0).+ 4*l.ρ^2/3
end

## Local Updates ##
function local_updates!(model::VGP{T,<:Matern3_2Likelihood,<:AnalyticVI}) where {T}
    model.likelihood.c² .= broadcast((Σ,μ,y)->Σ+abs2.(μ-y),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast(c²->3.0./(2.0.*sqrt.(3*c²)*model.likelihood.ρ.+2*model.likelihood.ρ^2),model.likelihood.c²)
end

function local_updates!(model::SVGP{T,<:Matern3_2Likelihood,<:AnalyticVI}) where {T}
    model.likelihood.c² .= broadcast((Σ,μ,y)->K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y),diag.(model.Σ),model.K̃,model.κ,model.Σ,model.μ,model.inference.y)
    model.likelihood.θ .= broadcast(c²->3.0./(2.0.*sqrt.(3*c²)*model.likelihood.ρ.+2*model.likelihood.ρ^2),model.likelihood.c²)
end

function sample_local!(model::VGP{T,<:Matern3_2Likelihood,<:GibbsSampling}) where {T}
    model.likelihood.c² .= broadcast((μ,y,ρ)->rand.(GeneralizedInverseGaussian.(3/(2*ρ^2),2.0.*abs2.(μ-y),1.5)),model.μ,model.inference.y,model.likelihood.ρ)
    model.likelihood.θ .= broadcast(c²->c²,model.likelihood.c²)
    return nothing
end

## Global Gradients ##

@inline ∇E_μ(model::AbstractGP{T,<:Matern3_2Likelihood,<:GibbsorVI}) where {T} = 2.0.*hadamard.(model.likelihood.θ,model.inference.y)
@inline ∇E_μ(model::AbstractGP{T,<:Matern3_2Likelihood,<:GibbsorVI},i::Int) where {T} = 2.0.*hadamard(model.likelihood.θ[i],model.inference.y[i])

@inline ∇E_Σ(model::AbstractGP{T,<:Matern3_2Likelihood,<:GibbsorVI}) where {T} = model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:Matern3_2Likelihood,<:GibbsorVI},i::Int) where {T} = model.likelihood.θ[i]

## ELBO Section ##

function ELBO(model::AbstractGP{T,<:Matern3_2Likelihood,<:AnalyticVI}) where {T}
    return NaN;
    #expecLogLikelihood(model) - InverseGammaKL(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{T,<:Matern3_2Likelihood,<:AnalyticVI}) where {T}
    # model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.σ^2*model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    # model.likelihood.θ .= broadcast(β->model.likelihood.α./β,model.likelihood.β)
    tot = -0.5*model.nLatent*model.nSample*(log(twoπ*model.likelihood.σ^2))
    tot += -sum(broadcast((α,β)->sum(log.(β).-digamma(α)), model.likelihood.α,model.likelihood.β))
    tot += -0.5.*sum(broadcast((θ,Σ,μ,y)->dot(θ,Σ)+dot(θ,abs2.(μ))-2.0*dot(θ,μ.*y)+dot(θ,abs2.(y)),model.likelihood.θ,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:Matern3_2Likelihood,<:AnalyticVI}) where {T}
    tot = -0.5*model.nLatent*model.inference.nSamplesUsed*(log(twoπ*model.likelihood.σ^2))
    tot += -sum(broadcast(β->sum(digamma(model.likelihood.α).-log.(β)),model.likelihood.β))
    tot += -0.5.*sum(broadcast((θ,K̃,κ,Σ,κμ,y)->dot(θ,(K̃+opt_diag(κ*Σ,κ)+abs2.(κμ)-2.0*(κμ).*y+abs2.(y))),model.likelihood.θ,model.K̃,model.κ,model.Σ,model.κ.*model.μ,model.inference.y))
    return model.inference.ρ*tot
end

function InverseGammaKL(model::AbstractGP{T,<:Matern3_2Likelihood}) where {T}
    α_p = model.likelihood.ν/2; β_p= α_p*model.likelihood.σ^2
    model.inference.ρ*sum(broadcast(InverseGammaKL,model.likelihood.α,model.likelihood.β,α_p,β_p))
end

## PDF and Log PDF Gradients ## (verified gradients)

function grad_log_pdf(l::Matern3_2Likelihood{T},y::Real,f::Real) where {T<:Real}
    3.0 * (y-f) / (l.ρ*(abs(f-y)*sqrt(3)+l.ρ))
end

function hessian_log_pdf(l::Matern3_2Likelihood{T},y::Real,f::Real) where {T<:Real}
    3.0 / (l.ρ+sqrt(3)*abs(f-y))^2
end
