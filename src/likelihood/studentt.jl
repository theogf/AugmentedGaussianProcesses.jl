"""
**Student-T likelihood**

Student-t likelihood for regression: ``\\frac{\\Gamma((\\nu+1)/2)}{\\sqrt{\\nu\\pi}\\Gamma(\\nu/2)}\\left(1+t^2/\\nu\\right)^{(-(\\nu+1)/2)}``
see [wiki page](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

```julia
StudentTLikelihood(ν::T,σ::Real=one(T)) #ν is the number of degrees of freedom
#σ is the variance for local scale of the data.
```

---

For the analytical solution, it is augmented via:
```math
p(y|f,\\omega) = \\mathcal{N}(y|f,\\omega)
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

function init_likelihood(likelihood::StudentTLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int) where T
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
    N = length(μ)
    st = TDist(l.ν)
    nSamples = 2000
    μ_pred = zeros(T,N)
    σ²_pred = zeros(T,N)
    temp_array = zeros(T,nSamples)
    for i in 1:N
        # e = expectation(Normal(μ[i],sqrt(σ²[i])))
        # μ_pred[i] = μ[i]
        #
        # σ²_pred[i] = e(x->pdf(LocationScale(x,1.0,st))^2) - e(x->pdf(LocationScale(x,1.0,st)))^2
        if σ²[i] <= 1e-3
            pyf =  LocationScale(μ[i],sqrt(l.σ),st)
            for j in 1:nSamples
                temp_array[j] = rand(pyf)
            end
        else
            d = Normal(μ[i],sqrt(σ²[i]))
            for j in 1:nSamples
                temp_array[j] = rand(LocationScale(rand(d),sqrt(l.σ),st))
            end
        end
        μ_pred[i] = μ[i];
        σ²_pred[i] = cov(temp_array)
    end
    return μ_pred,σ²_pred
end

## Local Updates ##

function local_updates!(model::VGP{<:StudentTLikelihood,<:AnalyticVI})
    model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.σ*model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast(β->model.likelihood.α./β,model.likelihood.β)
end

function local_updates!(model::SVGP{<:StudentTLikelihood,<:AnalyticVI})
    model.likelihood.β .= broadcast((K̃,κ,Σ,μ,y)->0.5*(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y[model.inference.MBIndices]).+model.likelihood.σ*model.likelihood.ν),model.K̃,model.κ,model.Σ,model.μ,model.y)
    model.likelihood.θ .= broadcast(β->model.likelihood.α./β,model.likelihood.β)
end

function sample_local!(model::VGP{<:StudentTLikelihood,<:GibbsSampling})
    model.likelihood.β .= broadcast((μ::AbstractVector{<:Real},y)->rand.(InverseGamma.(model.likelihood.α,0.5*(abs2.(μ-y).+model.likelihood.σ*model.likelihood.ν))),model.μ,model.y)
    model.likelihood.θ .= broadcast(β->1.0./β,model.likelihood.β)
    return nothing
end

## Global Gradients ##

function cond_mean(model::VGP{<:StudentTLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.θ[index].*model.y[index]
end

function ∇μ(model::VGP{<:StudentTLikelihood})
    return hadamard.(model.likelihood.θ,model.y)
end

function cond_mean(model::SVGP{<:StudentTLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.θ[index].*model.y[index][model.inference.MBIndices]
end

function ∇μ(model::SVGP{<:StudentTLikelihood,<:AnalyticVI})
    return hadamard.(model.likelihood.θ,getindex.(model.y,[model.inference.MBIndices]))
end

function ∇Σ(model::AbstractGP{<:StudentTLikelihood})
    return model.likelihood.θ
end

function ELBO(model::AbstractGP{<:StudentTLikelihood,<:AnalyticVI})
    return expecLogLikelihood(model) - InverseGammaKL(model) - GaussianKL(model)
end

## ELBO Section ##

function expecLogLikelihood(model::VGP{StudentTLikelihood{T},AnalyticVI{T}}) where T
    tot = -0.5*model.nLatent*model.nSample*log(twoπ)
    tot += -0.5.*sum(broadcast(β->sum(model.nSample*digamma(model.likelihood.α).-log.(β)),model.likelihood.β))
    tot += -0.5.*sum(broadcast((θ,Σ,μ,y)->dot(θ,Σ+abs2.(μ)-2.0*μ.*y-abs2.(y)),model.likelihood.θ,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{StudentTLikelihood{T},AnalyticVI{T}}) where T
    tot = -0.5*model.nLatent*model.inference.nSamplesUsed*log(twoπ)
    tot += -0.5.*sum(broadcast(β->sum(model.inference.nSamplesUsed*digamma(model.likelihood.α).-log.(β)),model.likelihood.β))
    tot += -0.5.*sum(broadcast((θ,K̃,κ,Σ,κμ,y)->dot(θ,(K̃+opt_diag(κ*Σ,κ)+abs2.(κμ)-2.0*(κμ).*y[model.inference.MBIndices]-abs2.(y[model.inference.MBIndices]))),model.likelihood.θ,model.K̃,model.κ,model.Σ,model.κ.*model.μ,model.y))
    return model.inference.ρ*tot
end

function InverseGammaKL(model::AbstractGP{<:StudentTLikelihood})
    α_p = model.likelihood.ν/2; β_p= α_p*model.likelihood.σ
    model.inference.ρ*sum(broadcast(InverseGammaKL,model.likelihood.α,model.likelihood.β,α_p,β_p))
end

## Numerical Gradients ##

function gradpdf(::StudentTLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end

function hessiandiagpdf(::StudentTLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end
