"""
**Logistic Likelihood**

Bernoulli likelihood with a logistic link for the Bernoulli likelihood
    ``p(y|f) = \\sigma(yf) = \\frac{1}{1+\\exp(-yf)}``, (for more info see : [wiki page](https://en.wikipedia.org/wiki/Logistic_function))

```julia
LogisticLikelihood()
```

---

For the analytic version the likelihood, it is augmented via:
```math
p(y|f,\\omega) = \\exp\\left(\\frac{1}{2}\\left(yf - (yf)^2 \\omega\\right)\\right)
```
where ``\\omega \\sim \\text{PG}(\\omega\\mid 1, 0)``, and PG is the Polya-Gamma distribution
See paper : [Efficient Gaussian Process Classification Using Polya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383)
"""
struct LogisticLikelihood{T<:Real} <: ClassificationLikelihood{T}
    c::AbstractVector{T}
    θ::AbstractVector{T}
    function LogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function LogisticLikelihood{T}(c::AbstractVector{<:Real},θ::AbstractVector{<:Real}) where {T<:Real}
        new{T}(c,θ)
    end
end

function LogisticLikelihood()
    LogisticLikelihood{Float64}()
end

function init_likelihood(likelihood::LogisticLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        LogisticLikelihood{T}(abs.(rand(T,nSamplesUsed)),zeros(T,nSamplesUsed))
    else
        LogisticLikelihood{T}()
    end
end

function pdf(l::LogisticLikelihood,y::Real,f::Real)
    logistic(y*f)
end

function Base.show(io::IO,model::LogisticLikelihood{T}) where T
    print(io,"Bernoulli Likelihood with Logistic Link")
end


function compute_proba(l::LogisticLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    sig_pred = zeros(T,N)
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = logistic(μ[i])
        else
            nodes = pred_nodes.*sqrt2.*sqrt.(σ²[i]).+μ[i]
            pred[i] = dot(pred_weights,logistic.(nodes))
            sig_pred[i] = dot(pred_weights,logistic.(nodes).^2)-pred[i]^2
            nodes = pred_nodes.*sqrt2.*sqrt.(σ²[i]).+μ[i]
        end
    end
    return pred, sig_pred
end

### Local Updates Section ###

function local_updates!(l::LogisticLikelihood,y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    l.c .= sqrt.(diag_cov+abs2.(μ))
    l.θ .= 0.5*tanh.(0.5.*l.c)./l.c
end

function sample_local!(l::LogisticLikelihood,y::AbstractVector,f::AbstractVector) where {T}
    pg = PolyaGammaDist()
    l.θ .= draw.([pg],[1.0],f)
    return nothing
end

### Natural Gradient Section ###

@inline ∇E_μ(l::LogisticLikelihood,y::AbstractVector) where {T} = 0.5*y
@inline ∇E_Σ(l::LogisticLikelihood,y::AbstractVector) where {T} = 0.5*l.θ

### ELBO Section ###

function ELBO(model::AbstractGP{T,<:LogisticLikelihood,<:AnalyticVI}) where {T}
    return model.inference.ρ*expecLogLikelihood(model.likelihood, get_y(model), mean_f(model), diag_cov_f(model)) - GaussianKL(model) - model.inference.ρ*PolyaGammaKL(model.likelihood)
end

function expecLogLikelihood(l::LogisticLikelihood,y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    tot = -(0.5*length(y)*logtwo)
    tot += 0.5.*(dot(μ,y)-dot(l.θ,diag_cov)-dot(l.θ,μ))
    return tot
end

function PolyaGammaKL(l::LogisticLikelihood{T}) where {T}
    sum(broadcast(PolyaGammaKL,ones(T,length(l.c)),l.c,l.θ))
end

### Gradient Section ###

@inline grad_log_pdf(::LogisticLikelihood{T},y::Real,f::Real) where {T} = y*logistic(-y*f)

function gradpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=logistic(y*f)
    σ*(one(T)-σ)
end

@inline hessian_log_pdf(::LogisticLikelihood{T},y::Real,f::Real) where {T<:Real} = -exp(y*f)/logistic(-y*f)^2

function hessiandiagpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=logistic(y*f)
    σ*(one(T)-2σ + abs2(σ))
end
