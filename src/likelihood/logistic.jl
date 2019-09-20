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
    c::AbstractVector{AbstractVector{T}}
    θ::AbstractVector{AbstractVector{T}}
    function LogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function LogisticLikelihood{T}(c::AbstractVector{<:AbstractVector{<:Real}},θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
        new{T}(c,θ)
    end
end

function LogisticLikelihood()
    LogisticLikelihood{Float64}()
end

function init_likelihood(likelihood::LogisticLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        LogisticLikelihood{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        LogisticLikelihood{T}()
    end
end

function pdf(l::LogisticLikelihood,y::Real,f::Real)
    logistic(y*f)
end

function logpdf(l::LogisticLikelihood,y::T,f::T) where {T<:Real}
    -log(one(T)+exp(-y*f))
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

function local_updates!(model::VGP{T,<:LogisticLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(Σ+abs2.(μ)),model.μ,diag.(model.Σ))
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.likelihood.c)
end

function local_updates!(model::SVGP{T,<:LogisticLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((μ,Σ,K̃,κ)->sqrt.(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)),model.μ,model.Σ,model.K̃,model.κ)
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.likelihood.c)
end

function sample_local!(model::VGP{T,<:LogisticLikelihood,<:GibbsSampling}) where {T}
    pg = PolyaGammaDist()
    model.likelihood.θ .= broadcast((μ::AbstractVector{<:Real})->draw.([pg],[1.0],μ),model.μ) #Sample from ω
    return nothing
end

### Natural Gradient Section ###

@inline ∇E_μ(model::AbstractGP{T,<:LogisticLikelihood,<:GibbsorVI}) where {T} = 0.5*model.inference.y
@inline ∇E_μ(model::AbstractGP{T,<:LogisticLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*model.inference.y[i]
@inline ∇E_Σ(model::AbstractGP{T,<:LogisticLikelihood,<:GibbsorVI}) where {T} = 0.5*model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:LogisticLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*model.likelihood.θ[i]

### ELBO Section ###

function ELBO(model::AbstractGP{T,<:LogisticLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{T,<:LogisticLikelihood,<:AnalyticVI}) where {T}
    tot = -model.nLatent*(model.nSample*logtwo)
    tot += 0.5*sum(broadcast((μ,y,θ,Σ)->dot(μ,y)-dot(θ,Σ)-dot(θ,abs2.(μ)),
                        model.μ,model.y,model.likelihood.θ,diag.(model.Σ)))
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:LogisticLikelihood,<:AnalyticVI}) where {T}
    tot = -model.nLatent*(0.5*model.inference.nSamplesUsed*logtwo)
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y)-dot(θ,K̃+κΣκ+abs2.(κμ))),
                        model.κ.*model.μ,model.inference.y,model.likelihood.θ,opt_diag.(model.κ.*model.Σ,model.κ),model.K̃))
    return model.inference.ρ*tot
end

function PolyaGammaKL(model::AbstractGP{T,<:LogisticLikelihood}) where {T}
    model.inference.ρ*sum(broadcast(PolyaGammaKL,[ones(length(model.likelihood.c[1]))],model.likelihood.c,model.likelihood.θ))
end

### Gradient Section ###

@inline grad_log_pdf(::LogisticLikelihood{T},y::Real,f::Real) where {T} = y*logistic(-y*f)

function gradpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=logistic(y*f)
    σ*(one(T)-σ)
end

@inline hessian_log_pdf(::LogisticLikelihood{T},y::Real,f::Real) where {T<:Real} = -exp(y*f)*logistic(-y*f)^2

function hessiandiagpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=logistic(y*f)
    σ*(one(T)-2σ + abs2(σ))
end
