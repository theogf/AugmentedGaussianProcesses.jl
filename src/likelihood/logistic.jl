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

function init_likelihood(likelihood::LogisticLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Integer) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        LogisticLikelihood{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
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
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = logistic(μ[i])
        else
            pred[i] =  expectation(logistic,Normal(μ[i],sqrt(σ²[i])))
        end
    end
    return pred
end

### Local Updates Section ###

function local_updates!(model::VGP{<:LogisticLikelihood,<:AnalyticVI})
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(Σ+abs2.(μ)),model.μ,diag.(model.Σ))
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.likelihood.c)
end

function local_updates!(model::SVGP{<:LogisticLikelihood,<:AnalyticVI})
    model.likelihood.c .= broadcast((μ,Σ,K̃,κ)->sqrt.(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)),model.μ,model.Σ,model.K̃,model.κ)
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.likelihood.c)
end

function sample_local!(model::VGP{<:LogisticLikelihood,<:GibbsSampling})
    pg = PolyaGammaDist()
    model.likelihood.θ .= broadcast((μ::AbstractVector{<:Real})->draw.([pg],[1.0],μ),model.μ)
    return nothing
end

### Natural Gradient Section ###

function expec_μ(model::VGP{<:LogisticLikelihood,<:AnalyticVI},index::Integer)
    return 0.5*model.y[index]
end

function ∇μ(model::VGP{<:LogisticLikelihood})
    return 0.5*model.y
end

function expec_μ(model::SVGP{<:LogisticLikelihood,<:AnalyticVI},index::Integer)
    return 0.5.*model.y[index][model.inference.MBIndices]
end

function ∇μ(model::SVGP{<:LogisticLikelihood})
    return 0.5.*getindex.(model.y,[model.inference.MBIndices])
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
