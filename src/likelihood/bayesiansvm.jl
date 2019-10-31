"""
**Bayesian SVM**

The [Bayesian SVM](https://arxiv.org/abs/1707.05532) is a Bayesian interpretation of the classical SVM.
``p(y|f) \\propto \\exp\\left(2\\max(1-yf,0)\\right)``

```julia
BayesianSVM()
```
---
For the analytic version of the likelihood, it is augmented via:
```math
p(y|f,\\omega) = \\frac{1}{\\sqrt{2\\pi\\omega}}\\exp\\left(-\\frac{1}{2}\\frac{(1+\\omega-yf)^2}{\\omega}\\right)
```
where ``\\omega\\sim 1_{[0,\\infty]}`` has an improper prior (his posterior is however has a valid distribution (Generalized Inverse Gaussian)). For reference [see this paper](http://ecmlpkdd2017.ijs.si/papers/paperID502.pdf)
"""
struct BayesianSVM{T<:Real} <: ClassificationLikelihood{T}
    ω::AbstractVector{T}
    θ::AbstractVector{T}
    function BayesianSVM{T}() where {T<:Real}
        new{T}()
    end
    function BayesianSVM{T}(ω::AbstractVector{<:Real},θ::AbstractVector{<:Real}) where {T<:Real}
        new{T}(ω,θ)
    end
end

function BayesianSVM()
    BayesianSVM{Float64}()
end

function init_likelihood(likelihood::BayesianSVM{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Integer,nFeatures::Integer) where T
    BayesianSVM{T}(abs.(rand(T,nSamplesUsed)),zeros(T,nSamplesUsed))
end
function pdf(l::BayesianSVM,y::Real,f::Real)
    svmlikelihood(y*f)
end

function Base.show(io::IO,model::BayesianSVM{T}) where T
    print(io,"Bayesian SVM")
end

"""Return likelihood equivalent to SVM hinge loss"""
function svmlikelihood(f::Real)
    pos = svmpseudolikelihood(f)
    return pos./(pos.+svmpseudolikelihood(-f))
end

"""Return the pseudo likelihood of the SVM hinge loss"""
function svmpseudolikelihood(f::Real)
    return exp(-2.0*max.(1.0-f,0))
end


function compute_proba(l::BayesianSVM{T},μ::Vector{T},σ²::Vector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    sig_pred = zeros(T,N)
    for i in 1:N
        nodes = pred_nodes.*sqrt(max(σ²[i],zero(T)).+μ[i]
        pred[i] =  dot(pred_weights,svmlikelihood.(nodes))
        sig_pred[i] = dot(pred_weights,svmlikelihood.(nodes).^2)-pred[i]^2
    end
    return pred, sig_pred
end

###############################################################################


function local_updates!(l::BayesianSVM{T},y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    l.ω .= abs2.(one(T) .- y.*μ) + diag_cov
    l.θ .= one(T)./sqrt.(l.ω)
end

@inline ∇E_μ(l::BayesianSVM{T},::AVIOptimizer,y::AbstractVector) where {T} = y.*(l.θ.+one(T))
@inline ∇E_Σ(l::BayesianSVM{T},::AVIOptimizer,y::AbstractVector) where {T} = 0.5.*l.θ

function ELBO(model::AbstractGP{T,<:BayesianSVM,<:AnalyticVI}) where {T}
    (model.inference.ρ*expec_logpdf(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model))
    - GaussianKL(model) - model.inference.ρ*GIGEntropy(model))
end

function expec_logpdf(l::BayesianSVM{T},y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    tot = -(0.5*length(y)*logtwo)
    tot += dot(μ,y)
    tot += -0.5*dot(θ,diag_cov)+dot(θ,abs2.(one(T).-y.*μ))
    return tot
end
