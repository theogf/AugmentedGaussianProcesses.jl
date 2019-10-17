"""NegBinomial Likelihood"""
struct NegBinomialLikelihood{T<:Real} <: EventLikelihood{T}
    r::LatentArray{Int}
    c::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function NegBinomialLikelihood{T}(r::AbstractVector{Int}) where {T<:Real}
        new{T}(r)
    end
    function NegBinomialLikelihood{T}(r,c,θ) where {T<:Real}
        new{T}(r,c,θ)
    end
end

function NegBinomialLikelihood(r::Int=10) where {T<:Real}
    NegBinomialLikelihood{Float64}([r])
end

function init_likelihood(likelihood::NegBinomialLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Int,nFeatures::Int) where T
    NegBinomialLikelihood{T}(
    [likelihood.r[1] for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent])
end

function pdf(l::NegBinomialLikelihood,y::Real,f::Real)
    pdf(NegativeBinomial(l.r[1],_get_p(l,f)),y) #WARNING not valid for multioutput
end

function get_p(l::NegBinomialLikelihood,μ::AbstractVector{<:AbstractVector})
    _get_p.(model.likelihood,μ)
end

function _get_p(::NegBinomialLikelihood,μ)
    logistic.(μ)
end

function Base.show(io::IO,model::NegBinomialLikelihood{T}) where T
    print(io,"Negative Binomial Likelihood")
end

function compute_proba(l::NegBinomialLikelihood{T},μ::Vector{T},σ²::Vector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = logistic(μ[i]) #WARNING Not valid for multioutput
        else
            nodes = pred_nodes.*sqrt2.*sqrt.(σ²[i]).+μ[i]
            pred[i] =  dot(pred_weights,logistic.(nodes)) #WARNING not valid for multioutput
        end
    end
    return pred
end

## Local Updates ##

function local_updates!(model::VGP{T,<:NegBinomialLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(abs2.(μ) + Σ) ,model.μ,diag.(model.Σ))
    model.likelihood.θ .= broadcast((y,r,c)->(r+y)./c.*tanh.(0.5*c),model.y,model.likelihood.r,model.likelihood.c)
end

function local_updates!(model::SVGP{T,<:NegBinomialLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((κ,μ,Σ,K̃)->sqrt.(abs2.(κ*μ) + opt_diag(κ*Σ,κ) + K̃),model.κ,model.μ,model.Σ,model.K̃)
    model.likelihood.θ .= broadcast((y,r,c)->(y+r)./c.*tanh.(0.5*c),model.inference.y,model.likelihood.r,model.likelihood.c)
end

## Global Updates ##

@inline ∇E_μ(model::AbstractGP{T,<:NegBinomialLikelihood,<:GibbsorVI}) where {T} = 0.5.*(model.inference.y.-model.likelihood.r)
@inline ∇E_μ(model::AbstractGP{T,<:NegBinomialLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*(model.inference.y[i]-model.likelihood.r[i])
@inline ∇E_Σ(model::AbstractGP{T,<:NegBinomialLikelihood,<:GibbsorVI}) where {T} = 0.5.*model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:NegBinomialLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*model.likelihood.θ[i]

## ELBO Section ##

function ELBO(model::AbstractGP{T,<:NegBinomialLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{T,<:NegBinomialLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(abs2.(μ) + Σ) ,model.μ,diag.(model.Σ))
    tot = sum(broadcast((y,r)->sum(logabsbinomial(y+r-1,y))-log(2.0)*sum((y+r)),model.y,model.likelihood.r))
    tot += sum(broadcast((μ,y,r,c,θ)->0.5*dot(μ,(y-r))-0.5*dot(c.^2,θ),model.μ,model.inference.y,model.likelihood.r,model.likelihood.c,model.likelihood.θ))
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:NegBinomialLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((κ,μ,Σ,K̃)->sqrt.(abs2.(κ*μ) + opt_diag(κ*Σ,κ) + K̃),model.κ,model.μ,model.Σ,model.K̃)
    tot = sum(broadcast((y,r)->sum(logabsbinomial(y+r-1,y))-log(2.0)*sum((y+r)),model.inference.y,model.likelihood.r))
    tot += sum(broadcast((κμ,y,r,c,θ)->0.5*dot(κμ,(y-r))-0.5*dot(c.^2,θ),model.κ.*model.μ,model.inference.y,model.likelihood.r,model.likelihood.c,model.likelihood.θ))
    return model.inference.ρ*tot
end

function PolyaGammaKL(model::VGP{T,<:NegBinomialLikelihood}) where {T}
    sum(broadcast(PolyaGammaKL,model.y.+model.likelihood.r,model.likelihood.c,model.likelihood.θ))
end

function PolyaGammaKL(model::SVGP{T,<:NegBinomialLikelihood}) where {T}
    model.inference.ρ*sum(broadcast(PolyaGammaKL,model.inference.y.+model.likelihood.r,model.likelihood.c,model.likelihood.θ))
end
