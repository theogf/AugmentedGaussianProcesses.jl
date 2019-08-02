"""Poisson Likelihood"""
struct PoissonLikelihood{T<:Real} <: EventLikelihood{T}
    λ::LatentArray{T}
    c::LatentArray{Vector{T}}
    γ::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function PoissonLikelihood{T}(λ::AbstractVector{T}) where {T<:Real}
        new{T}(λ)
    end
    function PoissonLikelihood{T}(λ,c,γ,θ) where {T<:Real}
        new{T}(λ,c,γ,θ)
    end
end

function PoissonLikelihood(λ::T=1.0) where {T<:Real}
    PoissonLikelihood{T}([λ])
end

function init_likelihood(likelihood::PoissonLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Int,nFeatures::Int) where T
    PoissonLikelihood{T}(
    [likelihood.λ[1] for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent])
end

function pdf(l::PoissonLikelihood,y::Real,f::Real)
    pdf(Poisson(l.λ[1]*logistic(f)),y) #WARNING not valid for multioutput
end

function Base.show(io::IO,model::PoissonLikelihood{T}) where T
    print(io,"Poisson Likelihood")
end

function compute_proba(l::PoissonLikelihood{T},μ::Vector{T},σ²::Vector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = l.λ[1]*logistic(μ[i]) #WARNING Not valid for multioutput
        else
            nodes = pred_nodes.*sqrt2.*sqrt.(σ²[i]).+μ[i]
            pred[i] =  dot(pred_weights,l.λ[1].*logistic.(nodes)) #WARNING not valid for multioutput
        end
    end
    return pred
end

## Local Updates ##

function local_updates!(model::VGP{T,<:PoissonLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(abs2.(μ) + Σ) ,model.μ,diag.(model.Σ))
    model.likelihood.γ .= broadcast((c,μ,λ)->0.5*λ*exp.(-0.5*μ)./cosh.(0.5*c),model.likelihood.c,model.μ,model.likelihood.λ)
    model.likelihood.θ .= broadcast((y,γ,c)->(y+γ)./c.*tanh.(0.5*c),model.y,model.likelihood.γ,model.likelihood.c)
    model.likelihood.λ .= broadcast((y,μ)->sum(y)./sum(logistic.(μ)),model.y,model.μ)
end

function local_updates!(model::SVGP{T,<:PoissonLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((κ,μ,Σ,K̃)->sqrt.(abs2.(κ*μ) + opt_diag(κ*Σ,κ) + K̃),model.κ,model.μ,model.Σ,model.K̃)
    model.likelihood.γ .= broadcast((c,κμ,λ)->0.5*λ*exp.(-0.5*κμ)./cosh.(0.5*c),model.likelihood.c,model.κ.*model.μ,model.likelihood.λ)
    model.likelihood.θ .= broadcast((y,γ,c)->(y+γ)./c.*tanh.(0.5*c),model.inference.y,model.likelihood.γ,model.likelihood.c)
    model.likelihood.λ .= broadcast((y,κμ)->sum(y)./sum(logistic.(κμ)),model.inference.y,model.κ.*model.μ)
end

## Global Updates ##

@inline ∇E_μ(model::AbstractGP{T,<:PoissonLikelihood,<:GibbsorVI}) where {T} = 0.5.*(model.inference.y.-model.likelihood.γ)
@inline ∇E_μ(model::AbstractGP{T,<:PoissonLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*(model.inference.y[i]-model.likelihood.γ[i])
@inline ∇E_Σ(model::AbstractGP{T,<:PoissonLikelihood,<:GibbsorVI}) where {T} = 0.5.*model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:PoissonLikelihood,<:GibbsorVI},i::Int) where {T} = 0.5*model.likelihood.θ[i]

## ELBO Section ##

function ELBO(model::AbstractGP{T,<:PoissonLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model) - PoissonKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{T,<:PoissonLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(abs2.(μ) + Σ) ,model.μ,diag.(model.Σ))
    tot = sum(broadcast((y,λ,γ)->sum(y*log(λ))-sum(lfactorial.(y))-log(2.0)*sum((y+γ)),model.y,model.likelihood.λ,model.likelihood.γ))
    tot += sum(broadcast((μ,y,γ,c,θ)->0.5*dot(μ,(y-γ))-0.5*dot(c.^2,θ),model.μ,model.inference.y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:PoissonLikelihood,<:AnalyticVI}) where {T}
    model.likelihood.c .= broadcast((κ,μ,Σ,K̃)->sqrt.(abs2.(κ*μ) + opt_diag(κ*Σ,κ) + K̃),model.κ,model.μ,model.Σ,model.K̃)
    tot = sum(broadcast((y,λ,γ)->sum(y*log(λ))-sum(lfactorial.(y))-log(2.0)*sum(y+γ),model.inference.y,model.likelihood.λ,model.likelihood.γ))
    tot += sum(broadcast((κμ,y,γ,c,θ)->0.5*dot(κμ,(y-γ))-0.5*dot(c.^2,θ),model.κ.*model.μ,model.inference.y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
    return model.inference.ρ*tot
end

function PoissonKL(model::AbstractGP{T,<:PoissonLikelihood}) where {T}
    return NaN
    #TODO replace with correct expectations
    model.inference.ρ*sum(broadcast(PoissonKL,model.likelihood.γ,model.likelihood.λ))
end

function PolyaGammaKL(model::VGP{T,<:PoissonLikelihood}) where {T}
    sum(broadcast(PolyaGammaKL,model.y.+model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end

function PolyaGammaKL(model::SVGP{T,<:PoissonLikelihood}) where {T}
    model.inference.ρ*sum(broadcast(PolyaGammaKL,model.inference.y.+model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end
