"""Poisson Likelihood"""
struct PoissonLikelihood{T<:Real} <: EventLikelihood{T}
    λ::LatentArray{T}
    opt_λ::LatentArray{Optimizer}
    c::LatentArray{Vector{T}}
    γ::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function PoissonLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function PoissonLikelihood{T}(λ,opt_λ,c,γ,θ) where {T<:Real}
        new{T}(λ,opt_λ,c,γ,θ)
    end
end

function PoissonLikelihood()
    PoissonLikelihood{Float64}()
end

function init_likelihood(likelihood::PoissonLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed) where T
    PoissonLikelihood{T}(ones(T,nLatent),
    [Adam(α=0.1) for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent],
    [zeros(T,nSamplesUsed) for _ in 1:nLatent])
end

function pdf(l::PoissonLikelihood,y::Real,f::Real)
    pdf(Poisson(l.λ*logistic(f)),y)
end

function Base.show(io::IO,model::PoissonLikelihood{T}) where T
    print(io,"Poisson Likelihood")
end

function compute_proba(l::PoissonLikelihood{T},μ::Vector{T},σ²::Vector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = l.λ*logistic(μ[i])
        else
            pred[i] =  expectation(x->l.λ*logistic(x),Normal(μ[i],sqrt(σ²[i])))
        end
    end
    return pred
end

###############################################################################


function local_updates!(model::VGP{PoissonLikelihood{T},<:AnalyticVI}) where {T<:Real}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(abs2.(μ) + Σ) ,model.μ,diag.(model.Σ))
    model.likelihood.γ .= broadcast((c,μ,λ)->0.5*λ*exp.(-0.5*μ)./cosh.(0.5*c),model.likelihood.c,model.μ,model.likelihood.λ)
    model.likelihood.θ .= broadcast((y,γ,c)->(y+γ)./c.*tanh.(0.5*c),model.y,model.likelihood.γ,model.likelihood.c)
    model.likelihood.λ .= broadcast((y,μ)->sum(y)./sum(logistic.(μ)),model.y,model.μ)
end

function local_updates!(model::SVGP{PoissonLikelihood{T},<:AnalyticVI}) where {T<:Real}
    model.likelihood.ω .= broadcast((κ,μ,Σ,y,K̃)->abs2.(one(T) .- y[model.inference.MBIndices].*(κ*μ)) + opt_diag(κ*Σ,κ) + K̃,model.κ,model.μ,model.Σ,model.y,model.K̃)
    model.likelihood.θ .= broadcast(b->one(T)./sqrt.(b),model.likelihood.ω)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{PoissonLikelihood{T}},index::Integer) where {T<:Real}
    return 0.5*(model.y[index]-model.likelihood.γ[index])
end

function ∇μ(model::VGP{PoissonLikelihood{T}}) where {T<:Real}
    return 0.5*(model.y.-model.likelihood.γ)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{PoissonLikelihood{T}},index::Integer) where {T<:Real}
    return model.y[index][model.inference.MBIndices].*(model.likelihood.θ[index].+one(T))
end

function ∇μ(model::SVGP{PoissonLikelihood{T}}) where {T<:Real}
    return broadcast((y,θ)->y[model.inference.MBIndices].*(θ.+one(T)),model.y,model.likelihood.θ)
end

function expec_Σ(model::AbstractGP{PoissonLikelihood{T}},index::Integer) where {T<:Real}
    return 0.5*model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{PoissonLikelihood{T}}) where {T<:Real}
    return model.likelihood.θ
end

function ELBO(model::AbstractGP{<:PoissonLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - PoissonKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{PoissonLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(abs2.(μ) + Σ) ,model.μ,diag.(model.Σ))
    tot = sum(broadcast((y,λ,γ)->sum(y*log(λ)-lfactorial.(y)-(y+γ)*log2),model.y,model.likelihood.λ,model.likelihood.γ))
    tot += sum(broadcast((μ,y,γ,c,θ)->0.5*dot(μ,(y-γ))-0.5*dot(c.^2,θ),model.μ,model.y,model.γ,model.c,model.θ))
    return tot
end

function expecLogLikelihood(model::SVGP{PoissonLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->(sum(κμ.*y[model.inference.MBIndices])-0.5*dot(θ,K̃+κΣκ+abs2.(one(T).-y[model.inference.MBIndices].*κμ))),
                        model.κ.*model.μ,model.y,model.likelihood.θ,opt_diag.(model.κ.*model.Σ,model.κ),model.K̃))
    return model.inference.ρ*tot
end
