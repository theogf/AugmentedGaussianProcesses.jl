"""Poisson Likelihood"""
struct PoissonLikelihood{T<:Real} <: EventLikelihood{T}
    λ::LatentArray{T}

end



function PoissonLikelihood()
    PoissonLikelihood{Float64}()
end

function init_likelihood(likelihood::PoissonLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed) where T
    PoissonLikelihood{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
end
function pdf(l::PoissonLikelihood,y::Real,f::Real)
    pdf(Poisson(l.λ*logistic(f),y)
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
            pred[i] =  expectation(svmlikelihood,Normal(μ[i],sqrt(σ²[i])))
        end
    end
    return pred
end

###############################################################################


function local_updates!(model::VGP{PoissonLikelihood{T},<:AnalyticVI}) where {T<:Real}
    model.likelihood.ω .= broadcast((μ,Σ,y)->abs2.(one(T) .- y.*μ) + Σ ,model.μ,diag.(model.Σ),model.y)
    model.likelihood.θ .= broadcast(b->one(T)./sqrt.(b),model.likelihood.ω)
end

function local_updates!(model::SVGP{PoissonLikelihood{T},<:AnalyticVI}) where {T<:Real}
    model.likelihood.ω .= broadcast((κ,μ,Σ,y,K̃)->abs2.(one(T) .- y[model.inference.MBIndices].*(κ*μ)) + opt_diag(κ*Σ,κ) + K̃,model.κ,model.μ,model.Σ,model.y,model.K̃)
    model.likelihood.θ .= broadcast(b->one(T)./sqrt.(b),model.likelihood.ω)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{PoissonLikelihood{T}},index::Integer) where {T<:Real}
    return model.y[index].*(model.likelihood.θ[index] .+ one(T))
end

function ∇μ(model::VGP{PoissonLikelihood{T}}) where {T<:Real}
    return broadcast((y,θ)->y.*(θ.+one(T)),model.y,model.likelihood.θ)
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
    return expecLogLikelihood(model) - GaussianKL(model) - GIGEntropy(model)
end

function expecLogLikelihood(model::VGP{PoissonLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((μ,y,θ,Σ)->(sum(μ.*y)-0.5*dot(θ,Σ+abs2.(one(T).-y.*μ))),
                        model.μ,model.y,model.likelihood.θ,diag.(model.Σ)))
    return tot
end

function expecLogLikelihood(model::SVGP{PoissonLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->(sum(κμ.*y[model.inference.MBIndices])-0.5*dot(θ,K̃+κΣκ+abs2.(one(T).-y[model.inference.MBIndices].*κμ))),
                        model.κ.*model.μ,model.y,model.likelihood.θ,opt_diag.(model.κ.*model.Σ,model.κ),model.K̃))
    return model.inference.ρ*tot
end
