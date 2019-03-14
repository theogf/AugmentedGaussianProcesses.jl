"""
Gaussian likelihood : ``p(y|f) = 𝓝(y|f,ϵ) ``
"""
struct GaussianLikelihood{T<:Real} <: RegressionLikelihood{T}
    ϵ::AbstractVector{T}
    function GaussianLikelihood{T}(ϵ::Real) where {T<:Real}
        new{T}([ϵ])
    end
    function GaussianLikelihood{T}(ϵ::AbstractVector) where {T<:Real}
        new{T}(ϵ)
    end
end

function GaussianLikelihood(ϵ::T=1e-3) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end

function GaussianLikelihood(ϵ::AbstractVector{T}) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end

function pdf(l::GaussianLikelihood,y::Real,f::Real)
    pdf(Normal(y,l.ϵ[1]),f)
end

function logpdf(l::GaussianLikelihood,y::Real,f::Real)
    logpdf(Normal(y,l.ϵ[1]),f)
end

function Base.show(io::IO,model::GaussianLikelihood{T}) where T
    print(io,"Gaussian likelihood")
end


function init_likelihood(likelihood::GaussianLikelihood{T},nLatent::Integer,nSamples::Integer) where {T<:Real}
    if length(likelihood.ϵ) ==1 && length(likelihood.ϵ) != nLatent
        return GaussianLikelihood{T}([likelihood.ϵ[1] for _ in 1:nLatent])
    elseif length(likelihood.ϵ) != nLatent
        @warn "Wrong dimension of ϵ : $(length(likelihood.ϵ)), using first value only"
        return GaussianLikelihood{T}([likelihood.ϵ[1] for _ in 1:nLatent])
    else
        return likelihood
    end
end

function local_updates!(model::VGP{GaussianLikelihood{T}}) where T
end

function local_updates!(model::SVGP{GaussianLikelihood{T}}) where T
    model.likelihood.ϵ .= 1.0/model.inference.nSamplesUsed *
    norm.(getindex.(model.y,[model.inference.MBIndices]).*model.κ.*model.μ) + opt_trace.((model.κ'.*model.κ),model.Σ) + sum.(model.K̃)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:GaussianLikelihood},index::Integer)
    return model.y[index][model.inference.MBIndices]./model.likelihood.ϵ[index]
end

function ∇μ(model::SVGP{<:GaussianLikelihood{T}}) where {T<:Real}
    return getindex.(model.y,[model.inference.MBIndices])./model.likelihood.ϵ[index]
end

function expec_Σ(model::SVGP{<:GaussianLikelihood},index::Integer)
    return 0.5/model.likelihood.ϵ[index]*ones(model.inference.nSamplesUsed)
end

function ∇Σ(model::SVGP{<:GaussianLikelihood{T}}) {T<:Real}
    return [0.5/model.likelihood.ϵ[i]*ones(model.inference.nSamplesUsed) for i in 1:model.nLatent]
end

function natural_gradient!(model::VGP{GaussianLikelihood{T},AnalyticInference{T}}) where {T<:Real}
end

function global_update!(model::VGP{GaussianLikelihood{T},AnalyticInference{T}}) where {T<:Real}
    if model.inference.nIter <= 1
        model.μ .= model.y
    end
end

function proba_y(model::GP{GaussianLikelihood{T},AnalyticInference{T}},X_test::AbstractMatrix) where {T<:Real}
    μ_f,Σ_f = predict_f(model,X_test,covf=true)
    Σ_f .+= [ones(size(X_test,1))].*model.likelihood.ϵ
    return μ_f,Σ_f
end

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::VGP{<:GaussianLikelihood{T}}) where {T<:Real}
    return -0.5*sum(broadcast((y,K,ϵ)->dot(y,inv(K+ϵ*I)*y)            + logdet(K+ϵ*I)+ model.nFeature*log(twoπ),model.y,model.Knn,model.likelihood.ϵ))
end

function ELBO(model::SVGP{<:GaussianLikelihood{T}}) where {T<:real}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::SVGP{GaussianLikelihood{T}}) where T
    return -0.5*(model.inference.nSamplesUsed*sum(log.(2π.*model.likelihood.ϵ)) +
                sum(broadcast(x->dot(x,x),getindex.(model.y,[model.inference.MBIndices]).-model.κ.*model.μ) .+
                sum.(model.K̃)+opt_trace.(model.κ.*model.Σ,model.κ))./model.likelihood.ϵ)
end

function hyperparameter_gradient_function(model::VGP{<:GaussianLikelihood})
    model.Σ .= broadcast((invK,ϵ)->Symmetric(inv(invK +ϵ*I)),model.invKnn,model.likelihood.ϵ)
    A = (model.Σ.*(model.µ.*transpose.(model.μ)).-[I]).*model.Σ
    if model.IndependentPriors
        return (function(Jnn,index)
                    return 0.5*opt_trace(Jnn,A[index])
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*opt_trace(model.Knn[index],A[index])
                end)
    else
        return (function(Jnn,index)
            return 0.5*sum(opt_trace.(Jnn.*transpose(A[i])) for i in 1:model.nLatent)
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*sum(sum(model.Knn[1].*transpose(A[i])) for i in 1:model.nLatent)
                end)
    end
end
