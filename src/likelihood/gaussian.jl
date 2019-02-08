"""
Gaussian likelihood : ``p(y|f) = 𝓝(y|f,ϵ) ``
"""
struct GaussianLikelihood{T<:Real} <: Likelihood{T}
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

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:GaussianLikelihood}
    @assert T<:Real "For regression target(s) should be real valued"
    @assert N <= 2 "Target should be a matrix or a vector"
    if N == 1
        return [y]
    else
        return [y[:,i] for i in 1:size(y,2)]
    end
end

function local_updates!(model::VGP{GaussianLikelihood{T}}) where T
end

function local_updates!(model::SVGP{GaussianLikelihood{T}}) where T
    model.likelihood.ϵ .= 1.0/model.inference.nSamplesUsed *
    norm.(getindex.(model.y,[model.inference.MBIndices]).*model.κ.*model.μ)
    + opt_trace.((model.κ'.*model.κ),model.Σ + sum.(model.K̃) )
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:GaussianLikelihood},index::Integer)
    return model.y[index][model.inference.MBIndices]./model.likelihood.ϵ[index]
end

function expec_μ(model::SVGP{<:GaussianLikelihood})
    return getindex.(model.y,[model.inference.MBIndices]))./model.likelihood.ϵ[index]
end

function expec_Σ(model::SVGP{<:GaussianLikelihood},index::Integer)
    return 0.5/model.likelihood.ϵ[index]*ones(model.inference.nSamplesUsed)
end

function expec_Σ(model::SVGP{<:GaussianLikelihood})
    return [0.5/model.likelihood.ϵ[i]*ones(model.inference.nSamplesUsed) for i in 1:model.nLatent]
end

function natural_gradient!(model::VGP{GaussianLikelihood{T}}) where T
end

function global_update!(model::VGP{GaussianLikelihood{T}}) where T
    if model.inference.nIter == 0
        model.μ .= model.y
    end
end

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::VGP{<:GaussianLikelihood})
    return -0.5*sum(dot.(model.y,inv.(model.Knn.+Diagonal.(model.likelihood.ϵ)).*model.y)
            + logdet.(model.Knn.+Diagonal.(model.likelihood.ϵ))
            .+ model.nFeature*log(2.0π))
end

function ELBO(model::SVGP{<:GaussianLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::SVGP{GaussianLikelihood{T}}) where T
    return -0.5*(model.inference.nSamplesUsed*sum(log.(2π.*model.likelihood.ϵ)) +
                sum(broadcast(x->dot(x,x),getindex.(model.y,[model.inference.MBIndices]).-model.κ.*model.μ) .+
                sum.(model.K̃)+opt_trace.(model.κ.*model.Σ,model.κ))./model.likelihood.ϵ)
end

function hyperparameter_gradient_function(model::VGP{<:GaussianLikelihood})
    model.Σ = inv.(model.invKnn.+model.likelihood.ϵ.*I)
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
