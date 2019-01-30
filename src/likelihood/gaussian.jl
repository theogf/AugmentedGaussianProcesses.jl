"""
Gaussian likelihood : ``p(y|f) = 𝓝(y|f,ϵ) ``
"""
struct GaussianLikelihood{T<:Real} <: Likelihood{T}
    ϵ::T
    function GaussianLikelihood{T}(ϵ) where {T<:Real}
        new{T}(ϵ)
    end
end

function GaussianLikelihood(ϵ::T=1e-3) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end


function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:Union{GaussianLikelihood}}
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
    model.likelihood.ϵ = 1.0/model.nSamplesUsed/model.nLatent * ( dot.(model.y[model.MBIndices],model.y[model.MBIndices])
    - 2.0*dot.(model.y[model.MBIndices],model.κ.*model.μ)
    + opt_trace.((model.κ'.*model.κ).*(model.μ.*transpose.(model.μ).+model.Σ)) + sum.(model.K̃) )
end

function natural_gradient!(model::VGP{GaussianLikelihood{T}}) where T
end

function natural_gradient!(model::SVGP{GaussianLikelihood})
    model.∇η₁ .= model.likelihood.ρ.*(model.κ'*model.y[model.MBIndices])./model.likelihood.ϵ - model.η₁
    model.∇η₁ = Symmetric(-0.5*(model.likelihood.ρ*(model.κ')*model.κ./model.likelihood.ϵ+model.invKmm) - model.η₂)
end

function global_update!(model::VGP{GaussianLikelihood{T}}) where T
    if model.inference.nIter == 0
        model.μ .= model.y
    end
end

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::VGP{<:GaussianLikelihood})
    return -0.5*sum(dot.(model.y,inv.(model.Knn.+[Diagonal(model.likelihood.ϵ*I,model.nFeature)]).*model.y)
            + logdet.(model.Knn.+[Diagonal(model.likelihood.ϵ*I,model.nFeature)])
            .+ model.nFeature*log(2.0π))
end

function ELBO(model::SVGP{<:GaussianLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::SVGP{GaussianLikelihood{T}}) where T
    return -0.5*(model.nSamplesUsed*log(2π*model.likelihood.ϵ) +
                (sum((model.y[model.MBIndices]-model.κ*model.μ).^2) +
                sum(model.K̃)+sum((model.κ*model.Σ).*model.κ))/model.likelihood.ϵ)
end

function hyperparameter_gradient_function(model::VGP{<:GaussianLikelihood})
    model.Σ = inv.(model.invKnn.+[model.likelihood.ϵ*I])
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
