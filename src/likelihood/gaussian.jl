"""
Gaussian likelihood : $$ p(y|f) = \mathcal{N}(y|f,noise) $$
"""
struct GaussianLikelihood <: Likelihood
    noise::Float64
end

function local_update!(model::SVGP{GaussianLikelihood})
    model.likelihood.noise = 1.0/model.nSamplesUsed * ( dot(model.y[model.MBIndices],model.y[model.MBIndices])
    - 2.0*dot(model.y[model.MBIndices],model.κ*model.μ)
    + sum((model.κ'*model.κ).*(model.μ*model.μ'+model.Σ)) + sum(model.Ktilde) )
end

function natural_gradient(model::SVGP{GaussianLikelihood})
    grad_1 = model.StochCoeff.*(model.κ'*model.y[model.MBIndices])./model.gnoise
    grad_2 = Symmetric(-0.5*(model.StochCoeff*(model.κ')*model.κ./model.gnoise+model.invKmm))
    return (grad_1,grad_2)
end

function ELBO(model::Union{VGP{T},SVGP{T}}) where {T<:GaussianLikelihood}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{GaussianLikelihood})
    return -0.5*dot(model.y,model.invKnn*model.y) #TODO to correct (missing noise etc)
end

function expecLogLikelihood(model::SVGP{GaussianLikelihood})
    return -0.5*(model.nSamplesUsed*log(2π*model.gnoise) +
                (sum((model.y[model.MBIndices]-model.κ*model.μ).^2) +
                sum(model.K̃)+sum((model.κ*model.Σ).*model.κ))/model.gnoise)
end
