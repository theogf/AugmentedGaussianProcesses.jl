function checklabels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:Union{GaussianLikelihood}}
    @assert T<:Real "For regression target(s) should be real valued"
end

function local_update!(model::VGP{GaussianLikelihood})
end

function local_update!(model::SVGP{GaussianLikelihood})
    model.likelihood.ϵ = 1.0/model.nSamplesUsed * ( dot(model.y[model.MBIndices],model.y[model.MBIndices])
    - 2.0*dot(model.y[model.MBIndices],model.κ*model.μ)
    + sum((model.κ'*model.κ).*(model.μ*model.μ'+model.Σ)) + sum(model.Ktilde) )
end

function natural_gradient!(model::SVGP{GaussianLikelihood})
    model.∇η₁ .= model.likelihood.ρ.*(model.κ'*model.y[model.MBIndices])./model.likelihood.ϵ - model.η₁
    model.∇η₁ = Symmetric(-0.5*(model.likelihood.ρ*(model.κ')*model.κ./model.likelihood.ϵ+model.invKmm) - model.η₂)
end

function ELBO(model::Union{VGP{L},SVGP{L}}) where {L<:GaussianLikelihood}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{GaussianLikelihood})
    return -0.5*dot(model.y,model.invKnn*model.y) #TODO to correct (missing noise etc)
end

function expecLogLikelihood(model::SVGP{GaussianLikelihood})
    return -0.5*(model.nSamplesUsed*log(2π*model.likelihood.ϵ) +
                (sum((model.y[model.MBIndices]-model.κ*model.μ).^2) +
                sum(model.K̃)+sum((model.κ*model.Σ).*model.κ))/model.likelihood.ϵ)
end
