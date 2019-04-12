"""Compute the KL Divergence between the GP Prior and the variational distribution for the variational full batch model"""
function GaussianKL(model::VGP)
    return 0.5*sum(opt_trace.(model.invKnn,model.Σ+model.μ.*transpose.(model.μ)).-model.nSample.-logdet.(model.Σ).-logdet.(model.invKnn))
end

"""Compute the KL Divergence between the Sparse GP Prior and the variational distribution for the sparse variational model"""
function GaussianKL(model::SVGP)
    return 0.5*sum(opt_trace.(model.invKmm,model.Σ+model.μ.*transpose.(model.μ)).-model.nFeature.-logdet.(model.Σ).-logdet.(model.invKmm))
end


""" Compute the equivalent of KL divergence between an improper prior and a variational Gamma distribution"""
function GammaImproperKL(model::AbstractGP)
    return model.inference.ρ*sum(-model.likelihood.α.+log(model.likelihood.β[1]).-lgamma.(model.likelihood.α).-(1.0.-model.likelihood.α).*digamma.(model.likelihood.α))
end

"""Compute KL divergence for Inverse-Gamma variables"""
function InverseGammaKL(model::AbstractGP{<:StudentTLikelihood})
    α_p = model.likelihood.ν/2; β_p= α_p*model.likelihood.σ
    return sum(broadcast(β->(α_p-model.likelihood.α)*digamma(α_p)-log(gamma(α_p))+log(gamma(model.likelihood.α))
            + model.likelihood.α*sum(log(β_p).-log.(β))+α_p*sum(β.-β_p)/β_p,model.likelihood.ω))
end

"""Compute KL divergence for Poisson variables"""
function PoissonKL(model::AbstractGP)
    return model.inference.ρ*sum(γ->sum(xlogx.(γ).+γ.*(-1.0.-digamma.(model.likelihood.α).+log.(model.likelihood.β))+model.likelihood.α./model.likelihood.β),model.likelihood.γ)
end

"""Compute KL divergence for Polya-Gamma variables in the binary setting"""
function PolyaGammaKL(model::AbstractGP{<:LogisticLikelihood})
    return model.inference.ρ*sum(broadcast((c,θ)->sum(-0.5*c.^2 .* θ .+ logcosh.(0.5.*c)),model.likelihood.c,model.likelihood.θ))
end

"""Compute KL divergence for Polya-Gamma variables in the multi-class setting"""
function PolyaGammaKL(model::VGP{<:LogisticSoftMaxLikelihood})
    return sum(broadcast((y,γ,c,θ)->sum((y+γ).*logcosh.(0.5.*c)-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end

"""Compute KL divergence for Polya-Gamma variables in the sparse multi-class setting"""
function PolyaGammaKL(model::SVGP{<:LogisticSoftMaxLikelihood})
    return model.inference.ρ*sum(broadcast((y,γ,c,θ)->sum((y[model.inference.MBIndices]+γ).*logcosh.(0.5.*c)-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end

"""Compute Entropy for Generalized inverse Gaussian latent variables (BayesianSVM)"""
function GIGEntropy(model::AbstractGP{<:BayesianSVM})
    return model.inference.ρ*sum(broadcast(b->0.5*sum(b)+sum(log.(2.0*besselk.(0.5,sqrt.(b))))-0.5*sum(sqrt.(b)),model.likelihood.ω))
end

function GIGEntropy(model::AbstractGP{<:LaplaceLikelihood})
end
