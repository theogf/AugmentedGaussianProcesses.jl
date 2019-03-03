"""Compute the KL Divergence between the GP Prior and the variational distribution for the full batch model"""
function GaussianKL(model::VGP)
    return 0.5*sum(opt_trace.(model.invKnn,model.Σ+model.μ.*transpose.(model.μ)).-model.nSample.-logdet.(model.Σ).-logdet.(model.invKnn))
end

"""Compute the KL Divergence between the Sparse GP Prior and the variational distribution for the sparse model"""
function GaussianKL(model::SVGP)
    return 0.5*sum(opt_trace.(model.invKmm,model.Σ+model.μ.*transpose.(model.μ)).-model.nFeature.-logdet.(model.Σ).-logdet.(model.invKmm))
end

""" Compute KL divergence for Polya-Gamma variables (in the binary case)"""
function PolyaGammaKL(model::GP)
    return model.inference.ρ*sum(broadcast((c,θ)->-0.5*c.^2 .* θ .+ log.(cosh.(0.5.*c)),model.c,model.θ))
end

function GammaImproperKL(model::GP)
    return model.inference.ρ*sum(-model.likelihood.α.+log(model.likelihood.β[1]).-lgamma.(model.likelihood.α).-(1.0.-model.likelihood.α).*digamma.(model.likelihood.α))
end

"""Return the KL divergence for the inverse gamma distributions"""
function InverseGammaKL(model::GP)
    α_p = β_p = model.likelihood.ν/2;
    return (α_p-model.likelihood.α)*digamma(α_p).-log(gamma(α_p)).+log(gamma(model.likelihood.α))
            .+ model.α*(log(β_p).-log.(model.β)).+α_p.*(model.β.-β_p)/β_p
end

function PoissonKL(model::GP)
    return model.inference.ρ*sum(γ->sum(xlogx.(γ).+γ.*(-1.0.-digamma.(model.likelihood.α).+log.(model.likelihood.β))+model.likelihood.α./model.likelihood.β),model.likelihood.γ)
end

function PolyaGammaKL(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
    return sum(broadcast((y,γ,c,θ)->sum((y+γ).*logcosh.(0.5.*c)-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end

function PolyaGammaKL(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
    return model.inference.ρ*sum(broadcast((y,γ,c,θ)->sum((y[model.inference.MBIndices]+γ).*logcosh.(0.5.*c)-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end
