"""Compute the KL Divergence between the GP Prior and the variational distribution for the variational full batch model"""
function GaussianKL(model::VGP)
    return 0.5*sum(opt_trace.(model.invKnn,model.Σ+(model.μ.-model.μ₀).*transpose.(model.μ.-model.μ₀)).-model.nFeature.-logdet.(model.Σ).-logdet.(model.invKnn))
end

"""Compute the KL Divergence between the Sparse GP Prior and the variational distribution for the sparse variational model"""
function GaussianKL(model::SVGP)
    return 0.5*sum(opt_trace.(model.invKmm,model.Σ+(model.μ.-model.μ₀).*transpose.(model.μ.-model.μ₀)).-model.nFeature.-logdet.(model.Σ).-logdet.(model.invKmm))
end


""" Compute the equivalent of KL divergence between an improper prior and a variational Gamma distribution"""
function GammaImproperKL(model::AbstractGP)
    return model.inference.ρ*sum(-model.likelihood.α.+log(model.likelihood.β[1]).-lgamma.(model.likelihood.α).-(1.0.-model.likelihood.α).*digamma.(model.likelihood.α))
end

"""KL(q(ω)||p(ω)), where q(ω) = IG(α,β) and p(ω) = IG(α_p,β_p)"""
function InverseGammaKL(α,β,α_p,β_p)
    sum((α_p-α).*digamma(α_p) .- log.(gamma.(α_p)).+log.(gamma.(α)) .+  α.*(log.(β_p).-log.(β)).+α_p.*(β.-β_p)./β_p)
end

"""KL(q(ω)||p(ω)), where q(ω) = Po(γ) and p(ω) = Po(λ)"""
function PoissonKL(γ::AbstractVector{<:Real},λ::Real)
    λ*length(γ)-(1.0+log(λ))*sum(γ)+dot(γ,log.(γ))
end

"""KL(q(ω)||p(ω)), where q(ω) = Po(γ) and p(ω) = Po(λ)"""
function PoissonKL(γ::AbstractVector{<:Real},λ::AbstractVector{<:Real})
    sum(λ)-sum(γ)+dot(γ,log.(γ))-dot(γ,log.(λ))
end

"""KL(q(ω)||p(ω)), where q(ω) = PG(b,c) and p(ω) = PG(b,0). θ = 𝑬[ω]"""
function PolyaGammaKL(b,c,θ)
    -0.5*dot(c.^2,θ)-0.5*dot(b,logcosh.(0.5*c))
end

"""Compute Entropy for Generalized inverse Gaussian latent variables (BayesianSVM)"""
function GIGEntropy(model::AbstractGP{<:BayesianSVM})
    return model.inference.ρ*sum(broadcast(b->0.5*sum(log.(b))+sum(log.(2.0*besselk.(0.5,sqrt.(b))))-0.5*sum(sqrt.(b)),model.likelihood.ω))
end

"""Entropy of GIG variables with parameters a,b and p and omitting the derivative d/dpK_p cf <https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution#Entropy>"""
function GIGEntropy(a,b,p)
    sqrtab = sqrt.(a.*b)
    return sum(0.5*log.(a./b))+sum(log.(2*besselk.(p,sqrtab)))+ sum(0.5*sqrtab./besselk.(p,sqrtab).*(besselk.(p+1,sqrtab)+besselk.(p-1,sqrtab)))
end
