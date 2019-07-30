"""Compute the KL Divergence between the GP Prior and the variational distribution for the variational full batch model"""
function GaussianKL(model::VGP)
    return sum(broadcast(GaussianKL,model.μ,model.μ₀,model.Σ,model.invKnn))
end

"""Compute the KL Divergence between the Sparse GP Prior and the variational distribution for the sparse variational model"""
function GaussianKL(model::SVGP)
    return sum(broadcast(GaussianKL,model.μ,model.μ₀,model.Σ,model.invKmm))
end


function GaussianKL(μ::AbstractVector{T},μ₀::PriorMean,Σ::Symmetric{T,Matrix{T}},invK::Symmetric{T,Matrix{T}}) where {T<:Real}
    0.5*(-logdet(Σ)-logdet(invK)+opt_trace(invK,Σ)+dot(μ-μ₀,invK*(μ-μ₀))-length(μ))
end

"""
    Compute the equivalent of KL divergence between an improper prior p(λ) (``1_{[0,\\infty]}``) and a variational Gamma distribution
"""
function GammaEntropy(model::AbstractGP)
    return model.inference.ρ*(-sum(model.likelihood.α)+sum(log,model.likelihood.β[1])-sum(lgamma,model.likelihood.α)-dot(1.0.-model.likelihood.α,digamma.(model.likelihood.α)))
end

"""
    KL(q(ω)||p(ω)), where q(ω) = IG(α,β) and p(ω) = IG(α_p,β_p)
"""
function InverseGammaKL(α,β,α_p,β_p)
    sum((α_p-α).*digamma(α_p) .- log.(gamma.(α_p)).+log.(gamma.(α)) .+  α.*(log.(β_p).-log.(β)).+α_p.*(β.-β_p)./β_p)
end

"""
    KL(q(ω)||p(ω)), where q(ω) = Po(ω|γ) and p(ω) = Po(ω|λ)
"""
function PoissonKL(γ::AbstractVector{<:Real},λ::Real)
    λ*length(γ)-(1.0+log(λ))*sum(γ)+sum(xlogx,γ)
end

"""
    KL(q(ω)||p(ω)), where q(ω) = Po(ω|γ) and p(ω) = Po(ω|λ) with ν = E[λ] and ψ = E[log(λ)]
"""
function PoissonKL(γ::AbstractVector{<:Real},ν::AbstractVector{<:Real},ψ::AbstractVector{<:Real})
    sum(ν)-sum(γ)+sum(xlogx,γ)-dot(γ,ψ)
end


"""KL(q(ω)||p(ω)), where q(ω) = PG(b,c) and p(ω) = PG(b,0). θ = 𝑬[ω]"""
function PolyaGammaKL(b,c,θ)
    dot(b,logcosh.(0.5*c))-0.5*dot(abs2.(c),θ)
end

"""Compute Entropy for Generalized inverse Gaussian latent variables (BayesianSVM)"""
function GIGEntropy(model::AbstractGP{T,<:BayesianSVM}) where {T}
    return model.inference.ρ*sum(broadcast(b->0.5*sum(log.(b))+sum(log.(2.0*besselk.(0.5,sqrt.(b))))-0.5*sum(sqrt.(b)),model.likelihood.ω))
end

"""Entropy of GIG variables with parameters a,b and p and omitting the derivative d/dpK_p cf <https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution#Entropy>"""
function GIGEntropy(a,b,p)
    sqrtab = sqrt.(a.*b)
    return sum(0.5*log.(a./b))+sum(log.(2*besselk.(p,sqrtab)))+ sum(0.5*sqrtab./besselk.(p,sqrtab).*(besselk.(p+1,sqrtab)+besselk.(p-1,sqrtab)))
end
