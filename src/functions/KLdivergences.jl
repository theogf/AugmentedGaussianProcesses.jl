"""
    KL Divergence between the GP Prior and the variational distribution
"""
GaussianKL(model::AbstractGP) = mapreduce(GaussianKL, +, model.f, Zviews(model))

GaussianKL(gp::AbstractLatent, X::AbstractVector) = GaussianKL(mean(gp), pr_mean(gp, X), cov(gp), pr_cov(gp))

## See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions ##
function GaussianKL(
    μ::AbstractVector{T},
    μ₀::AbstractVector,
    Σ::Symmetric{T,Matrix{T}},
    K::PDMat{T,Matrix{T}},
) where {T<:Real}
    0.5 * (logdet(K) - logdet(Σ) + tr(K \ Σ) + invquad(K, μ - μ₀) - length(μ))
end

extraKL(::AbstractGP{T}) where {T} = zero(T)

"""
    Extra KL term containing the divergence with the GP at time t and t+1
"""
function extraKL(model::OnlineSVGP{T}) where {T}
    KLₐ = zero(T)
    for gp in model.f
        κₐμ = gp.κₐ * mean(gp)
        KLₐ += gp.prev𝓛ₐ
        KLₐ += -0.5 *  sum(trace_ABt.(Ref(gp.invDₐ), [gp.K̃ₐ, gp.κₐ * cov(gp) * transpose(gp.κₐ)]))
        KLₐ += dot(gp.prevη₁, κₐμ) - 0.5 * dot(κₐμ, gp.invDₐ * κₐμ)
    end
    return KLₐ
end

InverseGammaKL(α, β, αₚ, βₚ) = GammaKL(α, β, αₚ, βₚ)
"""
    KL(q(ω)||p(ω)), where q(ω) = Ga(α,β) and p(ω) = Ga(αₚ,βₚ)
"""
function GammaKL(α, β, αₚ, βₚ)
    sum(
        (α - αₚ) .* digamma(α) .- log.(gamma.(α)) .+ log.(gamma.(αₚ)) .+
        αₚ .* (log.(β) .- log.(βₚ)) .+ α .* (βₚ .- β) ./ β,
    )
end

"""
    KL(q(ω)||p(ω)), where q(ω) = Po(ω|λ) and p(ω) = Po(ω|λ₀)
"""
function PoissonKL(λ::AbstractVector{T}, λ₀::Real) where {T}
    λ₀ * length(λ) - (one(T) + log(λ₀)) * sum(λ) + sum(xlogx, λ)
end

"""
    KL(q(ω)||p(ω)), where q(ω) = Po(ω|λ) and p(ω) = Po(ω|λ₀) with ψ = E[log(λ₀)]
"""
function PoissonKL(
    λ::AbstractVector{<:Real},
    λ₀::AbstractVector{<:Real},
    ψ::AbstractVector{<:Real},
)
    sum(λ₀) - sum(λ) + sum(xlogx, λ) - dot(λ, ψ)
end


"""
    KL(q(ω)||p(ω)), where q(ω) = PG(b,c) and p(ω) = PG(b,0). θ = 𝑬[ω]
"""
function PolyaGammaKL(b, c, θ)
    dot(b, logcosh.(0.5 * c)) - 0.5 * dot(abs2.(c), θ)
end

"""
    Entropy of GIG variables with parameters a,b and p and omitting the derivative d/dpK_p cf <https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution#Entropy>
"""
function GIGEntropy(a, b, p)
    sqrt_ab = sqrt.(a .* b)
    return 0.5 * (sum(log, a) - sum(log, b)) +
           mapreduce((p, s) -> log(2 * besselk(p, s)), +, p, sqrt_ab) +
           sum(
               0.5 * sqrt_ab ./ besselk.(p, sqrt_ab) .*
               (besselk.(p + 1, sqrt_ab) + besselk.(p - 1, sqrt_ab)),
           )
end
