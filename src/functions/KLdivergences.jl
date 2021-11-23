## KL Divergence between the GP Prior and the variational distribution
function GaussianKL(model::AbstractGPModel, state)
    return mapreduce(GaussianKL, +, model.f, Zviews(model), state.kernel_matrices)
end

function GaussianKL(gp::AbstractLatent, X::AbstractVector, k_mat)
    return GaussianKL(mean(gp), pr_mean(gp, X), cov(gp), k_mat.K)
end

## See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions ##
function GaussianKL(
    μ::AbstractVector,
    μ₀::AbstractVector,
    Σ::Symmetric{T,Matrix{T}},
    K::Cholesky{T,Matrix{T}},
) where {T<:Real}
    return (logdet(K) - logdet(Σ) + tr(K \ Σ) + invquad(K, μ - μ₀) - length(μ)) / 2
end

function GaussianKL(
    μ::AbstractVector{T},
    μ₀::AbstractVector,
    Σ::Symmetric{T,Matrix{T}},
    K::AbstractMatrix{T},
) where {T<:Real}
    K
    return (logdet(K) - logdet(Σ) + tr(K \ Σ) + dot(μ - μ₀, K \ (μ - μ₀)) - length(μ)) / 2
end

extraKL(::AbstractGPModel{T}, ::Any) where {T} = zero(T)

"""
    extraKL(model::OnlineSVGP)

Extra KL term containing the divergence with the GP at time t and t+1
"""
function extraKL(model::OnlineSVGP{T}, state) where {T}
    return mapreduce(
        +, model.f, state.opt_state, state.kernel_matrices
    ) do gp, opt_state, kernel_mat
        prev_gp = opt_state.previous_gp
        κₐμ = kernel_mat.κₐ * mean(gp)
        KLₐ = prev_gp.prev𝓛ₐ
        KLₐ +=
            -sum(
                trace_ABt.(
                    Ref(prev_gp.invDₐ),
                    [kernel_mat.K̃ₐ, kernel_mat.κₐ * cov(gp) * transpose(kernel_mat.κₐ)],
                ),
            ) / 2
        KLₐ += dot(prev_gp.prevη₁, κₐμ) - dot(κₐμ, prev_gp.invDₐ * κₐμ) / 2
        return KLₐ
    end
end

InverseGammaKL(α, β, αₚ, βₚ) = GammaKL(α, β, αₚ, βₚ)
"""
    GammaKL(α, β, αₚ, βₚ)

KL(q(ω)||p(ω)), where q(ω) = Ga(α,β) and p(ω) = Ga(αₚ,βₚ)
"""
function GammaKL(α, β, αₚ, βₚ)
    return sum(
        (α - αₚ) .* digamma(α) .- log.(gamma.(α)) .+ log.(gamma.(αₚ)) .+
        αₚ .* (log.(β) .- log.(βₚ)) .+ α .* (βₚ .- β) ./ β,
    )
end

"""
    PoissonKL(λ, λ₀)

KL(q(ω)||p(ω)), where q(ω) = Po(ω|λ) and p(ω) = Po(ω|λ₀)
"""
function PoissonKL(λ::AbstractVector{T}, λ₀::Real) where {T}
    return λ₀ * length(λ) - (one(T) + log(λ₀)) * sum(λ) + sum(xlogx, λ)
end

"""
    PoissonKL(λ, λ₀, ψ)

KL(q(ω)||p(ω)), where q(ω) = Po(ω|λ) and p(ω) = Po(ω|λ₀) with ψ = E[log(λ₀)]
"""
function PoissonKL(
    λ::AbstractVector{<:Real}, λ₀::AbstractVector{<:Real}, ψ::AbstractVector{<:Real}
)
    # sum(λ₀) - sum(λ) + sum(xlogx, λ) - dot(λ, ψ)
    # sum(λ₀) - sum(λ) + mapreduce(xlogx, +, λ) - dot(λ, ψ)
    return sum(λ₀) - sum(λ) + sum(xlogx.(λ)) - dot(λ, ψ)
end

"""
    PolyaGammaKL(b, c, θ)

KL(q(ω)||p(ω)), where q(ω) = PG(b,c) and p(ω) = PG(b,0). θ = 𝑬[ω]
"""
function PolyaGammaKL(b, c, θ)
    return dot(b, logcosh.(c / 2)) - dot(abs2.(c), θ) / 2
end

"""
    GIGEntropy(a, b, p)
    
Entropy of GIG variables with parameters a,b and p and omitting the derivative d/dpK_p cf <https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution#Entropy>
"""
function GIGEntropy(a, b, p)
    sqrt_ab = sqrt.(a .* b)
    return (sum(log, a) - sum(log, b)) / 2 +
           mapreduce((p, s) -> log(2 * besselk(p, s)), +, p, sqrt_ab) +
           sum(
               sqrt_ab ./ besselk.(p, sqrt_ab) .*
               (besselk.(p + 1, sqrt_ab) + besselk.(p - 1, sqrt_ab)),
           ) / 2
end
