abstract type Likelihood{T<:Real}  end

"""
Gaussian likelihood : ``p(y|f) = 𝓝(y|f,ϵ) ``
"""
struct GaussianLikelihood{T<:Real} <: Likelihood{T}
    ϵ::T
end

function GaussianLikelihood(ϵ::T=1e-3) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end
