abstract type MaternLikelihood <: RegressionLikelihood end

"""
    Matern3_2Likelihood(ρ::Real=1.0)

## Arguments
- `ρ::Real` : lengthscale

---

Matern 3/2 likelihood for regression:
see [wiki page](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
---

For the analytical solution, it is augmented via:
```math
p(y|f,\\omega) = \\mathcal{N}(y|f,\\sigma^2\\omega)
```
Where ``\\omega \\sim \\mathcal{IG}(\\frac{\\nu}{2},\\frac{\\nu}{2})`` where ``\\mathcal{IG}`` is the inverse gamma distribution
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
struct Matern3_2Likelihood{T<:Real} <: MaternLikelihood
    ρ::T
end

function implemented(
    ::Matern3_2Likelihood, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function (l::Matern3_2Likelihood)(y::Real, f::Real)
    u = sqrt(3) * abs(y - f) / l.ρ
    return 4 * l.ρ / sqrt(3) * (one(T) + u) * exp(-u)
end

function Distributions.loglikelihood(l::Matern3_2Likelihood, y::Real, f::Real)
    u = sqrt(3) * abs(y - f) / l.ρ
    return log(4 * l.ρ / sqrt(3)) + log(one(T) + u) - u
end

function Base.show(io::IO, l::Matern3_2Likelihood)
    return print(io, "Matern 3/2 likelihood (ρ = $(l.ρ))")
end

function compute_proba(
    l::Matern3_2Likelihood, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
)
    return μ, max.(σ², zero(eltype(σ²))) .+ 4 * l.ρ^2 / 3
end

## Local Updates ##

function init_local_vars(::Matern3_2Likelihood, batchsize::Int, T::DataType=Float64)
    return (; c=rand(T, batchsize), θ=zeros(T, batchsize))
end

function local_updates!(
    local_vars,
    l::Matern3_2Likelihood,
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    map!(sqrt_expec_square, local_vars.c, μ, diagΣ, y) # √E[(y-f)^2]
    map!(local_vars.θ, local_vars.c) do c
        3 / (2 * sqrt(3) * c * l.ρ + 2 * l.ρ^2)
    end
end

function sample_local!(
    local_vars, l::Matern3_2Likelihood, y::AbstractVector, f::AbstractVector
)
    map!(local_vars.θ, f, y) do f, y
        rand(GeneralizedInverseGaussian(3 / (2 * l.ρ^2), 2 * abs2(f - y), 1.5))
    end
    return local_vars
end

@inline ∇E_μ(l::Matern3_2Likelihood, ::AOptimizer, y::AbstractVector) = (2 * l.θ .* y,)
@inline ∇E_Σ(l::Matern3_2Likelihood, ::AOptimizer, ::AbstractVector) = (l.θ,)

## ELBO  ##

function expecLogLikelihood(
    l::Matern3_2Likelihood,
    i::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
)
    return throw(error("Not implemented yet"))
end

function InverseGammaKL(l::Matern3_2Likelihood, state)
    α_p = l.ν / 2
    β_p = α_p * l.σ^2
    return model.inference.ρ * sum(broadcast(InverseGammaKL, l.α, l.β, α_p, β_p))
end

## PDF and Log PDF Gradients ## (verified gradients)

@inline function ∇loglikehood(l::Matern3_2Likelihood, y::Real, f::Real)
    return 3 * (y - f) / (l.ρ * (abs(f - y) * sqrt(3) + l.ρ))
end

@inline function hessloglikelihood(l::Matern3_2Likelihood, y::Real, f::Real)
    return 3 / (l.ρ + sqrt(3) * abs(f - y))^2
end
