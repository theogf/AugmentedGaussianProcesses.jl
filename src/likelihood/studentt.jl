@doc raw"""
    StudentTLikelihood(ν::T, σ::Real=one(T))

## Arguments
- `ν::Real` : degrees of freedom of the student-T
- `σ::Real` : standard deviation of the local scale 

[Student-t likelihood](https://en.wikipedia.org/wiki/Student%27s_t-distribution) for regression:
```math
    p(y|f,ν,σ) = Γ(0.5(ν+1))/(\sqrt(νπ) σ Γ(0.5ν)) * (1+(y-f)^2/(σ^2ν))^{(-0.5(ν+1))},
```
where `ν` is the number of degrees of freedom and `σ` is the standard deviation for local scale of the data.

---

For the augmented analytical solution, it is augmented via:
```math
    p(y|f,ω) = N(y|f,σ^2 ω)
```
Where ``\omega \sim \mathcal{IG}(0.5\nu,0.5\nu)` where ``\mathcal{IG}`` is the inverse-gamma distribution.
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
struct StudentTLikelihood{T<:Real} <: RegressionLikelihood{T}
    ν::T
    α::T
    σ::T
    function StudentTLikelihood{T}(ν::T, σ::T=one(T)) where {T<:Real}
        ν > 0.5 || error("ν should be greater than 0.5")
        return new{T}(ν, (ν + one(T)) / 2.0, σ)
    end
end

function StudentTLikelihood(ν::T, σ::Real=one(T)) where {T<:Real}
    return StudentTLikelihood{T}(ν, σ)
end

function implemented(
    ::StudentTLikelihood, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function (l::StudentTLikelihood)(y::Real, f::Real)
    return gamma(l.α) / (sqrt(l.ν * π) * gamma(l.ν / 2)) * (1 + abs2((y - f) / l.σ))^(-l.α)
    # tdistpdf(l.ν, (y - f) / l.σ) uses R so not differentiable
end

function Distributions.loglikelihood(l::StudentTLikelihood, y::Real, f::Real)
    return log(l(y, f))
    # tdistlogpdf(l.ν, (y - f) / l.σ) uses R so not differentiable
end

function Base.show(io::IO, l::StudentTLikelihood{T}) where {T}
    return print(io, "Student-t likelihood (ν=", l.ν, ", σ=", l.σ, ")")
end

function compute_proba(
    l::StudentTLikelihood{T}, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
) where {T<:Real}
    return μ, max.(σ², zero(σ²)) .+ 0.5 * l.ν * l.σ^2 / (0.5 * l.ν - 1)
end

## Local Updates ##
function init_local_vars(state, ::StudentTLikelihood{T}, batchsize::Int) where {T}
    return merge(state, (; local_vars=(; c=rand(T, batchsize), θ=zeros(T, batchsize))))
end

function local_updates!(
    local_vars,
    l::StudentTLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    @. local_vars.c = 0.5 * (diag_cov + abs2(μ - y) + l.σ^2 * l.ν)
    @. local_vars.θ = l.α / l.c
    return local_vars
end

function sample_local!(
    l::StudentTLikelihood{T}, y::AbstractVector, f::AbstractVector
) where {T}
    l.c .= rand.(InverseGamma.(l.α, 0.5 * (abs2.(f - y) .+ l.σ^2 * l.ν)))
    set_ω!(l, inv.(l.c))
    return nothing
end

## Global Gradients ##

@inline ∇E_μ(::StudentTLikelihood, ::AOptimizer, y::AbstractVector, state) = (state.θ .* y,)
@inline function ∇E_Σ(::StudentTLikelihood, ::AOptimizer, ::AbstractVector, state)
    return (0.5 .* state.θ,)
end

## ELBO Section ##

function expec_loglikelihood(
    l::StudentTLikelihood{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
) where {T}
    tot = -0.5 * length(y) * (log(twoπ * l.σ^2))
    tot += -sum(log.(state.c) .- digamma(l.α))
    tot +=
        -0.5 * (
            dot(state.θ, diag_cov) + dot(state.θ, abs2.(μ)) - 2.0 * dot(state.θ, μ .* y) +
            dot(state.θ, abs2.(y))
        )
    return tot
end

AugmentedKL(l::StudentTLikelihood, ::AbstractVector, state) = InverseGammaKL(l, state)

function InverseGammaKL(l::StudentTLikelihood{T}, state) where {T}
    α_p = l.ν / 2
    β_p = α_p * l.σ^2
    return InverseGammaKL(l.α, state.c, α_p, β_p)
end

## PDF and Log PDF Gradients ## (verified gradients)

function ∇loglikelihood(l::StudentTLikelihood{T}, y::Real, f::Real) where {T<:Real}
    return (one(T) + l.ν) * (y - f) / ((f - y)^2 + l.σ^2 * l.ν)
end

function hessloglikelihood(l::StudentTLikelihood{T}, y::Real, f::Real) where {T<:Real}
    v = l.ν * l.σ^2
    Δ² = (f - y)^2
    return (one(T) + l.ν) * (-v + Δ²) / (v + Δ²)^2
end
