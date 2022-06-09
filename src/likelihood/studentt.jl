@doc raw"""
    StudentTLikelihood(ν::T, σ::Real=one(T))

## Arguments
- `ν::Real` : degrees of freedom of the student-T
- `σ::Real` : standard deviation of the local scale 

[Student-t likelihood](https://en.wikipedia.org/wiki/Student%27s_t-distribution) for regression:
```math
    p(y|f,ν,σ) = \frac{Γ(\frac{ν+1}{2})}{\sqrt(νπ) σ Γ(\frac{ν}{2})} (1+\frac{(y-f)^2}{σ^2ν})^{(-\frac{ν+1}{2})},
```
where `ν` is the number of degrees of freedom and `σ` is the standard deviation for local scale of the data.

---

For the augmented analytical solution, it is augmented via:
```math
    p(y|f,\omega) = N(y|f,\sigma^2 \omega)
```
Where ``\omega \sim \mathcal{IG}(\frac{\nu}{2},\frac{\nu}{2})`` where ``\mathcal{IG}`` is the inverse-gamma distribution.
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
struct StudentTLikelihood{T<:Real} <: RegressionLikelihood
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

function Base.show(io::IO, l::StudentTLikelihood)
    return print(io, "Student-t likelihood (ν=", l.ν, ", σ=", l.σ, ")")
end

function compute_proba(
    l::StudentTLikelihood, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
)
    return μ, max.(σ², zero(eltype(σ²))) .+ l.ν * l.σ^2 / 2(l.ν / 2 - 1)
end

## Local Updates ##
function init_local_vars(::StudentTLikelihood, batchsize::Int, T::DataType=Float64)
    return (; c=rand(T, batchsize), θ=zeros(T, batchsize))
end

function local_updates!(
    local_vars,
    l::StudentTLikelihood,
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    map!(local_vars.c, μ, diagΣ, y) do μ, σ², y
        (abs2(μ - y) + σ² + l.σ^2 * l.ν) / 2
    end
    map!(local_vars.θ, local_vars.c) do c
        l.α / c
    end
    return local_vars
end

function sample_local!(
    local_vars, l::StudentTLikelihood, y::AbstractVector, f::AbstractVector
)
    map!(local_vars.c, f, y) do f, y
        rand(InverseGamma(l.α, (abs2(f - y) + l.σ^2 * l.ν) / 2))
    end # sample ω
    map!(inv, local_vars.θ, local_vars.c)
    return local_vars
end

## Global Gradients ##

@inline ∇E_μ(::StudentTLikelihood, ::AOptimizer, y::AbstractVector, state) = (state.θ .* y,)
@inline function ∇E_Σ(::StudentTLikelihood, ::AOptimizer, ::AbstractVector, state)
    return (state.θ / 2,)
end

## ELBO Section ##

function expec_loglikelihood(
    l::StudentTLikelihood,
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
)
    tot = -length(y) * (log(twoπ * l.σ^2)) / 2
    tot += -sum(log.(state.c) .- digamma(l.α))
    tot +=
        -(
            dot(state.θ, diag_cov) + dot(state.θ, abs2.(μ)) - 2.0 * dot(state.θ, μ .* y) +
            dot(state.θ, abs2.(y))
        ) / 2
    return tot
end

AugmentedKL(l::StudentTLikelihood, state, ::Any) = InverseGammaKL(l, state)

function InverseGammaKL(l::StudentTLikelihood, state)
    α_p = l.ν / 2
    β_p = α_p * l.σ^2
    return InverseGammaKL(l.α, state.c, α_p, β_p)
end

## PDF and Log PDF Gradients ## (verified gradients)

function ∇loglikelihood(l::StudentTLikelihood, y::Real, f::Real)
    return (one(l.ν) + l.ν) * (y - f) / ((f - y)^2 + l.σ^2 * l.ν)
end

function hessloglikelihood(l::StudentTLikelihood, y::Real, f::Real)
    v = l.ν * l.σ^2
    Δ² = (f - y)^2
    return (one(l.ν) + l.ν) * (-v + Δ²) / (v + Δ²)^2
end
