"""
```julia
HeteroscedasticLikelihood(λ::T=1.0)
```

Gaussian with heteroscedastic noise given by another gp:
```math
    p(y|f,g) = N(y|f,(λ σ(g))⁻¹)
```
Where `σ` is the logistic function

Augmentation will be described in a future paper
"""
mutable struct HeteroscedasticLikelihood{T<:Real, A<:AbstractVector{T}} <: RegressionLikelihood{T}
    λ::T
    c::A
    ϕ::A
    γ::A
    θ::A
    σg::A
    function HeteroscedasticLikelihood{T}(λ::T) where {T<:Real}
        new{T, Vector{T}}(λ)
    end
    function HeteroscedasticLikelihood{T}(
        λ::T,
        c::A,
        ϕ::A,
        γ::A,
        θ::A,
        σg::A,
    ) where {T<:Real, A<:AbstractVector{T}}
        new{T,A}(λ, c, ϕ, γ, θ, σg)
    end
end

function HeteroscedasticLikelihood(λ::T = 1.0) where {T<:Real}
    HeteroscedasticLikelihood{T}(λ)
end

implemented(::HeteroscedasticLikelihood, ::AnalyticVI) = true

function pdf(l::HeteroscedasticLikelihood, y::Real, f::AbstractVector)
    pdf(Normal(y, inv(sqrt(l.λ * logistic(f[2])))), f[1])
end

function logpdf(l::HeteroscedasticLikelihood, y::Real, f::AbstractVector)
    logpdf(Normal(y, inv(sqrt(l.λ * logistic(f[2])))), f[1])
end

function Base.show(io::IO, model::HeteroscedasticLikelihood{T}) where {T}
    print(io, "Gaussian likelihood with heteroscedastic noise")
end

num_latent(::HeteroscedasticLikelihood) = 2

function treat_labels!(
    y::AbstractVector{<:Real},
    likelihood::HeteroscedasticLikelihood,
)
    return y, 2, likelihood
end

function treat_labels!(y::AbstractVector, likelihood::HeteroscedasticLikelihood)
    error("For regression target(s) should be real valued")
end

function init_likelihood(
    likelihood::HeteroscedasticLikelihood{T},
    inference::Inference{T},
    nLatent::Int,
    nMinibatch::Int,
) where {T<:Real}
    λ = likelihood.λ
    c = ones(T, nMinibatch)
    ϕ = ones(T, nMinibatch)
    γ = ones(T, nMinibatch)
    θ = ones(T, nMinibatch)
    σg = ones(T, nMinibatch)
    HeteroscedasticLikelihood{T}(λ, c, ϕ, γ, θ, σg)
end

function compute_proba(
    l::HeteroscedasticLikelihood{T},
    μ::AbstractVector{<:AbstractVector},
    Σ::AbstractVector{<:AbstractVector},
) where {T}
    return μ[1], max.(Σ[1], zero(T)) .+ inv.(l.λ * logistic.(μ[2]))
end

function local_updates!(
    l::HeteroscedasticLikelihood,
    y::AbstractVector,
    μ::NTuple{2,<:AbstractVector},
    diagΣ::NTuple{2,<:AbstractVector},
)
    # gp[1] is f and gp[2] is g (for approximating the noise)
    @. l.ϕ = 0.5 * (abs2(μ[1] - y) + diagΣ[1])
    @. l.c = sqrt(abs2(μ[2]) + diagΣ[2])
    @. l.γ = 0.5 * l.λ * l.ϕ * safe_expcosh(-0.5 * μ[2], 0.5 * l.c)
    @. l.θ = 0.5 * (0.5 + l.γ) / l.c * tanh(0.5 * l.c)
    @. l.σg = expectation(logistic, μ[2], diagΣ[2])
    l.λ = 0.5 * length(l.ϕ) / dot(l.ϕ, l.σg)
end

function variational_updates!(
    m::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI},
) where {T}
    local_updates!(
        likelihood(m),
        yview(m),
        mean_f(m),
        var_f(m),
    )
    natural_gradient!(
        ∇E_μ(likelihood(m), opt_type(inference(m)), yview(m))[2],
        ∇E_Σ(likelihood(m), opt_type(inference(m)), yview(m))[2],
        getρ(inference(m)),
        opt_type(inference(m)),
        last(Zviews(m)),
        m.f[2],
    )
    global_update!(m.f[2], opt_type(inference(m)), inference(m))
    heteroscedastic_expectations!(
        likelihood(m),
        mean_f(m.f[2]),
        var_f(m.f[2]),
    )
    natural_gradient!(
        ∇E_μ(likelihood(m), opt_type(inference(m)), yview(m))[1],
        ∇E_Σ(likelihood(m), opt_type(inference(m)), yview(m))[1],
        getρ(inference(m)),
        opt_type(inference(m)),
        first(Zviews(m)),
        m.f[1],
    )
    global_update!(m.f[1], opt_type(inference(m)), inference(m))
end

function heteroscedastic_expectations!(
    l::HeteroscedasticLikelihood{T},
    μ::AbstractVector,
    Σ::AbstractVector,
) where {T}
    @. l.σg = expectation(logistic, μ, Σ)
    l.λ = 0.5 * length(l.ϕ) / dot(l.ϕ, l.σg)
end

function expectation(f::Function, μ::Real, σ²::Real)
    x = pred_nodes * sqrt(max(σ², zero(σ²))) .+ μ
    dot(pred_weights, f.(x))
end

@inline ∇E_μ(
    l::HeteroscedasticLikelihood,
    ::AOptimizer,
    y::AbstractVector,
) = (0.5 * y .* l.λ .* l.σg, 0.5 * (0.5 .- l.γ))

@inline ∇E_Σ(
    l::HeteroscedasticLikelihood,
    ::AOptimizer,
    ::AbstractVector,
) = (0.5 * l.λ .* l.σg, 0.5 * l.θ)

function proba_y(
    model::AbstractGP{T,HeteroscedasticLikelihood{T},AnalyticVI{T}},
    X_test::AbstractMatrix{T},
) where {T<:Real}
    (μf, σ²f), (μg, σ²g) = predict_f(model, X_test, covf = true)
    return μf,
    σ²f + expectation.(x -> inv(model.likelihood.λ * logistic(x)), μg, σ²g)
end

function expec_log_likelihood(
    l::HeteroscedasticLikelihood{T},
    i::AnalyticVI,
    y::AbstractVector,
    μ,
    diag_cov,
) where {T}
    tot = length(y) * (0.5 * log(l.λ) - log(2 * sqrt(twoπ)))
    tot +=
        0.5 * (
            dot(μ[2], (0.5 .- l.γ)) - dot(abs2.(μ[2]), l.θ) -
            dot(diag_cov[2], l.θ)
        )
    tot -= PoissonKL(l, y, μ[1], diag_cov[1])
    return tot
end

AugmentedKL(l::HeteroscedasticLikelihood, ::AbstractVector) = PolyaGammaKL(l)

function PoissonKL(
    l::HeteroscedasticLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
) where {T}
    return PoissonKL(
        l.γ,
        0.5 * l.λ * (abs2.(y - μ) + Σ),
        log.(0.5 * l.λ * (abs2.(μ - y) + Σ)),
    )
end

function PolyaGammaKL(l::HeteroscedasticLikelihood{T}) where {T}
    PolyaGammaKL(0.5 .+ l.γ, l.c, l.θ)
end
