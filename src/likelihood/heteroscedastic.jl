@doc raw"""
    HeteroscedasticLikelihood(λ::T=1.0)

## Arguments
- `λ::Real` : The maximum precision possible (this is optimized during training)

---

Gaussian with heteroscedastic noise given by another gp:
```math
    p(y|f,g) = \mathcal{N}(y|f,(\lambda \sigma(g))^{-1})
```
Where ``\sigma`` is the logistic function

The augmentation is not trivial and will be described in a future paper
"""
struct HeteroscedasticLikelihood{T<:Real} <: RegressionLikelihood{T}
    λ::Vector{T}
    function HeteroscedasticLikelihood{T}(λ::T) where {T<:Real}
        return new{T}([λ])
    end
end

function HeteroscedasticLikelihood(λ::T=1.0) where {T<:Real}
    return HeteroscedasticLikelihood{T}(λ)
end

implemented(::HeteroscedasticLikelihood, ::AnalyticVI) = true

function (l::HeteroscedasticLikelihood)(y::Real, f::AbstractVector)
    return pdf(Normal(y, inv(sqrt(l.λ[1] * logistic(f[2])))), f[1])
end

function Distributions.loglikelihood(
    l::HeteroscedasticLikelihood, y::Real, f::AbstractVector
)
    return logpdf(Normal(y, inv(sqrt(l.λ[1] * logistic(f[2])))), f[1])
end

function Base.show(io::IO, ::HeteroscedasticLikelihood{T}) where {T}
    return print(io, "Gaussian likelihood with heteroscedastic noise")
end

n_latent(::HeteroscedasticLikelihood) = 2

function treat_labels!(y::AbstractVector{<:Real}, ::HeteroscedasticLikelihood)
    return y
end

function treat_labels!(::AbstractVector, ::HeteroscedasticLikelihood)
    return error("For regression, target(s) should be real valued")
end

function init_local_vars(
    state, ::HeteroscedasticLikelihood{T}, batchsize::Int
) where {T<:Real}
    local_vars = (;
        c=ones(T, batchsize),
        ϕ=ones(T, batchsize),
        γ=ones(T, batchsize),
        θ=ones(T, batchsize),
        σg=ones(T, batchsize),
    )
    return merge(state, (; local_vars))
end

function compute_proba(
    l::HeteroscedasticLikelihood{T},
    μ::AbstractVector{<:AbstractVector},
    Σ::AbstractVector{<:AbstractVector},
) where {T}
    return μ[1], max.(Σ[1], zero(T)) .+ inv.(l.λ[1] * logistic.(μ[2]))
end

function local_updates!(
    local_vars,
    l::HeteroscedasticLikelihood,
    y::AbstractVector,
    μ::NTuple{2,<:AbstractVector},
    diagΣ::NTuple{2,<:AbstractVector},
)
    # gp[1] is f and gp[2] is g (for approximating the noise)
    @. local_vars.ϕ = 0.5 * (abs2(μ[1] - y) + diagΣ[1])
    @. local_vars.c = sqrt(abs2(μ[2]) + diagΣ[2])
    @. local_vars.γ =
        0.5 * l.λ[1] * local_vars.ϕ * safe_expcosh(-0.5 * μ[2], 0.5 * local_vars.c)
    @. local_vars.θ = 0.5 * (0.5 + local_vars.γ) / local_vars.c * tanh(0.5 * local_vars.c)
    @. local_vars.σg = expectation(logistic, μ[2], diagΣ[2])
    l.λ .= 0.5 * length(local_vars.ϕ) / dot(local_vars.ϕ, local_vars.σg)
    return local_vars
end

function variational_updates!(
    m::AbstractGPModel{T,<:HeteroscedasticLikelihood,<:AnalyticVI}
) where {T}
    local_vars = local_updates!(
        state.local_vars, likelihood(m), yview(m), mean_f(m), var_f(m)
    )
    natural_gradient!(
        m.f[2],
        ∇E_μ(likelihood(m), opt_type(inference(m)), yview(m), local_vars)[2],
        ∇E_Σ(likelihood(m), opt_type(inference(m)), yview(m), local_vars)[2],
        ρ(m),
        opt_type(inference(m)),
        last(Zviews(m)),
        state.kernel_matrices[2],
        state.opt_state[2],
    )
    vi_opt_state_2 = global_update!(
        m.f[2], opt_type(inference(m)), inference(m), state.opt_state[2]
    )
    local_vars = heteroscedastic_expectations!(
        local_vars, likelihood(m), mean_f(m.f[2]), var_f(m.f[2])
    )
    natural_gradient!(
        m.f[1],
        ∇E_μ(likelihood(m), opt_type(inference(m)), yview(m), local_vars)[1],
        ∇E_Σ(likelihood(m), opt_type(inference(m)), yview(m), local_vars)[1],
        ρ(m),
        opt_type(inference(m)),
        first(Zviews(m)),
        state.kernel_matrices[1],
        state.opt_state[1],
    )
    vi_opt_state_1 = global_update!(
        m.f[1], opt_type(inference(m)), inference(m), state.opt_state[1]
    )
    opt_state = (vi_opt_state_1, vi_opt_state_2)
    return merge(state, (; opt_state))
end

function heteroscedastic_expectations!(
    local_vars, l::HeteroscedasticLikelihood{T}, μ::AbstractVector, Σ::AbstractVector
) where {T}
    @. local_vars.σg = expectation(logistic, μ, Σ)
    l.λ = 0.5 * length(local_vars.ϕ) / dot(local_vars.ϕ, local_vars.σg)
    return local_vars
end

function expectation(f::Function, μ::Real, σ²::Real)
    x = pred_nodes * sqrt(max(σ², zero(σ²))) .+ μ
    return dot(pred_weights, f.(x))
end

@inline function ∇E_μ(l::HeteroscedasticLikelihood, ::AOptimizer, y::AbstractVector, state)
    return (0.5 * y .* l.λ[1] .* state.σg, 0.5 * (0.5 .- state.γ))
end

@inline function ∇E_Σ(l::HeteroscedasticLikelihood, ::AOptimizer, ::AbstractVector, state)
    return (0.5 * l.λ[1] .* state.σg, 0.5 * state.θ)
end

function proba_y(
    model::AbstractGPModel{T,HeteroscedasticLikelihood{T},AnalyticVI{T}},
    X_test::AbstractMatrix{T},
) where {T<:Real}
    (μf, σ²f), (μg, σ²g) = predict_f(model, X_test; cov=true)
    return μf, σ²f + expectation.(x -> inv(model.likelihood.λ[1] * logistic(x)), μg, σ²g)
end

function expec_loglikelihood(
    l::HeteroscedasticLikelihood{T}, ::AnalyticVI, y::AbstractVector, μ, diag_cov, state
) where {T}
    tot = length(y) * (0.5 * log(l.λ[1]) - log(2 * sqrt(twoπ)))
    tot +=
        0.5 * (
            dot(μ[2], (0.5 .- state.γ)) - dot(abs2.(μ[2]), state.θ) -
            dot(diag_cov[2], state.θ)
        )
    tot -= PoissonKL(l, y, μ[1], diag_cov[1], state)
    return tot
end

AugmentedKL(l::HeteroscedasticLikelihood, state, ::Any) = PolyaGammaKL(l, state)

function PoissonKL(
    l::HeteroscedasticLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
    state,
) where {T}
    return PoissonKL(
        state.γ, 0.5 * l.λ[1] * (abs2.(y - μ) + Σ), log.(0.5 * l.λ[1] * (abs2.(μ - y) + Σ))
    )
end

function PolyaGammaKL(::HeteroscedasticLikelihood{T}, state) where {T}
    return PolyaGammaKL(0.5 .+ state.γ, state.c, state.θ)
end
