@doc raw"""
    HeteroscedasticLikelihood(λ::T=1.0)->HeteroscedasticGaussianLikelihood

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
HeteroscedasticLikelihood(λ::Real) = HeteroscedasticGaussianLikelihood(InvScaledLogistic(λ))

struct InvScaledLogistic{T} <: AbstractLink
    λ::Vector{T}
end

InvScaledLogistic(λ::Real) = InvScaledLogistic([λ])

(l::InvScaledLogistic)(f::Real) = inv(l.λ[1] * logistic(f))

implemented(::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic}, ::AnalyticVI) = true

function (l::HeteroscedasticGaussianLikelihood)(f, y::Real)
    return pdf(l(f), y)
end

function Distributions.loglikelihood(
    l::HeteroscedasticGaussianLikelihood, f::AbstractVector, y
)
    return logpdf(l(f), y)
end

function Base.show(io::IO, ::HeteroscedasticGaussianLikelihood)
    return print(io, "Gaussian likelihood with heteroscedastic noise")
end

n_latent(::HeteroscedasticGaussianLikelihood) = 2

function init_local_vars(
    state,
    ::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    batchsize::Int,
    T::DataType=Float64,
)
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
    l::HeteroscedasticGaussianLikelihood,
    μ::AbstractVector{<:AbstractVector},
    Σ::AbstractVector{<:AbstractVector},
)
    return μ[1], Σ[1] .+ l.link(μ[2])
end

function local_updates!(
    local_vars,
    l::HeteroscedasticGaussianLikelihood,
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
    m::AbstractGPModel{
        T,<:HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},<:AnalyticVI
    },
) where {T}
    local_vars = local_updates!(
        state.local_vars, likelihood(m), yview(m), mean_f(m), var_f(m)
    )
    natural_gradient!(
        m.f[2],
        ∇E_μ(likelihood(m), opt(inference(m)), yview(m), local_vars)[2],
        ∇E_Σ(likelihood(m), opt(inference(m)), yview(m), local_vars)[2],
        ρ(m),
        opt(inference(m)),
        last(Zviews(m)),
        state.kernel_matrices[2],
        state.opt_state[2],
    )
    vi_opt_state_2 = global_update!(
        m.f[2], opt(inference(m)), inference(m), state.opt_state[2]
    )
    local_vars = heteroscedastic_expectations!(
        local_vars, likelihood(m), mean_f(m.f[2]), var_f(m.f[2])
    )
    natural_gradient!(
        m.f[1],
        ∇E_μ(likelihood(m), opt(inference(m)), yview(m), local_vars)[1],
        ∇E_Σ(likelihood(m), opt(inference(m)), yview(m), local_vars)[1],
        ρ(m),
        opt(inference(m)),
        first(Zviews(m)),
        state.kernel_matrices[1],
        state.opt_state[1],
    )
    vi_opt_state_1 = global_update!(
        m.f[1], opt(inference(m)), inference(m), state.opt_state[1]
    )
    opt_state = (vi_opt_state_1, vi_opt_state_2)
    return merge(state, (; opt_state))
end

function heteroscedastic_expectations!(
    local_vars,
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    μ::AbstractVector,
    Σ::AbstractVector,
)
    @. local_vars.σg = expectation(logistic, μ, Σ)
    l.link.λ .= 0.5 * length(local_vars.ϕ) / dot(local_vars.ϕ, local_vars.σg)
    return local_vars
end

function expectation(f::Function, μ::Real, σ²::Real)
    x = pred_nodes * sqrt(max(σ², zero(σ²))) .+ μ
    return dot(pred_weights, f.(x))
end

@inline function ∇E_μ(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::AOptimizer,
    y::AbstractVector,
    state,
)
    return (0.5 * y .* l.link.λ[1] .* state.σg, 0.5 * (0.5 .- state.γ))
end

@inline function ∇E_Σ(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::AOptimizer,
    ::AbstractVector,
    state,
)
    return (0.5 * l.link.λ[1] .* state.σg, 0.5 * state.θ)
end

function proba_y(
    model::AbstractGPModel{T,<:HeteroscedasticGaussianLikelihood}, X_test::AbstractMatrix
) where {T<:Real}
    (μf, σ²f), (μg, σ²g) = predict_f(model, X_test; cov=true)
    l = likelihood(model)
    return μf, σ²f + expectation.(l.link, μg, σ²g)
end

function expec_loglikelihood(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::AnalyticVI,
    y::AbstractVector,
    μ,
    diag_cov,
    state,
)
    tot = length(y) * (0.5 * log(l.link.λ[1]) - log(2 * sqrt(twoπ)))
    tot +=
        0.5 * (
            dot(μ[2], (0.5 .- state.γ)) - dot(abs2.(μ[2]), state.θ) -
            dot(diag_cov[2], state.θ)
        )
    tot -= PoissonKL(l, y, μ[1], diag_cov[1], state)
    return tot
end

function AugmentedKL(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic}, state, ::Any
)
    return PolyaGammaKL(l, state)
end

function PoissonKL(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
    state,
)
    return PoissonKL(
        state.γ,
        0.5 * l.link.λ[1] * (abs2.(y - μ) + Σ),
        log.(0.5 * l.link.λ[1] * (abs2.(μ - y) + Σ)),
    )
end

function PolyaGammaKL(::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic}, state)
    return PolyaGammaKL(0.5 .+ state.γ, state.c, state.θ)
end
