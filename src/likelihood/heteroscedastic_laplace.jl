"""
```julia
HeteroscedasticLaplaceLikelihood(λ::T=1.0)
```

Gaussian with heteroscedastic noise given by another gp:
```math
    p(y|f,g,β) = N(y|f,(sqrt(2β σ(g)))⁻¹)
```
Where `σ` is the logistic function

Augmentation will be described in a future paper
"""
mutable struct HeteroscedasticLaplaceLikelihood{T<:Real} <:
               RegressionLikelihood{T}
    β::T # Maximum of precision
    c::Vector{T} # Second variational parameter of ω
    ϕ::Vector{T} # Variational parameter of n
    a::Vector{T} # First var parameter of γ
    b::T # Second var parameter of γ
    p::Vector{T} # Second var parameter of γ
    ψ::Vector{T} # Expectation of γ
    ν::Vector{T} # Expectation of γ⁻¹
    θ::Vector{T} # Expectation of ω
    σg::Vector{T} # Expectation of σ(g)
    function HeteroscedasticLaplaceLikelihood{T}(β::T) where {T<:Real}
        new{T}(β)
    end
    function HeteroscedasticLaplaceLikelihood{T}(
        β::T,
        c::AbstractVector{T},
        ϕ::AbstractVector{T},
        a::AbstractVector{T},
        b::T,
        p::AbstractVector{T},
        ψ::AbstractVector{T},
        ν::AbstractVector{T},
        θ::AbstractVector{T},
        σg::AbstractVector{T},
    ) where {T<:Real}
        new{T}(β, c, ϕ, a, b, p, ψ, ν, θ, σg)
    end
end

function HeteroscedasticLaplaceLikelihood(β::T = 1.0) where {T<:Real}
    @assert β > 0
    HeteroscedasticLaplaceLikelihood{T}(β)
end

implemented(
    ::HeteroscedasticLaplaceLikelihood,
    ::Union{<:GibbsSampling,<:AnalyticVI},
) = true


function pdf(l::HeteroscedasticLaplaceLikelihood, y::Real, f::AbstractVector)
    pdf(Laplace(y, inv(sqrt(2 * l.β * logistic(f[2])))), f[1])
end

function logpdf(l::HeteroscedasticLaplaceLikelihood, y::Real, f::AbstractVector)
    logpdf(Normal(y, inv(sqrt(2 * l.β * logistic(f[2])))), f[1])
end

function Base.show(io::IO, model::HeteroscedasticLaplaceLikelihood{T}) where {T}
    print(io, "Laplace likelihood with heteroscedastic noise")
end

num_latent(::HeteroscedasticLaplaceLikelihood) = 2

function treat_labels!(
    y::AbstractVector{T},
    likelihood::L,
) where {T,L<:HeteroscedasticLaplaceLikelihood}
    @assert T <: Real "For regression target(s) should be real valued"
    return y, 2, likelihood
end

function init_likelihood(
    likelihood::HeteroscedasticLaplaceLikelihood{T},
    inference::Inference{T},
    nLatent::Integer,
    nMinibatch::Integer,
    nFeatures::Integer,
) where {T<:Real}
    β = likelihood.β
    c = ones(T, nMinibatch)
    ϕ = ones(T, nMinibatch)
    a = ones(T, nMinibatch)
    b = 2 * likelihood.β
    p = ones(T, nMinibatch)
    ψ = ones(T, nMinibatch)
    ν = ones(T, nMinibatch)
    θ = ones(T, nMinibatch)
    σg = ones(T, nMinibatch)
    HeteroscedasticLaplaceLikelihood{T}(β, c, ϕ, a, b, p, ψ, ν, θ, σg)
end

function compute_proba(
    l::HeteroscedasticLaplaceLikelihood{T},
    μ::AbstractVector{<:AbstractVector},
    Σ::AbstractVector{<:AbstractVector},
) where {T}
    return μ[1], max.(Σ[1], zero(T)) .+ inv.(l.β * logistic.(μ[2]))
end

function expec_GIG(a::Real, b::Real, p::Real)
    sab = sqrt(a * b)
    sqrt(b / a) * besselk(p + 1, sab) / besselk(p, sab)
end

function expec_invGIG(a::Real, b::Real, p::Real)
    sab = sqrt(a * b)
    sqrt(a / b) * besselk(p + 1, sab) / besselk(p, sab) - 2p / b
end

function local_updates!(
    l::HeteroscedasticLaplaceLikelihood{T},
    y::AbstractVector,
    μ::NTuple{2,<:AbstractVector},
    diag_cov::NTuple{2,<:AbstractVector},
) where {T}
    # gp[1] is f and gp[2] is g (for approximating the noise)
    l.σg .= expectation.(logistic, μ[2], diag_cov[2])
    l.a .= abs2.(μ[1] - y) + diag_cov[1]
    l.b = 2 * l.β

    for i = 1:100
        l.p .= -l.ϕ .- 0.5
        l.ν .= expec_invGIG.(l.a, l.b, l.p)
        l.ϕ .= 0.5 * l.β * l.σg .* l.ν
    end
    l.ψ .= expec_GIG.(l.a, l.b, l.p)
    l.c .= sqrt.(abs2.(μ[2]) + diag_cov[2])
    l.θ .= 0.5 * (1.0 .+ l.ϕ) ./ l.c .* tanh.(0.5 * l.c)
    l.β = abs2( length(y) / dot(abs.(μ[1]-y), sqrt.(2l.σg)))
end


function sample_local!(
    l::HeteroscedasticLaplaceLikelihood{T},
    y::AbstractVector,
    f::NTuple{2,<:AbstractVector},
) where {T}
    l.σg .= logistic.(f[2])
    l.β = abs2( length(y) / dot(abs.(f[1]-y), sqrt.(2l.σg)))
    l.ψ .= rand.(GeneralizedInverseGaussian.(abs2.(y-f[1]), 2l.β, -l.ϕ .- 0.5))
    l.ϕ .= rand.(Poisson.(l.β*l.σg./l.ψ))
    l.θ .= rand.(PolyaGamma.(1.0 .+ l.ϕ, abs.(f[2])))
end


function heteroscedastic_expectations!(
    l::HeteroscedasticLaplaceLikelihood{T},
    μ::AbstractVector,
    Σ::AbstractVector,
) where {T}
    l.σg .= expectation.(logistic, μ, Σ)
    # l.β = 0.5*length(l.ϕ)/dot(l.ϕ,l.σg)
end

@inline ∇E_μ(
    l::HeteroscedasticLaplaceLikelihood,
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (l.ψ .* y, 0.5 * (1.0 .- l.ϕ))

@inline ∇E_Σ(
    l::HeteroscedasticLaplaceLikelihood,
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (0.5 * l.ψ, 0.5 * l.θ)

function proba_y(
    model::AbstractGP{T,HeteroscedasticLaplaceLikelihood{T},AnalyticVI{T}},
    X_test::AbstractMatrix{T},
) where {T<:Real}
    (μf, σ²f), (μg, σ²g) = predict_f(model, X_test, covf = true)
    return μf,
    σ²f + expectation.(x -> inv(model.likelihood.β * logistic(x)), μg, σ²g)
end

function expec_log_likelihood(
    l::HeteroscedasticLaplaceLikelihood{T},
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

AugmentedKL(l::HeteroscedasticLaplaceLikelihood, ::AbstractVector) =
    PolyaGammaKL(l)

function PoissonKL(
    l::HeteroscedasticLaplaceLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
) where {T}
    return PoissonKL(
        l.γ,
        0.5 * l.β * (abs2.(y - μ) + Σ),
        log.(0.5 * l.λ * (abs2.(μ - y) + Σ)),
    )
end

function PolyaGammaKL(l::HeteroscedasticLaplaceLikelihood{T}) where {T}
    PolyaGammaKL(0.5 .+ l.γ, l.c, l.θ)
end
