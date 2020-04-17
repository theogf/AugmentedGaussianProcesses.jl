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
mutable struct HeteroscedasticLikelihood{T<:Real} <: RegressionLikelihood{T}
    λ::T
    c::Vector{T}
    ϕ::Vector{T}
    γ::Vector{T}
    θ::Vector{T}
    σg::Vector{T}
    function HeteroscedasticLikelihood{T}(λ::T) where {T<:Real}
            new{T}(λ)
    end
    function HeteroscedasticLikelihood{T}(λ::T,c::AbstractVector{T},ϕ::AbstractVector{T},γ::AbstractVector{T},θ::AbstractVector{T},σg::AbstractVector{T}) where {T<:Real}
         new{T}(λ,c,ϕ,γ,θ,σg)
    end
end

function HeteroscedasticLikelihood(λ::T=1.0) where {T<:Real}
        HeteroscedasticLikelihood{T}(λ)
end

implemented(::HeteroscedasticLikelihood,::Union{<:AnalyticVI,<:GibbsSampling}) = true


function pdf(l::HeteroscedasticLikelihood,y::Real,f::AbstractVector)
    pdf(Normal(y,inv(sqrt(l.λ*logistic(f[2])))),f[1])
end

function logpdf(l::HeteroscedasticLikelihood,y::Real,f::AbstractVector)
    logpdf(Normal(y,inv(sqrt(l.λ*logistic(f[2])))),f[1])
end

function Base.show(io::IO,model::HeteroscedasticLikelihood{T}) where T
    print(io,"Gaussian likelihood with heteroscedastic noise")
end

num_latent(::HeteroscedasticLikelihood) = 2

function treat_labels!(y::AbstractVector{T},likelihood::L) where {T,L<:HeteroscedasticLikelihood}
    @assert T<:Real "For regression target(s) should be real valued"
    return y,2,likelihood
end

function init_likelihood(likelihood::HeteroscedasticLikelihood{T},inference::Inference{T},nLatent::Integer,nMinibatch::Integer,nFeatures::Integer) where {T<:Real}
    λ = likelihood.λ
    c = ones(T,nMinibatch)
    ϕ = ones(T,nMinibatch)
    γ = ones(T,nMinibatch)
    θ = ones(T,nMinibatch)
    σg = ones(T,nMinibatch)
    HeteroscedasticLikelihood{T}(λ,c,ϕ,γ,θ,σg)
end

function compute_proba(l::HeteroscedasticLikelihood{T},μ::AbstractVector{<:AbstractVector},Σ::AbstractVector{<:AbstractVector}) where {T}
    return μ[1],max.(Σ[1],zero(T)).+inv.(l.λ*logistic.(μ[2]))
end

function local_updates!(
    l::HeteroscedasticLikelihood{T},
    y::AbstractVector,
    μ::NTuple{2,<:AbstractVector},
    diag_cov::NTuple{2,<:AbstractVector},
) where {T}
    # gp[1] is f (mean) and gp[2] is g (variance)
    l.ϕ .= 0.5 * (abs2.(μ[1] - y) + diag_cov[1])
    l.c .= sqrt.(abs2.(μ[2]) + diag_cov[2])
    l.γ .= 0.5 * l.λ * l.ϕ .* safe_expcosh.(-0.5 * μ[2], 0.5 * l.c)
    l.θ .= 0.5 * (0.5 .+ l.γ) ./ l.c .* tanh.(0.5 * l.c)
    l.σg .= expectation.(logistic, μ[2], diag_cov[2])
    l.λ = 0.5 * length(l.ϕ) / dot(l.ϕ, l.σg)
end

function sample_local!(
    l::HeteroscedasticLikelihood{T},
    y::AbstractVector,
    f) where {T}
    l.σg = logistic.(f[2])
    l.γ .= rand.(Poisson.(l.λ * l.σg .* abs2.(y .- f[1]))) # Sample of poisson
    l.θ .= rand.(PolyaGamma.(0.5 .+ l.γ, abs.(f[2])))
    return nothing
end

function variational_updates!(
    model::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI},
) where {T,L}
    local_updates!(
        model.likelihood,
        get_y(model),
        mean_f(model),
        diag_cov_f(model),
    )
    natural_gradient!(
        ∇E_μ(model.likelihood, opt_type(model.inference), get_y(model))[2],
        ∇E_Σ(model.likelihood, opt_type(model.inference), get_y(model))[2],
        getρ(model.inference),
        opt_type(model.inference),
        get_Z(model, 2),
        model.f[2],
    )
    global_update!(model.f[2], opt_type(model.inference), model.inference)
    heteroscedastic_expectations!(
        model.likelihood,
        mean_f(model.f[2]),
        diag_cov_f(model.f[2]),
    )
    natural_gradient!(
        ∇E_μ(model.likelihood, opt_type(model.inference), get_y(model))[1],
        ∇E_Σ(model.likelihood, opt_type(model.inference), get_y(model))[1],
        getρ(model.inference),
        opt_type(model.inference),
        get_Z(model, 1),
        model.f[1],
    )
    global_update!(model.f[1], opt_type(model.inference), model.inference)
end

function heteroscedastic_expectations!(l::HeteroscedasticLikelihood{T},μ::AbstractVector,Σ::AbstractVector) where {T}
    l.σg .= expectation.(logistic,μ,Σ)
    l.λ = 0.5*length(l.ϕ)/dot(l.ϕ,l.σg)
end

@inline ∇E_μ(
    l::HeteroscedasticLikelihood,
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (0.5 * y .* l.λ .* l.σg, 0.5 * (0.5 .- l.γ))

@inline ∇E_Σ(
    l::HeteroscedasticLikelihood,
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (0.5 * l.λ .* l.σg, 0.5 * l.θ)

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

AugmentedKL(l::HeteroscedasticLikelihood,::AbstractVector) = PolyaGammaKL(l)

function PoissonKL(l::HeteroscedasticLikelihood{T},y::AbstractVector,μ::AbstractVector,Σ::AbstractVector) where {T}
    return PoissonKL(l.γ,0.5*l.λ*(abs2.(y-μ)+Σ),log.(0.5*l.λ*(abs2.(μ-y)+Σ)))
end

function PolyaGammaKL(l::HeteroscedasticLikelihood{T}) where {T}
    PolyaGammaKL(0.5.+l.γ,l.c,l.θ)
end
