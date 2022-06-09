include("gibbssampling.jl")
include("hmcsampling.jl")

is_stochastic(::SamplingInference) = false

function log_gp_prior(gp::SampledLatent, f::AbstractVector, X::AbstractVector)
    return logpdf(MvNormal(pr_mean(gp, X), pr_cov(gp)), f)
end

function logjoint(model::MCGP{T,L,<:SamplingInference}, x) where {T,L}
    fs = unsqueeze(model, x)
    return loglikelihood(model.likelihood, yview(model), fs) +
           sum(log_gp_prior.(model.f, fs))
end

function unsqueeze(model::MCGP, f)
    n = model.nFeatures
    return Tuple(f[((i - 1) * n + 1):(i * n)] for i in 1:(model.nLatent))
end

function ∇logjoint(model::MCGP{T,L,<:SamplingInference}, x) where {T,L}
    fs = unsqueeze(model, x)
    return vcat(∇loglikehood(likelihood(model), yview(model), fs)...) +
           vcat(∇logprior(model.f, fs)...)
end

function Distributions.loglikelihood(
    l::AbstractLikelihood, y::AbstractVector, f::Tuple{<:AbstractVector{T}}
) where {T<:Real}
    return sum(logpdf.(l, y, f...))
end

function ∇loglikehood(
    l::AbstractLikelihood, y::AbstractVector, f::Tuple{<:AbstractVector{T}}
) where {T}
    return ∇loglikehood.(l, y, f...)
end

function ∇logprior(gp::AbstractLatent, f)
    return -pr_cov(gp) / f # Remove μ₀ temp
end

function logprior(gp::AbstractLatent, f)
    return -logdet(pr_cov(gp)) / 2 - invquad(pr_cov(gp), f) / 2 # Remove μ₀ temp
end

function store_variables!(i::SamplingInference{T}, fs) where {T}
    return i.sample_store[(n_iter(i) - i.nBurnin) ÷ i.thinning, :, :] .= hcat(fs...)
end
