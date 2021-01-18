include("gibbssampling.jl")
include("hmcsampling.jl")

isStochastic(::SamplingInference) = false

function log_gp_prior(gp::SampledLatent, f::AbstractVector, X::AbstractVector)
    logpdf(MvNormal(pr_mean(gp, X), pr_cov(gp)), f)
end

function logjoint(model::MCGP{T,L,<:SamplingInference}, x) where {T,L}
    fs = unsqueeze(model, x)
    loglikelihood(model.likelihood, yview(model), fs) +
    sum(log_gp_prior.(model.f, fs))
end

function unsqueeze(model::MCGP, f)
    n = model.nFeatures
    Tuple(f[((i-1)*n+1):(i*n)] for i = 1:model.nLatent)
end

function ∇logjoint(
    model::MCGP{T,L,<:SamplingInference},
    x,
) where {T,L}
    fs = unsqueeze(model, x)
    vcat(∇loglikehood(likelihood(model), yview(model), fs)...) +
    vcat(∇logprior(model.f, fs)...)
end

Distributions.loglikelihood(
    l::Likelihood,
    y::AbstractVector,
    f::Tuple{<:AbstractVector{T}},
) where {T<:Real} = sum(logpdf.(l, y, f...))

function ∇loglikehood(
    l::Likelihood,
    y::AbstractVector,
    f::Tuple{<:AbstractVector{T}},
) where {T}
    ∇loglikehood.(l, y, f...)
end


function ∇logprior(gp::AbstractLatent, f)
    -pr_cov(gp) / f # Remove μ₀ temp
end

function logprior(gp::AbstractLatent, f)
    -0.5 * logdet(pr_cov(gp)) - 0.5 * invquad(pr_cov(gp), f) # Remove μ₀ temp
end

function store_variables!(i::SamplingInference{T}, fs) where {T}
    i.sample_store[(nIter(i)-i.nBurnin)÷i.samplefrequency, :, :] .=
        hcat(fs...)
end
