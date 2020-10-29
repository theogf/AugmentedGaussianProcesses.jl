include("gibbssampling.jl")
include("hmcsampling.jl")

isStochastic(::SamplingInference) = false

function log_gp_prior(gp::SampledLatent, f::AbstractVector, X::AbstractVector)
    Distributions.logpdf(MvNormal(pr_mean(gp, X), pr_cov(gp)), f)
end

function log_joint_model(model::MCGP{T,L,<:SamplingInference}, x) where {T,L}
    fs = unsqueeze(model, x)
    log_likelihood(model.likelihood, yview(model), fs) +
    sum(log_gp_prior.(model.f, fs))
end

function unsqueeze(model::MCGP, f)
    n = model.nFeatures
    Tuple(f[((i-1)*n+1):(i*n)] for i = 1:model.nLatent)
end

function grad_log_joint_model(
    model::MCGP{T,L,<:SamplingInference},
    x,
) where {T,L}
    fs = unsqueeze(model, x)
    vcat(grad_loglike(likelihood(model), yview(model), fs)...) +
    vcat(grad_logprior(model.f, fs)...)
end

Distributions.loglikelihood(
    l::Likelihood,
    y::AbstractVector,
    f::Tuple{<:AbstractVector{T}},
) where {T<:Real} = sum(logpdf.(l, y, f...))

function grad_loglike(
    l::Likelihood,
    y::AbstractVector,
    f::Tuple{<:AbstractVector{T}},
) where {T}
    grad_loglike.(l, y, f...)
end


function grad_logprior(gp::AbstractLatent, f)
    -pr_cov(gp) / f # Remove μ₀ temp
end

function logprior(gp::AbstractLatent, f)
    -0.5 * logdet(pr_cov(gp)) - 0.5 * invquad(pr_cov(gp), f) # Remove μ₀ temp
end

function store_variables!(i::SamplingInference{T}, fs) where {T}
    i.sample_store[(nIter(i)-i.nBurnin)÷i.samplefrequency, :, :] .=
        hcat(fs...)
end
