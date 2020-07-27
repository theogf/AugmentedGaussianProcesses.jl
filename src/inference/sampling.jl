include("gibbssampling.jl")
include("hmcsampling.jl")

function log_gp_prior(gp::SampledLatent, f::AbstractVector, X::AbstractVector)
    Distributions.logpdf(MvNormal(pr_mean(gp, X), pr_cov(gp)), f)
end

function log_joint_model(model::MCGP{T,L,<:SamplingInference}, x) where {T,L}
    fs = unsqueeze(model, x)
    log_likelihood(model.likelihood, get_y(model), fs) +
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
    vcat(grad_log_likelihood(model.likelihood, get_y(model), fs)...) +
    vcat(grad_log_gp_prior.(model.f, fs)...)
end

log_likelihood(
    l::Likelihood,
    y::AbstractVector,
    f::Tuple{<:AbstractVector{T}},
) where {T<:Real} = sum(logpdf.(l, y, first(f)))

function grad_log_likelihood(
    l::Likelihood,
    y::AbstractVector,
    f::Tuple{<:AbstractVector{T}},
) where {T}
    grad_logpdf.(l, y, f)
end

function grad_log_gp_prior(gp, f)
    -gp.K / (f)#Remove μ₀ temp
end

function log_gp_prior(gp, f)
    -0.5 * logdet(gp.K) - 0.5 * invquad(gp.K, f)#Remove μ₀ temp
end

function store_variables!(i::SamplingInference{T}, fs) where {T}
    i.sample_store[(nIter(i)-i.nBurnin)÷i.samplefrequency, :, :] .=
        hcat(fs...)
end
