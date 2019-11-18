include("gibbssampling.jl")
include("hmcsampling.jl")

function log_gp_prior(gp::_MCGP,f::AbstractVector)
    logpdf(MvNormal(gp.μ₀,gp.K),f)
end

function log_joint_model(model::MCGP{T,L,<:SamplingInference},x) where {T,L}
    fs = unsqueeze(model,x)
    log_likelihood(model.likelihood,get_y(model),fs) + sum(log_gp_prior.(model.f),fs)
end

function unsqueeze(model::MCGP,f)
    n = model.nFeatures
    Tuple(f[((i-1)*n+1):(i*n)] for i in 1:model.nLatent)
end

function grad_log_joint_model(model::MCGP{T,L,<:SamplingInference},x) where {T,L}
    fs = unsqueeze(model,x)
    vcat(grad_log_likelihood(model.likelihood,get_y(model),fs)...) + vcat(grad_log_gp_prior.(model.f,fs)...)
end

log_likelihood(l::Likelihood,y::AbstractVector,f::Tuple{<:AbstractVector{T}}) where {T<:Real} = logpdf.(l,y,first(f))

function grad_log_likelihood(l::Likelihood,y::AbstractVector,f::Tuple{<:AbstractVector{T}}) where {T}
    grad_log_pdf.(l,y,f)
end

function grad_log_gp_prior(gp,f)
    gp.K/(f-gp.μ₀)
end
