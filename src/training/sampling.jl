function StatsBase.sample(model::MCGP{T}, nSamples::Int=1000; callback::Union{Nothing,Function}=nothing,cat_samples::Bool=false) where {T}
    if verbose(model) > 0
      @info "Starting sampling $model with $(model.nSamples) samples with $(size(model.X,2)) features and $(nLatent(model)) latent GP" * (model.nLatent > 1 ? "s" : "")
    end
    nSamples > 0 || error("Number of samples should be positive")
    return sample_parameters(model, nSamples, callback, cat_samples)
end

StatsBase.sample(model::MCGP, nSamples::Int; kwargs...) = sample(Random.GLOBAL_RNG, model, nSamples; kwargs...)
StatsBase.sample(rng::Random.AbstractRNG, model::MCGP, nSamples::Int; kwargs...) = sample(rng, model, inference(model), nSamples; kwargs...)

function AbstractMCMC.step(rng, model, sampler; kwargs...)
    sample_local!(likelihood(m), yview(m), means(m))
    f = sample_global!.(∇E_μ(m), ∇E_Σ(m), Zviews(m), getf(m))
end

function AbstractMCMC.step(rng, model, sampler, state; kwargs...)
    sample_local!(likelihood(m), yview(m), means(m))
    f = sample_global!.(∇E_μ(m), ∇E_Σ(m), Zviews(m), getf(m))
end