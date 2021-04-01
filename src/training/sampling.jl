struct GPModel{TGP} <: AbstractMCMC.AbstractModel
    gp::TGP
end

struct GPSampler{TS} <: AbstractMCMC.AbstractSampler
    sampler::TS
end

function StatsBase.sample(model::MCGP, nSamples::Int; kwargs...)
    return sample(Random.GLOBAL_RNG, model, nSamples; kwargs...)
end
function StatsBase.sample(rng::Random.AbstractRNG, model::MCGP, nSamples::Int; kwargs...)
    sampler = inference(model)
    thinnning = get(kwargs, :thinning, sampler.thinning)
    discard_initial = get(kwargs, :discard_initial, sampler.nBurnin)
    progressname = get(kwargs, :progressname, "Sampling with $sampler")
    return sample(
        rng,
        GPModel(model),
        GPSampler(sampler),
        nSamples;
        thinning,
        discard_inital,
        progressname,
        kwargs...,
    )
end
function AbstractMCMC.step(
    rng, model::GPModel, sampler::GPSampler{<:GibbsSampling}; kwargs...
)
    computeMatrices!(model.gp)
    sample_local!(likelihood(model.gp), yview(model.gp), means(model.gp))
    f = sample_global!.(∇E_μ(model.gp), ∇E_Σ(model.gp), Zviews(model.gp), getf(model.gp))
    sampler.sampler.nIter += 1
    return f, nothing
end

function AbstractMCMC.step(
    rng, model::GPModel, sampler::GPSampler{<:GibbsSampling}, state; kwargs...
)
    sample_local!(likelihood(model.gp), yview(model.gp), means(model.gp))
    f = sample_global!.(∇E_μ(model.gp), ∇E_Σ(model.gp), Zviews(model.gp), getf(model.gp))
    sampler.sampler.nIter += 1
    return f, nothing
end

function AbstractMCMC.bundle_samples(
    samples, model::GPModel, sampler::GPSampler, state, chain_type::Type; kwargs...
)
    resume = get(kwargs, :cat, true)
    if !resume || isempty(sampler.sampler.sample_store) # check if either one should restart sampling or if no samples was ever taken
        sampler.sampler.sample_store = samples
    else
        sampler.sampler.sample_store = vcat(sampler.sampler.sample_store, samples)
    end
    return samples
end
