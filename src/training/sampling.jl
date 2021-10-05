struct GPModel{TGP,Tx,Ty} <: AbstractMCMC.AbstractModel
    gp::TGP
    x::Tx
    y::Ty
end

struct GPSampler{TS} <: AbstractMCMC.AbstractSampler
    sampler::TS
end

function StatsBase.sample(model::MCGP, N::Int; kwargs...)
    return sample(Random.GLOBAL_RNG, model, N; kwargs...)
end
function StatsBase.sample(rng::Random.AbstractRNG, model::MCGP, N::Int; kwargs...)
    sampler = inference(model)
    thinning = get(kwargs, :thinning, sampler.thinning)
    discard_initial = get(kwargs, :discard_initial, sampler.nBurnin)
    progressname = get(kwargs, :progressname, "Sampling with $sampler")
    return sample(
        rng,
        GPModel(model, input(model.data), output(model.data)),
        GPSampler(sampler),
        N;
        thinning,
        discard_initial,
        progressname,
        kwargs...,
    )
end
function AbstractMCMC.step(
    rng::AbstractRNG, model::GPModel, sampler::GPSampler{<:GibbsSampling}; kwargs...
)
    state = compute_kernel_matrices(model.gp, (;), model.x, true)
    local_vars = init_local_vars(likelihood(model.gp), length(model.x))
    local_vars = sample_local!(local_vars, likelihood(model.gp), model.y, means(model.gp))
    state = merge(state, (; local_vars))
    f =
        sample_global!.(
            ∇E_μ(model.gp, model.y, state.local_vars),
            ∇E_Σ(model.gp, model.y, state.local_vars),
            Zviews(model.gp),
            getf(model.gp),
            state.kernel_matrices,
        )
    sampler.sampler.n_iter += 1
    return f, state
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::GPModel, sampler::GPSampler{<:GibbsSampling}, state; kwargs...
)
    local_vars = sample_local!(
        state.local_vars, likelihood(model.gp), model.y, means(model.gp)
    )
    state = merge(state, (; local_vars))
    f =
        sample_global!.(
            ∇E_μ(model.gp, model.y, state.local_vars),
            ∇E_Σ(model.gp, model.y, state.local_vars),
            Zviews(model.gp),
            getf(model.gp),
            state.kernel_matrices,
        )
    sampler.sampler.n_iter += 1
    return f, state
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
