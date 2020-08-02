"""
    HMCSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10)

Draw samples from the true posterior via Hamiltonian Monte Carlo.

**Keywords arguments**
    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct HMCSampling{T<:Real,N,Tx,Ty} <: SamplingInference{T}
    nBurnin::Int # Number of burnin samples
    samplefrequency::Integer # Frequency at which samples are saved
    ϵ::T #Convergence criteria
    nIter::Int #Number of samples computed
    nSamples::Int # Number of data samples
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    opt::NTuple{N,SOptimizer}
    xview::Tx
    yview::Ty
    sample_store::Array{T,3}
    function HMCSampling{T}(
        nBurnin::Int,
        samplefrequency::Int,
        ϵ::Real,
    ) where {T}
        nBurnin >= 0 || error("nBurnin should be a positive integer")
        samplefrequency >= 0 || error("samplefrequency should be a positive integer")
        return new{T,1,Vector{T},Vector{T}}(nBurnin, samplefrequency, ϵ)
    end
    function HMCSampling{T}(
        nBurnin::Int,
        samplefrequency::Int,
        ϵ::Real,
        nFeatures::Vector{<:Int},
        nSamples::Int,
        nLatent::Int,
        xview::Tx,
        yview::Ty
    ) where {T,Tx,Ty}
        opts = ntuple(_ -> SOptimizer{T}(nothing), nLatent)
        new{T,nLatent,Tx,Ty}(
            nBurnin,
            samplefrequency,
            ϵ,
            0,
            nSamples,
            true,
            opts,
            xview,
            yview,
        )
    end
end

isStochastic(::HMCSampling) = false
getρ(::HMCSampling{T}) where {T} = one(T)
nMinibatch(i::HMCSampling) = i.nSamples

function HMCSampling(;
    ϵ::T = 1e-5,
    nBurnin::Int = 100,
    samplefrequency::Int = 10,
) where {T<:Real}
    HMCSampling{Float64}(nBurnin, samplefrequency, ϵ)
end

function Base.show(io::IO, inference::HMCSampling{T}) where {T<:Real}
    print(io, "Hamilton Monte Carlo Sampler")
end

function tuple_inference(
    i::HMCSampling{T},
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Int,
    nMinibatch::Int, #unused
    xview,
    yview
) where {T}
    return HMCSampling{T}(
        i.nBurnin,
        i.samplefrequency,
        i.ϵ,
        nFeatures,
        nSamples,
        nLatent,
        xview,
        yview
    )
end

function grad_log_joint_pdf(model::MCGP{T,L,HMCSampling{T}}, x) where {T,L}
    grad_log_likelihood(likelihood(model), yview(model), x) +
    sum(grad_log_gp_prior.(model.f, x))
end

function init_sampler!(
    inference::HMCSampling{T},
    nLatent::Integer,
    nFeatures::Integer,
    nSamples::Integer,
    cat_samples::Bool
) where {T<:Real}
    if inference.nIter == 0 || !cat_samples
        inference.sample_store = zeros(T, nSamples, nFeatures, nLatent)
    else
        inference.sample_store = cat(
            inference.sample_store,
            zeros(T, nSamples, nFeatures, nLatent),
            dims = 1,
        )
    end
    return inference
end

function sample_parameters(
    model::MCGP{T,L,<:HMCSampling{T}},
    nSamples::Int,
    callback,
    cat_samples::Bool,
) where {T,L}
    f_init = vcat(means(model)...)
    logpdf(x) = log_joint_model(model, x)
    gradlogpdf(x) = (log_joint_model(model, x), grad_log_joint_model(model, x))
    metric = DiagEuclideanMetric(length(f_init))
    h = Hamiltonian(metric, logpdf, gradlogpdf)
    int = Leapfrog(find_good_eps(h, f_init))
    prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
    adaptor = StanHMCAdaptor(
        n_adapts,
        Preconditioner(metric),
        NesterovDualAveraging(0.8, int.ϵ),
    )
    samples, stats =
        sample(h, prop, f_init, nSamples, adaptor, n_adapts; progress = true)
end
