"""
    HMCSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10)

Draw samples from the true posterior via Hamiltonian Monte Carlo.

**Keywords arguments**
    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct HMCSampling{T<:Real,N} <: SamplingInference{T}
    nBurnin::Integer # Number of burnin samples
    samplefrequency::Integer # Frequency at which samples are saved
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Vector{T} # Number of samples
    nMinibatch::Vector{Int64}
    ρ::Vector{T}
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    opt::NTuple{N,SOptimizer}
    sample_store::AbstractArray{T,3}
    xview::Vector#,Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    yview::Vector
    function HMCSampling{T}(
        nBurnin::Int,
        samplefrequency::Int,
        ϵ::Real,
    ) where {T}
        @assert nBurnin >= 0 "nBurnin should be a positive integer"
        @assert samplefrequency >= 0 "samplefrequency should be a positive integer"
        return new{T,1}(nBurnin, samplefrequency, ϵ)
    end
    function HMCSampling{T,1}(
        nBurnin::Int,
        samplefrequency::Int,
        ϵ::Real,
        nFeatures::Vector{<:Int},
        nSamples::Vector{<:Int},
        nMinibatch::Vector{<:Int},
        nLatent::Int,
    ) where {T}
        opts = ntuple(_ -> SOptimizer{T}(Descent(1.0)), nLatent)
        new{T,nLatent}(
            nBurnin,
            samplefrequency,
            ϵ,
            0,
            false,
            nSamples,
            nMinibatch,
            T.(nSamples ./ nMinibatch),
            true,
            opts,
        )
    end
end

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
    i::TSamp,
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Vector{<:Int},
    nMinibatch::Vector{<:Int}, #unused
) where {TSamp<:HMCSampling}
    return TSamp(
        i.nBurnin,
        i.samplefrequency,
        i.ϵ,
        nFeatures,
        nSamples,
        nSamples,
        nLatent,
    )
end

function grad_log_joint_pdf(model::MCGP{T,L,HMCSampling{T}}, x) where {T,L}
    grad_log_likelihood(model.likelihood, get_y(model), x) +
    sum(grad_log_gp_prior.(model.f, x))
end

function init_sampler!(
    inference::HMCSampling{T},
    nLatent::Integer,
    nFeatures::Integer,
    nSamples::Integer,
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
    f_init = vcat(get_f(model)...)
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
