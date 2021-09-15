"""
    HMCSampling(;ϵ::T=1e-5,nBurnin::Int=100,thinning::Int=10)

Draw samples from the true posterior via Hamiltonian Monte Carlo.

## Keywords arguments
- `ϵ::T` : convergence criteria
- `nBurnin::Int` : Number of samples discarded before starting to save samples
- `thinning::Int` : Frequency of sampling
"""
mutable struct HMCSampling{T<:Real,N,Tx,Ty} <: SamplingInference{T}
    nBurnin::Int # Number of burnin samples
    thinning::Integer # Frequency at which samples are saved
    ϵ::T #Convergence criteria
    n_iter::Int #Number of samples computed
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    opt::NTuple{N,SOptimizer}
    xview::Tx
    yview::Ty
    sample_store::Array{T,3}
    function HMCSampling{T}(nBurnin::Int, thinning::Int, ϵ::Real) where {T}
        nBurnin >= 0 || error("nBurnin should be positive")
        thinning >= 0 || error("thinning should be positive")
        return new{T,1,Vector{T},Vector{T}}(nBurnin, thinning, ϵ)
    end
    function HMCSampling{T}(
        nBurnin::Int,
        thinning::Int,
        ϵ::Real,
        nFeatures::Vector{<:Int},
        nSamples::Int,
        nLatent::Int,
        xview::Tx,
        yview::Ty,
    ) where {T,Tx,Ty}
        opts = ntuple(_ -> SOptimizer{T}(nothing), nLatent)
        return new{T,nLatent,Tx,Ty}(
            nBurnin, thinning, ϵ, 0, nSamples, true, opts, xview, yview
        )
    end
end

ρ(::HMCSampling{T}) where {T} = one(T)
nMinibatch(i::HMCSampling) = i.nSamples

function HMCSampling(; ϵ::T=1e-5, nBurnin::Int=100, thinning::Int=10) where {T<:Real}
    return HMCSampling{Float64}(nBurnin, thinning, ϵ)
end

function Base.show(io::IO, inference::HMCSampling{T}) where {T<:Real}
    return print(io, "Hamilton Monte Carlo Sampler")
end

function tuple_inference(
    i::HMCSampling{T},
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Int,
    nMinibatch::Int, #unused
    xview,
    yview,
) where {T}
    return HMCSampling{T}(
        i.nBurnin, i.thinning, i.ϵ, nFeatures, nSamples, nLatent, xview, yview
    )
end

function grad_log_joint_pdf(model::MCGP{T,L,HMCSampling{T}}, x) where {T,L}
    return grad_log_likelihood(likelihood(model), yview(model), x) +
           sum(grad_log_gp_prior.(model.f, x))
end

function init_sampler!(
    inference::HMCSampling{T},
    nLatent::Integer,
    nFeatures::Integer,
    nSamples::Integer,
    cat_samples::Bool,
) where {T<:Real}
    if n_iter(inference) == 0 || !cat_samples
        inference.sample_store = zeros(T, nSamples, nFeatures, nLatent)
    else
        inference.sample_store = cat(
            inference.sample_store, zeros(T, nSamples, nFeatures, nLatent); dims=1
        )
    end
    return inference
end

function sample_parameters(
    model::MCGP{T,L,<:HMCSampling{T}}, nSamples::Int, callback, cat_samples::Bool
) where {T,L}
    f_init = vcat(means(model)...)
    logpdf(x) = log_joint_model(model, x)
    gradlogpdf(x) = (log_joint_model(model, x), grad_log_joint_model(model, x))
    metric = DiagEuclideanMetric(length(f_init))
    h = Hamiltonian(metric, logpdf, gradlogpdf)
    int = Leapfrog(find_good_eps(h, f_init))
    prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
    adaptor = StanHMCAdaptor(
        n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int.ϵ)
    )
    return samples, stats = sample(
        h, prop, f_init, nSamples, adaptor, n_adapts; progress=true
    )
end
