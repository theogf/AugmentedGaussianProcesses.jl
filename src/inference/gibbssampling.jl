"""
    GibbsSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=1)

Draw samples from the true posterior via Gibbs Sampling.

**Keywords arguments**
    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct GibbsSampling{T<:Real,N,Tx,Ty} <: SamplingInference{T}
    nBurnin::Int # Number of burnin samples
    samplefrequency::Int # Frequency at which samples are saved
    ϵ::T #Convergence criteria
    nIter::Integer #Number of samples computed
    nSamples::Int # Number of data samples
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    opt::NTuple{N,SOptimizer}
    xview::Tx
    yview::Ty
    sample_store::Array{T,3}
    function GibbsSampling{T}(
        nBurnin::Int,
        samplefrequency::Int,
        ϵ::Real,
    ) where {T}
        @assert nBurnin >= 0 "nBurnin should be a positive integer"
        @assert samplefrequency >= 0 "samplefrequency should be a positive integer"
        return new{T,1,Vector{T},Vector{T}}(nBurnin, samplefrequency, ϵ)
    end
    function GibbsSampling{T}(
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

isStochastic(::GibbsSampling) = false
getρ(::GibbsSampling{T}) where {T} = one(T)
nMinibatch(i::GibbsSampling) = i.nSamples

function GibbsSampling(;
    ϵ::T = 1e-5,
    nBurnin::Int = 100,
    samplefrequency::Int = 1,
) where {T<:Real}
    GibbsSampling{T}(nBurnin, samplefrequency, ϵ)
end

function Base.show(io::IO, inference::GibbsSampling{T}) where {T<:Real}
    print(io, "Gibbs Sampler")
end

function tuple_inference(
    inf::GibbsSampling{T},
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Int,
    nMinibatch::Int, #unused
    xview,
    yview
) where {T}
    return GibbsSampling{T}(
        inf.nBurnin,
        inf.samplefrequency,
        inf.ϵ,
        nFeatures,
        nSamples,
        nLatent,
        xview,
        yview,
    )
end

function init_sampler!(
    inference::GibbsSampling{T},
    nLatent::Int,
    nFeatures::Integer,
    nSamples::Integer,
    cat_samples::Bool,
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
    m::MCGP{T,L,<:GibbsSampling},
    nSamples::Int,
    callback::Union{Nothing,Function},
    cat_samples::Bool,
) where {T,L}
    init_sampler!(
        inference(m),
        nLatent(m),
        nFeatures(m),
        nSamples,
        cat_samples,
    )
    computeMatrices!(m)
    for i in 1:(m.inference.nBurnin+nSamples*m.inference.samplefrequency)
        m.inference.nIter += 1
        sample_local!(likelihood(m), yview(m), means(m))
        sample_global!.(∇E_μ(m), ∇E_Σ(m), Zviews(m), getf(m))
        if nIter(inference(m)) > m.inference.nBurnin &&
           (nIter(inference(m)) - m.inference.nBurnin) %
           m.inference.samplefrequency == 0 # Store variables every samplefrequency
            store_variables!(inference(m), means(m))
        end
    end
    symbols = ["f_" * string(i) for i = 1:nFeatures(m)]
    if nLatent(m) == 1
        return Chains(
            reshape(
                m.inference.sample_store[:, :, 1],
                :,
                nFeatures(m),
                1,
            ),
            symbols,
        )
    else
        return [
            Chains(
                reshape(
                    m.inference.sample_store[:, :, i],
                    :,
                    nFeatures(m),
                    1,
                ),
                symbols,
            ) for i = 1:nLatent(m)
        ]
    end
end

sample_local!(l::Likelihood, y, f::Tuple{<:AbstractVector{T}}) where {T} =
    sample_local!(l, y, first(f))
set_ω!(l::Likelihood, ω) = l.θ .= ω
get_ω(l::Likelihood) = l.θ

function sample_global!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    X::AbstractVector,
    gp::SampledLatent{T},
) where {T}
    gp.post.Σ .= inv(Symmetric(2.0 * Diagonal(∇E_Σ) + inv(pr_cov(gp))))
    rand!(MvNormal(cov(gp) * (∇E_μ + pr_cov(gp) \ pr_mean(gp, X)), cov(gp)), gp.post.f)
    return nothing
end

function post_process!(
    m::MCGP{T}
) where {T}
    # for k in 1:model.nLatent
    #     model.μ[k] =
    #         vec(mean(hcat(model.inference.sample_store[k]...), dims = 2))
    #     model.Σ[k] =
    #         Symmetric(cov(hcat(model.inference.sample_store[k]...), dims = 2))
    # end
    nothing
end
