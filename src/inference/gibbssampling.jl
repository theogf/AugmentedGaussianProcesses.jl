"""
    GibbsSampling(;ϵ::T=1e-5, nBurnin::Int=100, thinning::Int=1)

Draw samples from the true posterior via Gibbs Sampling.

## Keywords arguments
- `ϵ::T` : convergence criteria
- `nBurnin::Int` : Number of samples discarded before starting to save samples
- `thinning::Int` : Frequency at which samples are saved 
"""
mutable struct GibbsSampling{T<:Real,N,Tx,Ty} <: SamplingInference{T}
    nBurnin::Int # Number of burnin samples
    thinning::Int # Frequency at which samples are saved
    ϵ::T # Convergence criteria
    nIter::Integer # Number of samples computed
    nSamples::Int # Number of data samples
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    opt::NTuple{N,SOptimizer}
    xview::Tx
    yview::Ty
    sample_store::Vector{Vector{Vector{T}}}
    function GibbsSampling{T}(
        nBurnin::Int,
        thinning::Int,
        ϵ::Real,
    ) where {T}
        nBurnin >= 0 || error("nBurnin should be positive")
        thinning >= 0 || error("thinning should be positive")
        return new{T,1,Vector{T},Vector{T}}(nBurnin, thinning, ϵ)
    end
    function GibbsSampling{T}(
        nBurnin::Int,
        thinning::Int,
        ϵ::Real,
        ::Vector{<:Int},
        nSamples::Int,
        nLatent::Int,
        xview::Tx,
        yview::Ty
    ) where {T,Tx,Ty}
        opts = ntuple(_ -> SOptimizer{T}(nothing), nLatent)
        new{T,nLatent,Tx,Ty}(
            nBurnin,
            thinning,
            ϵ,
            0,
            nSamples,
            true,
            opts,
            xview,
            yview,
            [],
        )
    end
end

isStochastic(::GibbsSampling) = false
getρ(::GibbsSampling{T}) where {T} = one(T)
nMinibatch(i::GibbsSampling) = i.nSamples

function GibbsSampling(;
    ϵ::T = 1e-5,
    nBurnin::Int = 100,
    thinning::Int = 1,
) where {T<:Real}
    GibbsSampling{T}(nBurnin, thinning, ϵ)
end

function Base.show(io::IO, ::GibbsSampling{T}) where {T<:Real}
    print(io, "Gibbs Sampler")
end

function tuple_inference(
    inf::GibbsSampling{T},
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Int,
    ::Int,
    xview,
    yview
) where {T}
    return GibbsSampling{T}(
        inf.nBurnin,
        inf.thinning,
        inf.ϵ,
        nFeatures,
        nSamples,
        nLatent,
        xview,
        yview,
    )
end

sample_local!(l::AbstractLikelihood, y, f::Tuple{<:AbstractVector{T}}) where {T} =
    sample_local!(l, y, first(f))
set_ω!(l::AbstractLikelihood, ω) = l.θ .= ω
get_ω(l::AbstractLikelihood) = l.θ

function sample_global!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    X::AbstractVector,
    gp::SampledLatent{T},
) where {T}
    gp.post.Σ .= inv(Symmetric(2.0 * Diagonal(∇E_Σ) + inv(pr_cov(gp))))
    rand!(MvNormal(cov(gp) * (∇E_μ + pr_cov(gp) \ pr_mean(gp, X)), cov(gp)), gp.post.f)
    return deepcopy(posterior(gp).f)
end