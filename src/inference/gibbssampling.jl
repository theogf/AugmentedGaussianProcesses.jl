"""
    GibbsSampling(;ϵ::T=1e-5, nBurnin::Int=100, thinning::Int=1)

Draw samples from the true posterior via Gibbs Sampling.

## Keywords arguments
- `ϵ::T` : convergence criteria
- `nBurnin::Int` : Number of samples discarded before starting to save samples
- `thinning::Int` : Frequency at which samples are saved 
"""
mutable struct GibbsSampling{T<:Real} <: SamplingInference{T}
    nBurnin::Int # Number of burnin samples
    thinning::Int # Frequency at which samples are saved
    ϵ::T # Convergence criteria
    n_iter::Integer # Number of samples computed
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    opt::SOptimizer
    sample_store::Vector{Vector{Vector{T}}}
    function GibbsSampling{T}(nBurnin::Int, thinning::Int, ϵ::Real) where {T}
        nBurnin >= 0 || error("nBurnin should be positive")
        thinning >= 0 || error("thinning should be positive")
        return new{T}(
            nBurnin,
            thinning,
            ϵ,
            0,
            false,
            SOptimizer(Descent()),
            Vector{Vector{Vector{T}}}[],
        )
    end
end

ρ(::GibbsSampling{T}) where {T} = one(T)

function GibbsSampling(; ϵ::T=1e-5, nBurnin::Int=100, thinning::Int=1) where {T<:Real}
    return GibbsSampling{T}(nBurnin, thinning, ϵ)
end

function Base.show(io::IO, ::GibbsSampling{T}) where {T<:Real}
    return print(io, "Gibbs Sampler")
end

function sample_local!(
    local_vars, l::AbstractLikelihood, y, f::Tuple{<:AbstractVector{T}}
) where {T}
    return sample_local!(local_vars, l, y, only(f))
end

function sample_global!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    X::AbstractVector,
    gp::SampledLatent{T},
    kernel_mat,
) where {T}
    gp.post.Σ .= inv(Symmetric(2.0 * Diagonal(∇E_Σ) + inv(kernel_mat.K)))
    rand!(MvNormal(cov(gp) * (∇E_μ + kernel_mat.K \ pr_mean(gp, X)), cov(gp)), gp.post.f)
    return copy(posterior(gp).f)
end
