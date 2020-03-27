"""
```julia
GibbsSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=1)
```

Draw samples from the true posterior via Gibbs Sampling.

**Keywords arguments**
    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct GibbsSampling{T<:Real,N} <: SamplingInference{T}
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
    xview::Vector
    yview::Vector
    function GibbsSampling{T}(
        nBurnin::Int,
        samplefrequency::Int,
        ϵ::Real,
    ) where {T}
        @assert nBurnin >= 0 "nBurnin should be a positive integer"
        @assert samplefrequency >= 0 "samplefrequency should be a positive integer"
        return new{T,1}(nBurnin, samplefrequency, ϵ)
    end
    function GibbsSampling{T,1}(
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

function GibbsSampling(;
    ϵ::T = 1e-5,
    nBurnin::Int = 100,
    samplefrequency::Int = 1,
) where {T<:Real}
    GibbsSampling{T}(nBurnin, samplefrequency, ϵ)
end

function Base.show(io::IO,inference::GibbsSampling{T}) where {T<:Real}
    print(io,"Gibbs Sampler")
end

function tuple_inference(
    inf::TSamp,
    nLatent::Int,
    nFeatures::Vector{<:Integer},
    nSamples::Vector{<:Integer},
    nMinibatch::Vector{<:Integer}, #unused
) where {TSamp<:GibbsSampling}
    return TSamp(
        inf.nBurnin,
        inf.samplefrequency,
        inf.ϵ,
        nFeatures,
        nSamples,
        nSamples,
        nLatent,
    )
end

function init_sampler(
    inference::GibbsSampling{T},
    nLatent::Integer,
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
    init_sampler(
        m.inference,
        m.nLatent,
        m.nFeatures,
        nSamples,
        cat_samples,
    )
    computeMatrices!(m)
    for i in 1:(m.inference.nBurnin+nSamples*m.inference.samplefrequency)
        m.inference.nIter += 1
        sample_local!(m.likelihood, get_y(m), get_f(m))
        sample_global!.(∇E_μ(m), ∇E_Σ(m), get_Z(m), m.f)
        if nIter(m.inference) > m.inference.nBurnin &&
           (nIter(m.inference) - m.inference.nBurnin) %
           m.inference.samplefrequency == 0 # Store variables every samplefrequency
            store_variables!(m.inference, get_f(m))
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

logpdf(model::AbstractGP{T,<:Likelihood,<:GibbsSampling}) where {T} = zero(T)

function sample_global!(
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    X::AbstractMatrix,
    gp::_MCGP{T},
) where {T}
    Σ = inv(Symmetric(2.0 * Diagonal(∇E_Σ) + inv(gp.K).mat))
    gp.f .= rand(MvNormal(Σ * (∇E_μ + inv(gp.K) * gp.μ₀(X)), Σ))
    return nothing
end

function store_variables!(i::SamplingInference{T}, fs) where {T}
    i.sample_store[(nIter(i)-i.nBurnin)÷i.samplefrequency, :, :] .= hcat(fs...)
end

function post_process!(
    model::AbstractGP{T,<:Likelihood,<:GibbsSampling},
) where {T}
    for k in 1:model.nLatent
        model.μ[k] =
            vec(mean(hcat(model.inference.sample_store[k]...), dims = 2))
        model.Σ[k] =
            Symmetric(cov(hcat(model.inference.sample_store[k]...), dims = 2))
    end
    nothing
end
