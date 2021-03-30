"""
    QuadratureVI(;ϵ::T=1e-5, nGaussHermite::Integer=20, clipping=Inf, natural::Bool=true, optimiser=Momentum(0.0001))

Variational Inference solver by approximating gradients via numerical integration via Quadrature

## Keyword arguments
- `ϵ::T` : convergence criteria
- `nGaussHermite::Int` : Number of points for the integral estimation
- `clipping::Real` : Limit the gradients values to avoid overshooting
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.0001)`
"""
mutable struct QuadratureVI{T,N,Tx,Ty} <: NumericalVI{T}
    nPoints::Int # Number of points for the quadrature
    nodes::Vector{T} # Nodes locations
    weights::Vector{T} # Weights for each node
    clipping::T # Clipping value of the gradient
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    stoch::Bool #Use of mini-batches
    nSamples::Int #Number of samples of the data
    nMinibatch::Int #Size of mini-batches
    ρ::T #Stochastic Coefficient
    NaturalGradient::Bool
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    vi_opt::NTuple{N,NVIOptimizer}
    MBIndices::Vector{Int} #Indices of the minibatch
    xview::Tx
    yview::Ty

    function QuadratureVI{T}(
    ϵ::T,
    nPoints::Integer,
    optimiser,
    Stochastic::Bool,
    clipping::Real,
    nMinibatch::Int,
    natural::Bool,
    ) where {T}
        return new{T,1,Vector{T},Vector{T}}(
            nPoints,
            [],
            [],
            clipping,
            ϵ,
            0,
            Stochastic,
            0,
            nMinibatch,
            one(T),
            natural,
            true,
            (NVIOptimizer{T}(0, 0, optimiser),),
        )
    end

    function QuadratureVI{T}(
        ϵ::T,
        Stochastic::Bool,
        nPoints::Int,
        clipping::Real,
        nFeatures::Vector{<:Int},
        nSamples::Int,
        nMinibatch::Int,
        nLatent::Int,
        optimiser,
        natural::Bool,
        xview::Tx,
        yview::Ty
    ) where {T,Tx,Ty}
        gh = gausshermite(nPoints)
        vi_opts =
            ntuple(i -> NVIOptimizer{T}(nFeatures[i], nMinibatch[i], optimiser), nLatent)
        new{T,nLatent,Tx,Ty}(
            nPoints,
            gh[1] .* sqrt2,
            gh[2] ./ sqrtπ,
            clipping,
            ϵ,
            0,
            Stochastic,
            nSamples,
            nMinibatch,
            T(nSamples / nMinibatch),
            natural,
            true,
            vi_opts,
            1:nMinibatch,
            xview,
            yview,
        )
    end
end

function QuadratureVI(;
    ϵ::T = 1e-5,
    nGaussHermite::Integer = 100,
    optimiser = Momentum(1e-5),
    clipping::Real = 0.0,
    natural::Bool = true,
) where {T<:Real}
    QuadratureVI{T}(ϵ, nGaussHermite, optimiser, false, clipping, 1, natural)
end


"""
    QuadratureSVI(nMinibatch::Int; ϵ::T=1e-5, nGaussHermite::Int=20, clipping=Inf, natural=true, optimiser=Momentum(0.0001))

Stochastic Variational Inference solver by approximating gradients via numerical integration via Gaussian Quadrature.
See [`QuadratureVI`](@ref) for a more detailed reference.

## Arguments 
-`nMinibatch::Integer` : Number of samples per mini-batches

## Keyword arguments
- `ϵ::T` : convergence criteria, which can be user defined
- `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.0001)`
"""
function QuadratureSVI(
    nMinibatch::Integer;
    ϵ::T = 1e-5,
    nGaussHermite::Integer = 100,
    optimiser = Momentum(1e-5),
    clipping::Real = 0.0,
    natural = true,
) where {T<:Real}
    QuadratureVI{T}(
        ϵ,
        nGaussHermite,
        optimiser,
        true,
        clipping,
        nMinibatch,
        natural,
    )
end

function tuple_inference(
    i::QuadratureVI{T},
    nLatent::Int,
    nFeatures:: Vector{<:Int},
    nSamples::Int,
    nMinibatch::Int,
    xview,
    yview
) where {T}
    return QuadratureVI{T}(
        conv_crit(i),
        isStochastic(i),
        i.nPoints,
        i.clipping,
        nFeatures,
        nSamples,
        nMinibatch,
        nLatent,
        i.vi_opt[1].optimiser,
        i.NaturalGradient,
        xview,
        yview,
    )
end

function expec_loglikelihood(
    l::AbstractLikelihood,
    i::QuadratureVI,
    y,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    mapreduce(apply_quad, :+, y, μ, diagΣ, i, l)
end

function apply_quad(
    y::Real,
    μ::Real,
    σ²::Real,
    i::QuadratureVI,
    l::AbstractLikelihood,
)
    xs = i.nodes * sqrt(σ²) .+ μ
    return dot(i.weights, loglikelihood.(Ref(l), y, xs))
    # return mapreduce((w, x) -> w * Distributions.loglikelihood(l, y, x), +, i.weights, xs)# loglikelihood.(l, y, x))
end

function grad_expectations!(
    m::AbstractGP{T,L,<:QuadratureVI},
) where {T,L}
    y = yview(m)
    for (gp, opt) in zip(m.f, get_opt(inference(m)))
        μ = mean_f(gp)
        Σ = var_f(gp)
        for i in 1:nMinibatch(inference(m))
            opt.ν[i], opt.λ[i] =
                grad_quad(likelihood(m), y[i], μ[i], Σ[i], inference(m))
        end
    end
end

# Compute the first and second derivative of the log-likelihood using the quadrature nodes
function grad_quad(
    l::AbstractLikelihood{T},
    y::Real,
    μ::Real,
    σ²::Real,
    i::AbstractInference,
) where {T<:Real}
    x = i.nodes * sqrt(max(σ², zero(T))) .+ μ
    Edloglike = dot(i.weights, ∇loglikehood.(l, y, x))
    Ed²loglike = dot(i.weights, hessloglikehood.(l, y, x))
    if i.clipping != 0
        return (
            abs(Edloglike) > i.clipping ? sign(Edloglike) * i.clipping :
            -Edloglike::T,
            abs(Ed²loglike) > i.clipping ? sign(Ed²loglike) * i.clipping :
            -Ed²loglike::T,
        )
    else
        return -Edloglike::T, Ed²loglike::T
    end
end
