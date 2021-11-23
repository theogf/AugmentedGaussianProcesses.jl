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
mutable struct QuadratureVI{T} <: NumericalVI{T}
    n_points::Int # Number of points for the quadrature
    nodes::Vector{T} # Nodes locations
    weights::Vector{T} # Weights for each node
    clipping::T # Clipping value of the gradient
    ϵ::T #Convergence criteria
    n_iter::Integer #Number of steps performed
    stoch::Bool #Use of mini-batches
    batchsize::Int #Size of mini-batches
    ρ::T #Stochastic Coefficient
    NaturalGradient::Bool
    HyperParametersUpdated::Bool # Flag for updating kernel matrices
    vi_opt::NVIOptimizer

    function QuadratureVI{T}(
        ϵ::T,
        n_points::Integer,
        optimiser,
        Stochastic::Bool,
        clipping::Real,
        batchsize::Int,
        natural::Bool,
    ) where {T}
        gh = gausshermite(n_points)
        return new{T}(
            n_points,
            gh[1] .* sqrt2,
            gh[2] ./ sqrtπ,
            clipping,
            ϵ,
            0,
            Stochastic,
            batchsize,
            one(T),
            natural,
            true,
            NVIOptimizer(optimiser),
        )
    end
end

function QuadratureVI(;
    ϵ::T=1e-5,
    nGaussHermite::Integer=100,
    optimiser=Momentum(1e-5),
    clipping::Real=0.0,
    natural::Bool=true,
) where {T<:Real}
    return QuadratureVI{T}(ϵ, nGaussHermite, optimiser, false, clipping, 1, natural)
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
    ϵ::T=1e-5,
    nGaussHermite::Integer=100,
    optimiser=Momentum(1e-5),
    clipping::Real=0.0,
    natural=true,
) where {T<:Real}
    return QuadratureVI{T}(ϵ, nGaussHermite, optimiser, true, clipping, nMinibatch, natural)
end

function expec_loglikelihood(
    l::AbstractLikelihood, i::QuadratureVI, y, μ::AbstractVector, diagΣ::AbstractVector
)
    return mapreduce(+, y, μ, diagΣ) do y, μ, σ²
        apply_quad(i, y, μ, σ², l)
    end
end

function apply_quad(i::QuadratureVI, y::Real, μ::Real, σ²::Real, l::AbstractLikelihood)
    xs = i.nodes * sqrt(σ²) .+ μ
    return dot(i.weights, loglikelihood.(Ref(l), y, xs))
    # return mapreduce((w, x) -> w * Distributions.loglikelihood(l, y, x), +, i.weights, xs)# loglikelihood.(l, y, x))
end

function grad_expectations!(m::AbstractGPModel{T,L,<:QuadratureVI}, state, y) where {T,L}
    for (gp, opt_state, kernel_matrices) in zip(m.f, state.opt_state, state.kernel_matrices)
        μ = mean_f(gp, kernel_matrices)
        Σ = var_f(gp, kernel_matrices)
        for i in 1:batchsize(inference(m))
            opt_state.ν[i], opt_state.λ[i] = grad_quad(
                likelihood(m), y[i], μ[i], Σ[i], inference(m)
            )
        end
    end
end

# Compute the first and second derivative of the log-likelihood using the quadrature nodes
function grad_quad(l::AbstractLikelihood, y::Real, μ::Real, σ²::Real, i::AbstractInference)
    x = i.nodes * sqrt(max(σ², zero(σ²))) .+ μ
    Edloglike = dot(i.weights, ∇loglikehood.(l, y, x))
    Ed²loglike = dot(i.weights, hessloglikehood.(l, y, x))
    if i.clipping != 0
        return (
            abs(Edloglike) > i.clipping ? sign(Edloglike) * i.clipping : -Edloglike,
            abs(Ed²loglike) > i.clipping ? sign(Ed²loglike) * i.clipping : -Ed²loglike,
        )
    else
        return -Edloglike, Ed²loglike
    end
end
