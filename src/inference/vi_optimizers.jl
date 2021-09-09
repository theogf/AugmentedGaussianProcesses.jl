abstract type InferenceOptimizer{T} end

## Abstract type for analytical optimiser (closed forms are known)
abstract type AOptimizer{T} <: InferenceOptimizer{T} end
## Abstract type for numerical optimiser (need for numerical integration)
abstract type NOptimizer{T} <: InferenceOptimizer{T} end

# Analytic VI Optimizer
mutable struct AVIOptimizer{T<:Real,O,Tstate1,Tstate2} <: AOptimizer{T}
    optimiser::O #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    η₁_state::Tstate1
    η₂_state::Tstate2
    function AVIOptimizer{T}(n::Int, opt::O) where {T,O}
        ∇η₁ = zeros(T, n)
        ∇η₂ = zeros(T, n, n)
        η₁_state = Optimisers.init(opt, ∇η₁)
        η₂_state = Optimisers.init(opt, ∇η₂)
        return new{T,O,typeof(η₁_state),typeof(η₂_state)}(opt, ∇η₁, ∇η₂, η₁_state, η₂_state)
    end
end

# Numerical VI Optimizer
mutable struct NVIOptimizer{T<:Real,O} <: NOptimizer{T}
    optimiser::O #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    ν::Vector{T} #Derivative -<dv/dx>_qn
    λ::Vector{T} #Derivative  <d²V/dx²>_qm
    L::LowerTriangular{T,Matrix{T}}
    function NVIOptimizer{T}(n::Int, b::Int, opt::O) where {T,O}
        return new{T,O}(
            opt,
            zeros(T, n),
            zeros(T, n, n),
            zeros(T, b),
            zeros(T, b),
            LowerTriangular(diagm(sqrt(0.5) * ones(T, n))),
        )
    end
end

# Sampling Optimizer, does not contain anyting, just placeholder for sampling
mutable struct SOptimizer{T<:Real,O} <: AOptimizer{T}
    optimiser::O
end

function SOptimizer{T}(opt::O) where {T,O}
    return SOptimizer{T,O}(opt)
end
