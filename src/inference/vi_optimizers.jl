abstract type InferenceOptimizer end

## Abstract type for analytical optimiser (closed forms are known)
abstract type AOptimizer <: InferenceOptimizer end
## Abstract type for numerical optimiser (need for numerical integration)
abstract type NOptimizer <: InferenceOptimizer end

# Analytic VI Optimizer
struct AVIOptimizer{O} <: AOptimizer
    optimiser::O # Optimiser for stochastic updates
end

# Numerical VI Optimizer
struct NVIOptimizer{T<:Real,O} <: NOptimizer
    optimiser::O #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    ν::Vector{T} #Derivative -<dv/dx>_qn
    λ::Vector{T} #Derivative  <d²V/dx²>_qm
    L::LowerTriangular{T,Matrix{T}}
end

# Sampling Optimizer, does not contain anyting, just a place-holder for sampling
struct SOptimizer{O} <: AOptimizer
    optimiser::O
end

Base.length(::InferenceOptimizer) = 1

Base.iterate(i::InferenceOptimizer) = (i, nothing)
Base.iterate(::InferenceOptimizer, ::Any) = nothing
