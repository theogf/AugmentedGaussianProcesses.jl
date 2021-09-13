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

# Sampling Optimizer, does not contain anyting, just a place-holder for sampling
struct SOptimizer{O} <: AOptimizer
    optimiser::O
end