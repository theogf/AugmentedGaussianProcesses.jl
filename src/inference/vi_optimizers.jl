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
struct NVIOptimizer{O} <: NOptimizer
    optimiser::O # Optimiser for stochastic updates
end

# Sampling Optimizer, does not contain anyting, just a place-holder for sampling
struct SOptimizer{O} <: AOptimizer
    optimiser::O
end

Base.length(::InferenceOptimizer) = 1

Base.iterate(i::InferenceOptimizer) = (i, nothing)
Base.iterate(::InferenceOptimizer, ::Any) = nothing
