abstract type InferenceOptimizer{T} end

## Abstract type for analytical optimiser (closed forms are known)
abstract type AOptimizer{T} <: InferenceOptimizer{T} end
## Abstract type for numerical optimiser (need for numerical integration)
abstract type NOptimizer{T} <: InferenceOptimizer{T} end

mutable struct AVIOptimizer{T<:Real,O} <: AOptimizer{T}
    optimiser::O #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    function AVIOptimizer{T}(n::Int,opt::O) where {T,O}
        new{T,O}(opt,zeros(T,n),zeros(T,n,n))
    end
end

mutable struct NVIOptimizer{T<:Real,O} <: NOptimizer{T}
    optimiser::O #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    ν::Vector{T} #Derivative -<dv/dx>_qn
    λ::Vector{T} #Derivative  <d²V/dx²>_qm
    L::LowerTriangular{T,Matrix{T}}
    function NVIOptimizer{T}(n::Int,b::Int,opt::O) where {T,O}
        new{T,O}(opt,zeros(T,n),zeros(T,n,n),zeros(T,b),zeros(T,b),LowerTriangular(diagm(sqrt(0.5)*ones(T,n))))
    end
end

mutable struct SOptimizer{T<:Real,O} <: AOptimizer{T}
    optimiser::O
end

function SOptimizer{T}(opt::O) where {T,O}
    SOptimizer{T,O}(opt)
end
