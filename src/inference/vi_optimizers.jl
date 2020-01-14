abstract type InferenceOptimizer{T} end

## Abstract type for analytical optimizer (closed forms are known)
abstract type AOptimizer{T} <: InferenceOptimizer{T} end
## Abstract type for numerical optimizer (need for numerical integration)
abstract type NOptimizer{T} <: InferenceOptimizer{T} end

mutable struct AVIOptimizer{T<:Real} <: AOptimizer{T}
    optimizer #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    function AVIOptimizer{T}(n::Int,opt) where {T}
        new{T}(opt,zeros(T,n),zeros(T,n,n))
    end
end

mutable struct NVIOptimizer{T<:Real} <: NOptimizer{T}
    optimizer #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    ν::Vector{T} #Derivative -<dv/dx>_qn
    λ::Vector{T} #Derivative  <d²V/dx²>_qm
    L::LowerTriangular{T,Matrix{T}}
    function NVIOptimizer{T}(n::Int,b::Int,opt) where {T}
        new{T}(opt,zeros(T,n),zeros(T,n,n),zeros(T,b),zeros(T,b),LowerTriangular(diagm(sqrt(0.5)*ones(T,n))))
    end
end

mutable struct SOptimizer{T<:Real} <: AOptimizer{T}
    optimizer
end
