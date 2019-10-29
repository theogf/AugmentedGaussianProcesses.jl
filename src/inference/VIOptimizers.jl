mutable struct AVIOptimizer{T<:Real} <: AbstractVIOptimizer{T}
    optimizer::Optimizer #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    AVIOptimizer{T}(n::Int,opt::Optimizer) where {T}
        new{T}(opt,zeros(T,n),zeros(T,n,n))
    end
end

mutable struct NVIOptimizer{T<:Real} <: AbstractVIOptimizer{T}
    optimizer::Optimizer #Learning rate for stochastic updates
    ∇η₁::Vector{T}
    ∇η₂::Matrix{T}
    ν::Vector{T} #Derivative -<dv/dx>_qn
    λ::Vector{T} #Derivative  <d²V/dx²>_qm
    AVIOptimizer{T}(n::Int,opt::Optimizer) where {T}
        new{T}(opt,zeros(T,n),zeros(T,n,n),zeros(T,n),zeros(T,n))
    end
end
