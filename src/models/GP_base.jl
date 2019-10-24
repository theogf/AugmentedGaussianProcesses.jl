struct _VGP{T} <: Abstract_GP
    dim::Int
    μ::Vector{T}
    Σ::PDMat{T,Matrix{T}}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::GPKernel
    μ₀::PriorMean{T}
    K::PDMat{T,Matrix{T}}
end

struct _SVGP{T} <: Abstract_GP
    dim::Int
    μ::Vector{T}
    Σ::PDMat{T,Matrix{T}}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    μ₀::PriorMean{T}
    Z::InducingPoints
    K::PDMat{T,Matrix{T}}
    κ::Matrix{T}
    K̃::Vector{T}
end
