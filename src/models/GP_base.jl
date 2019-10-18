struct _VGP{T,V}
    dim::Int
    μ::V
    Σ::PDMat{T,Matrix{T}}
    η₁::V
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    μ₀::PriorMean{T}
end

struct _SVGP{T,V}
    dim::Int
    μ::V
    Σ::PDMat{T,Matrix{T}}
    η₁::V
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    μ₀::PriorMean{T}
    Z::InducingPoints
end
