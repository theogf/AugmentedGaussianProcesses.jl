abstract type AbstractGPPrior{T<:Real,K<:Kernel,Tmean<:PriorMean} end

kernel(gp::AbstractGPPrior) = gp.kernel
Distributions.mean(gp::AbstractGPPrior) = gp.μ₀
Distributions.mean(gp::AbstractGPPrior, X::AbstractVector) = gp.μ₀(X)
Distributions.cov(gp::AbstractGPPrior) = gp.K

mutable struct GPPrior{T,K<:Kernel,Tmean<:PriorMean} <: AbstractGPPrior{T,K,Tmean}
    kernel::K
    μ₀::Tmean
    K::PDMat{T,Matrix{T}}
end

mutable struct TPrior{T,K<:Kernel,Tmean<:PriorMean} <: AbstractGPPrior{T,K,Tmean}
    kernel::K
    μ₀::Tmean
    K::PDMat{T,Matrix{T}}
    ν::T # Number of degrees of freedom
    l²::T # Expectation of ||L^{-1}(f-μ⁰)||₂²
    χ::T  # Expectation of σ
end
