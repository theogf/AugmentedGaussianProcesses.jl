abstract type AbstractGPPrior{K<:Kernel,Tmean<:PriorMean} end# <: AbstractGPs.AbstractGPModel end

kernel(gp::AbstractGPPrior) = gp.kernel
setkernel!(gp::AbstractGPPrior, k::Kernel) = gp.kernel = k
Distributions.mean(gp::AbstractGPPrior) = gp.μ₀
Distributions.mean(gp::AbstractGPPrior, X::AbstractVector) = gp.μ₀(X)
setmean!(gp::AbstractGPPrior, μ₀::PriorMean) = gp.μ₀ = μ₀
Distributions.cov(gp::AbstractGPPrior) = gp.K

mutable struct GPPrior{K<:Kernel,Tmean<:PriorMean} <: AbstractGPPrior{K,Tmean}
    kernel::K
    μ₀::Tmean
end

mutable struct TPrior{K<:Kernel,Tmean<:PriorMean,T} <: AbstractGPPrior{K,Tmean}
    kernel::K
    μ₀::Tmean
    ν::T # Number of degrees of freedom
    l²::T # Expectation of ||L^{-1}(f-μ⁰)||₂² (L2 norm)
    χ::T  # Expectation of σ
end
