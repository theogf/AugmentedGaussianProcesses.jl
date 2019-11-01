mutable struct _VGP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    σ_k::Float64
    μ₀::PriorMean{T}
    K::PDMat{T,Matrix{T}}
    opt_ρ::Union{Optimizer,Nothing}
    opt_σ::Union{Optimizer,Nothing}
end

function _VGP{T}(dim::Int,kernel::Kernel,mean::PriorMean,σ_k::Real,opt_ρ::Union{Optimizer,Nothing}=Adam(α=0.01),opt_σ=deepcopy(opt_ρ)) where {T<:Real}
    _VGP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            zeros(T,dim),
            Symmetric(Matrix{T}(-0.5*I,dim,dim)),
            deepcopy(kernel),
            σ_k,
            deepcopy(mean),
            PDMat(Matrix{T}(I,dim,dim)),
            deepcopy(opt_σ),
            deepcopy(opt_ρ))
end

mutable struct _SVGP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    σ_k::Float64
    μ₀::PriorMean{T}
    Z::InducingPoints
    K::PDMat{T,Matrix{T}}
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    opt_ρ::Union{Optimizer,Nothing}
    opt_σ::Union{Optimizer,Nothing}
end

function _SVGP{T}(  dim::Int,nSamplesUsed::Int,
                    Z::Union{AbstractMatrix,InducingPoints},
                    kernel::Kernel,mean::PriorMean,σ_k::Real,
                    opt_ρ::Union{Optimizer,Nothing}=Adam(α=0.01),opt_σ::Union{Optimizer,Nothing}=opt_ρ
                 ) where {T<:Real}
    _SVGP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            zeros(T,dim),
            Symmetric(Matrix{T}(-0.5*I,dim,dim)),
            deepcopy(kernel),
            σ_k,
            deepcopy(mean),
            deepcopy(Z),
            PDMat(Matrix{T}(I,dim,dim)),
            Matrix{T}(undef,nSamplesUsed,dim),
            Matrix{T}(undef,nSamplesUsed,dim),
            Vector{T}(undef,nSamplesUsed),
            deepcopy(opt_ρ),
            deepcopy(opt_σ))
end

mean_f(model::AbstractGP) = mean_f.(model.f)

mean_f(gp::_VGP) = gp.μ
mean_f(gp::_SVGP) = gp.κ*gp.μ

diag_cov_f(model::AbstractGP) = diag_cov_f.(model.f)
diag_cov_f(gp::_VGP) = diag(gp.Σ)
diag_cov_f(gp::_SVGP) = opt_diag(gp.κ*gp.Σ,gp.κ) + gp.K̃

compute_K!(gp::_VGP,X::AbstractMatrix,jitter) = gp.K = PDMat(gp.σ_k*(kernelmatrix(gp.kernel,X,obsdim=1)+jitter*I))
compute_K!(gp::_SVGP,jitter) = gp.K = PDMat(gp.σ_k*(kernelmatrix(gp.kernel,gp.Z.Z,obsdim=1)+jitter*I))

function compute_κ!(gp::_SVGP,X,jitter)
    gp.Knm .= gp.σ_k * kernelmatrix(gp.kernel,X,gp.Z.Z,obsdim=1)
    gp.κ .= gp.Knm / gp.K
    gp.K̃ .= gp.σ_k * (kerneldiagmatrix(gp.kernel,X,obsdim=1) .+ jitter) - opt_diag(gp.κ,gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
