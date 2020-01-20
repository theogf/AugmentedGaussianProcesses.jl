## Gaussian Process

mutable struct _GP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    kernel::Kernel
    σ_k::Vector{T}
    μ₀::PriorMean{T}
    K::PDMat{T,Matrix{T}}
    opt
end

function _GP{T}(dim::Int,kernel::Kernel,mean::PriorMean,σ_k::Real,opt) where {T<:Real}
    _GP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            kernel,
            [σ_k],
            deepcopy(mean),
            PDMat(Matrix{T}(I,dim,dim)),
            deepcopy(opt))
end

@traitimpl IsFull{_GP}

## Variational Gaussian Process

mutable struct _VGP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    σ_k::Vector{T}
    μ₀::PriorMean{T}
    K::PDMat{T,Matrix{T}}
    opt
end

function _VGP{T}(dim::Int,kernel::Kernel,mean::PriorMean,σ_k::Real,opt) where {T<:Real}
    _VGP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            zeros(T,dim),
            Symmetric(Matrix{T}(-0.5*I,dim,dim)),
            kernel,
            [σ_k],
            deepcopy(mean),
            PDMat(Matrix{T}(I,dim,dim)),
            deepcopy(opt))
end

@traitimpl IsFull{_VGP}

## Sparse Variational Gaussian Process

mutable struct _SVGP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    σ_k::Vector{T}
    μ₀::PriorMean{T}
    Z::FixedInducingPoints
    K::PDMat{T,Matrix{T}}
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    opt
end

function _SVGP{T}(  dim::Int,nSamplesUsed::Int,
                    Z::InducingPoints,
                    kernel::Kernel,mean::PriorMean,σ_k::Real,
                    opt
                 ) where {T<:Real}
    _SVGP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            zeros(T,dim),
            Symmetric(Matrix{T}(-0.5*I,dim,dim)),
            deepcopy(kernel),
            [σ_k],
            deepcopy(mean),
            deepcopy(Z),
            PDMat(Matrix{T}(I,dim,dim)),
            Matrix{T}(undef,nSamplesUsed,dim),
            Matrix{T}(undef,nSamplesUsed,dim),
            Vector{T}(undef,nSamplesUsed),
            deepcopy(opt))
end

@traitimpl IsSparse{_SVGP}

## Monte-Carlo Gaussian Process

mutable struct _MCGP{T} <: Abstract_GP{T}
    dim::Int
    f::Vector{T}
    kernel::Kernel
    σ_k::Vector{T}
    μ₀::PriorMean{T}
    K::PDMat{T,Matrix{T}}
end

function _MCGP{T}(dim::Int,kernel::Kernel,mean::PriorMean,σ_k::Real) where {T<:Real}
    _MCGP{T}(dim,
            zeros(T,dim),
            wrapper(kernel,nothing),
            [σ_k],
            deepcopy(mean),
            PDMat(Matrix{T}(I,dim,dim)))
end

@traitimpl IsFull{_MCGP}

## Online Sparse Variational Process

mutable struct _OSVGP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    σ_k::Vector{T}
    μ₀::PriorMean{T}
    Z::InducingPoints
    K::PDMat{T,Matrix{T}}
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    Zupdated::Bool
    opt
    Zₐ::Matrix{T}
    Kab::Matrix{T}
    κₐ::Matrix{T}
    K̃ₐ::Matrix{T}
    invDₐ::Symmetric{T,Matrix{T}}
    prevη₁::Vector{T}
    prev𝓛ₐ::T
end

function _OSVGP{T}(dim::Int,nSamplesUsed::Int,
                    Z::InducingPoints,
                    kernel::Kernel,mean::PriorMean,σ_k::Real,
                    opt
                 ) where {T<:Real}
    _OSVGP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            zeros(T,dim),
            Symmetric(Matrix{T}(-0.5*I,dim,dim)),
            kernel,
            [σ_k],
            deepcopy(mean),
            deepcopy(Z),
            PDMat(Matrix{T}(I,dim,dim)),
            Matrix{T}(undef,nSamplesUsed,dim),
            Matrix{T}(undef,nSamplesUsed,dim),
            Vector{T}(undef,nSamplesUsed),
            false,
            deepcopy(opt),
            Matrix{T}(I,dim,dim),
            Matrix{T}(I,dim,dim),
            Matrix{T}(I,dim,dim),
            Matrix{T}(I,dim,dim),
            Symmetric(Matrix{T}(I,dim,dim)),
            zeros(T,dim),
            zero(T))
end

@traitimpl IsSparse{_OSVGP}

## Variational Student-T Process

mutable struct _VStP{T} <: Abstract_GP{T}
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
    kernel::Kernel
    σ_k::Vector{T}
    μ₀::PriorMean{T}
    K::PDMat{T,Matrix{T}}
    invL::LowerTriangular{T,Matrix{T}}
    ν::T # Number of degrees of freedom
    l²::T # Expectation of ||L^{-1}(f-μ⁰)||₂²
    χ::T  # Expectation of σ
    opt_kernel
    opt_σ::OptorNothing
end

function _VStP{T}(ν::Real,dim::Int,kernel::Kernel,mean::PriorMean,σ_k::Real,opt_kernel,opt_σ::OptorNothing) where {T<:Real}
    _VGP{T}(dim,
            zeros(T,dim),
            Matrix{T}(I,dim,dim),
            zeros(T,dim),
            Symmetric(Matrix{T}(-0.5*I,dim,dim)),
            kernel,
            [σ_k],
            deepcopy(mean),
            PDMat(Matrix{T}(I,dim,dim)),
            LowerTriangular(Matrix{T}(I,dim,dim)),
            ν,
            rand(T),
            rand(T),
            deepcopy(opt_kernel))
end

@traitimpl IsFull{_VStP}

### Functions

mean_f(model::AbstractGP) = mean_f.(model.f)

@traitfn mean_f(gp::T) where {T<:Abstract_GP;!IsSparse{T}} = gp.μ
@traitfn mean_f(gp::T) where {T<:Abstract_GP;IsSparse{T}} = gp.κ*gp.μ

diag_cov_f(model::AbstractGP) = diag_cov_f.(model.f)

diag_cov_f(gp::_GP{T}) where {T} = zeros(T,gp.dim)
diag_cov_f(gp::_VGP) = diag(gp.Σ)
diag_cov_f(gp::_SVGP) = opt_diag(gp.κ*gp.Σ,gp.κ) + gp.K̃
diag_cov_f(gp::_OSVGP) = opt_diag(gp.κ*gp.Σ,gp.κ) + gp.K̃

get_Z(gp::Abstract_GP) = gp.Z.Z

@traitfn compute_K!(gp::T,X::AbstractMatrix,jitt::Real) where {T<:Abstract_GP;!IsSparse{T}} = gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,X,obsdim=1)+jitt*I))
@traitfn compute_K!(gp::T,jitt::Real) where {T<:Abstract_GP;IsSparse{T}} = gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,gp.Z,obsdim=1)+jitt*I))

function compute_κ!(gp::_SVGP,X::AbstractMatrix,jitt::Real)
    gp.Knm .= first(gp.σ_k) * kernelmatrix(gp.kernel, X, gp.Z, obsdim=1)
    gp.κ .= gp.Knm / gp.K.mat
    gp.K̃ .= first(gp.σ_k) * (kerneldiagmatrix(gp.kernel, X, obsdim=1) .+ jitt) - opt_diag(gp.κ,gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end

function compute_κ!(gp::_OSVGP, X::AbstractMatrix, jitt::Real)
    # Covariance with the model at t-1
    gp.Kab = kernelmatrix(gp.kernel, gp.Zₐ, gp.Z, obsdim=1)
    gp.κₐ = gp.Kab / gp.K.mat
    Kₐ = Symmetric(first(gp.σ_k)*(kernelmatrix(gp.kernel, gp.Zₐ, obsdim=1)+jitt*I))
    gp.K̃ₐ = Kₐ - gp.κₐ*transpose(gp.Kab)

    # Covariance with a new batch
    gp.Knm = first(gp.σ_k) * kernelmatrix(gp.kernel, X, gp.Z.Z, obsdim=1)
    gp.κ = gp.Knm / gp.K.mat
    gp.K̃ = first(gp.σ_k) * (kerneldiagmatrix(gp.kernel, X, obsdim=1) .+ jitt) - opt_diag(gp.κ,gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
