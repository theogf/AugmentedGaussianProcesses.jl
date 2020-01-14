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

mutable struct _SVGP{T} <: Abstract_GP{T}
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
            kernel,
            [σ_k],
            deepcopy(mean),
            deepcopy(Z),
            PDMat(Matrix{T}(I,dim,dim)),
            Matrix{T}(undef,nSamplesUsed,dim),
            Matrix{T}(undef,nSamplesUsed,dim),
            Vector{T}(undef,nSamplesUsed),
            deepcopy(opt))
end

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

@traitfn mean_f(gp::T) where {T<:Abstract_GP;IsFull{T}} = gp.μ
@traitfn mean_f(gp::T) where {T<:Abstract_GP;!IsFull{T}} = gp.κ*gp.μ

diag_cov_f(model::AbstractGP) = diag_cov_f.(model.f)

diag_cov_f(gp::_GP{T}) where {T} = zeros(T,gp.dim)
diag_cov_f(gp::_VGP) = diag(gp.Σ)
diag_cov_f(gp::_SVGP) = opt_diag(gp.κ*gp.Σ,gp.κ) + gp.K̃

@traitfn compute_K!(gp::T,X::AbstractMatrix,jitter::Real) where {T<:Abstract_GP;IsFull{T}} = gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,X,obsdim=1)+jitter*I))
compute_K!(gp::_SVGP,jitter::Real) = gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,gp.Z,obsdim=1)+jitter*I))

function compute_κ!(gp::TGP,X::AbstractMatrix,jitter::Real) where {TGP<:Abstract_GP}
    gp.Knm .= first(gp.σ_k) * kernelmatrix(gp.kernel, X, gp.Z, obsdim=1)
    gp.κ .= gp.Knm / gp.K.mat
    gp.K̃ .= first(gp.σ_k) * (kerneldiagmatrix(gp.kernel, X, obsdim=1) .+ jitter) - opt_diag(gp.κ,gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
