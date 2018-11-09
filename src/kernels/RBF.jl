"""
    Radial Basis Function Kernel also called RBF or SE(Squared Exponential)
"""
struct RBFKernel{T<:AbstractFloat,KT<:KernelType} <: Kernel{T,KT}
    fields::KernelFields{T,KT}
    function RBFKernel{T,KT}(θ::Vector{T};variance::T=one(T),dim::Integer=0) where {T<:Real,KT<:KernelType}
        if KT == ARDKernel
            if length(θ)==1 && dim ==0
                error("You defined an ARD RBF kernel without precising the number of dimensions or giving a vector for the lengthscale                   Please set dim in your kernel initialization")
            elseif dim!=0 && (length(θ)!=dim && length(θ)!=1)
                @warn "You did not use the same dimension for your params and dim, using the first value of params for all dimensions"
                θ = ones(dim,T=T)*θ[1]
            elseif length(θ)==1 && dim!=0
                θ = ones(dim)*θ[1]
            end
            dim = length(θ)
            return new{T,ARDKernel}(KernelFields{T,ARDKernel}(
                                        "Radial Basis ",
                                        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound(zero(T)),NullBound{T}()) for _ in 1:dim]),dim,
                                        WeightedSqEuclidean(one(T)./(θ.^2))))
        else
            return new{T,IsoKernel}(KernelFields{T,IsoKernel}(
                                        "RBF ",
                                        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound(zero(T)),NullBound{T}())]),1,
                                        SqEuclidean()))
        end
    end
end

const SEKernel = RBFKernel

function RBFKernel(θ::T1=1.0;variance::T2=one(T1),dim::Integer=0,ARD::Bool=false) where {T1<:Real,T2<:Real}
    if ARD
        RBFKernel{floattype(T1,T2),ARDKernel}([θ],variance=variance,dim=dim)
    else
        RBFKernel{floattype(T1,T2),IsoKernel}([θ],variance=variance)
    end
 end

function RBFKernel(θ::Array{T1,1};variance::T2=one(T1),dim::Integer=0) where {T1<:Real,T2<:Real}
    RBFKernel{floattype(T1,T2),ARDKernel}(θ,variance=variance,dim=dim)
end



@inline rbfkernel(z::T, l::T) where {T<:Real} = exp(-0.5*z/(l^2))

@inline rbfkernel(z::Real) = exp(-0.5*z)

function kappa(k::RBFKernel{T,IsoKernel}) where {T<:Real,KT}
    return z->rbfkernel(z,getlengthscales(k))
end

function kappa(k::RBFKernel{T,ARDKernel}) where {T<:Real,KT}
    return z->rbfkernel(z)
end

function updateweights!(k::RBFKernel{T,KT},w::Vector{T}) where {T,KT}
    k.fields.metric.weights .= 1.0./(w.^2)
end

function computeIndPointsJmm(k::RBFKernel{T,KT},X::Matrix{T},iPoint::Integer,K::Symmetric{T,Matrix{T}}) where {T,KT}
    l2 = (getlengthscales(k)).^2; v = getvariance(k)
    return -v.*((X[iPoint,:]'.-X)./l2').*K[:,iPoint]
end

function computeIndPointsJnm(k::RBFKernel{T,KT},X::Matrix{T},x::Vector{T},iPoint::Integer,K::Matrix{T}) where {T,KT}
    l2 = (getlengthscales(k)).^2; v = getvariance(k)
    return -v.*((x'.-X)./l2').*K[:,iPoint]
end


################# Matrix derivatives for the RBF kernel###################################
"Return the kernel matrix derivative for the Iso RBFKernel"
function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X'); K = zero(P)
    map!(kappa(kernel),K,P)
    return Symmetric(lmul!(v./(l^3),P.*=K))
end

"""Return the matrix derivative for the Iso RBFKernel with the covariance matrix precomputed"""
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X')
    return Symmetric(lmul!(v./(l^3),P.*=K))
end

"Return the matrix derivative for the ARD RBFKernel"
function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X')
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:length(ls)]
    map!(kappa(kernel),K,K)
    return Symmetric.(map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls))
end

"""Return the matrix derivative for the Iso RBFKernel with the covariance matrix precomputed"""
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:length(ls)]
    return Symmetric.(map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls))
end

########## DERIVATIVE MATRICES FOR TWO MATRICES #######

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X',Y'); K = zero(P);
    map!(kappa(kernel),K,P)
    return lmul!(v./(l^3),P.*=K)
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X',Y')
    return lmul!(v./(l^3),P.*=K)
end

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X',Y')
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]') for i in 1:length(ls)]
    map!(kappa(kernel),K,K)
    return map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls)
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]') for i in 1:length(ls)]
    return map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls)
end


############ DIAGONAL DERIVATIVES ###################

function kernelderivativediagmatrix(X::Array{T,N},kernel::RBFKernel{T,IsoKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n)
    return P
end

function kernelderivativediagmatrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return [P for _ in 1:kernel.fields.Ndim]
end
