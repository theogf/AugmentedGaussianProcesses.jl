"Radial Basis Function Kernel also called RBF or SE(Squared Exponential)"
struct RBFKernel{T<:Real,KT<:KernelType} <: Kernel{T,KT}
    fields::KernelFields{T,KT}
    function RBFKernel{T,KT}(θ::Vector{T};variance=one(T),dim::Integer=0) where {T<:Real,KT<:KernelType}
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
                                        "Radial Basis",
                                        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound(zero(T)),NullBound{T}()) for _ in 1:dim]),dim,
                                        WeightedSqEuclidean(one(T)./(θ.^2))))
        else
            return new{T,IsoKernel}(KernelFields{T,IsoKernel}(
                                        "Radial Basis",
                                        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound(zero(T)),NullBound{T}())]),1,
                                        SqEuclidean()))
        end
    end
end

const SEKernel = RBFKernel

"""`RBFKernel(θ::T1=1.0;variance::T2=one(T1),dim::Integer=0,ARD::Bool=false)`

Create a Squared Exponential/Radial Basis Function kernel

``variance\\exp\\left(-\\frac{||x-x'||}{2\\theta^2}\\right)``

The kernel can be made with ARD (Automatic Relevance Determination) by giving `θ` as a vector or setting `ARD` as true, and in both cases setting the `dim` variable appropriately
"""
function RBFKernel(θ::T1=1.0;variance::T2=one(T1),dim::Integer=0,ARD::Bool=false) where {T1<:Real,T2<:Real}
    if ARD
        RBFKernel{T1,ARDKernel}([θ],variance=variance,dim=dim)
    else
        RBFKernel{T1,IsoKernel}([θ],variance=variance)
    end
 end

function RBFKernel(θ::Array{T1,1};variance::T2=one(T1),dim::Integer=0) where {T1<:Real,T2<:Real}
    RBFKernel{T1,ARDKernel}(θ,variance=variance,dim=dim)
end



@inline rbfkernel(z::Real, l::Real) = exp(-0.5*z/(l^2))

@inline rbfkernel(z::Real) = exp(-0.5*z)

function kappa(k::RBFKernel{T,IsoKernel}) where {T<:Real}
    return z->rbfkernel(z,getlengthscales(k))
end

function kappa(k::RBFKernel{T,ARDKernel}) where {T<:Real}
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
function kernelderivativematrix(X::AbstractArray{T},kernel::RBFKernel{T,IsoKernel}) where {T<:Real}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X,dims=1); K = zero(P)
    map!(kappa(kernel),K,P)
    return Symmetric(lmul!(v./(l^3),P.*=K))
end

"""Return the matrix derivative for the Iso RBFKernel with the covariance matrix precomputed"""
function kernelderivativematrix_K(X::AbstractArray{T},K::Symmetric{T,AbstractArray{T,2}},kernel::RBFKernel{T,IsoKernel}) where {T<:Real}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X,dims=1)
    return Symmetric(lmul!(v./(l^3),P.*=K))
end

"Return the matrix derivative for the ARD RBFKernel"
function kernelderivativematrix(X::AbstractArray{T},kernel::RBFKernel{T,ARDKernel}) where {T<:Real}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X,dims=1)
    Pi = [pairwise(SqEuclidean(), X[:,i]',dims=2) for i in 1:length(ls)]
    map!(kappa(kernel),K,K)
    return Symmetric.(map!((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,Pi,ls))
end

"""Return the matrix derivative for the Iso RBFKernel with the covariance matrix precomputed"""
function kernelderivativematrix_K(X::AbstractArray{T},K::Symmetric{T,AbstractArray{T,2}},kernel::RBFKernel{T,ARDKernel}) where {T<:Real}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    Pi = [pairwise(SqEuclidean(),X[:,i],dims=1) for i in 1:length(ls)]
    return Symmetric.(map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls))
end

########## DERIVATIVE MATRICES FOR TWO MATRICES #######

function kernelderivativematrix(X::AbstractArray{T},Y::AbstractArray{T},kernel::RBFKernel{T,IsoKernel}) where {T<:Real}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X,Y,dims=1); K = zero(P);
    map!(kappa(kernel),K,P)
    return lmul!(v./(l^3),P.*=K)
end

"When K has already been computed"
function kernelderivativematrix_K(X::AbstractArray{T},Y::AbstractArray{T},K::AbstractArray{T},kernel::RBFKernel{T,IsoKernel}) where {T}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X,Y,dims=1)
    return lmul!(v./(l^3),P.*=K)
end

function kernelderivativematrix(X::AbstractArray{T},Y::AbstractArray{T},kernel::RBFKernel{T,ARDKernel}) where {T}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X,Y,dims=1)
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]',dims=2) for i in 1:length(ls)]
    map!(kappa(kernel),K,K)
    return map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls)
end

"When K has already been computed"
function kernelderivativematrix_K(X::AbstractArray{T},Y::AbstractArray{T},K::AbstractArray{T},kernel::RBFKernel{T,ARDKernel}) where {T}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    Pi = [pairwise(SqEuclidean(),X[:,i],Y[:,i],dims=1) for i in 1:length(ls)]
    return map((pi,l)->lmul!(v./(l^3),pi.*=K),Pi,ls)
end


############ DIAGONAL DERIVATIVES ###################

function kernelderivativediagmatrix(X::AbstractArray{T},kernel::RBFKernel{T,IsoKernel}) where {T<:Real}
    return zeros(T,size(X,1))
end

function kernelderivativediagmatrix(X::AbstractArray{T},kernel::RBFKernel{T,ARDKernel}) where {T}
    return [zeros(T,size(X,1)) for _ in 1:kernel.fields.Ndim]
end
