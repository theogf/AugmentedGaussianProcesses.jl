"""
    Matern 3/2 Kernel
"""
struct Matern3_2Kernel{T<:AbstractFloat,KT<:KernelType} <: Kernel{T,KT}
    fields::KernelFields{T,KT}
    function Matern3_2Kernel{T,KT}(θ::Vector{T};variance::T=one(T),dim::Integer=0) where {T<:AbstractFloat,KT<:KernelType}
        if KT == ARDKernel
            if length(θ)==1 && dim ==0
                error("You defined an ARD Matern3_2 kernel without precising the number of dimensions or giving a vector for the lengthscale                   Please set dim in your kernel initialization")
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
                                        Euclidean(one(T)./(θ.^2))))
        else
            return new{T,IsoKernel}(KernelFields{T,IsoKernel}(
                                        "Matern3_2",
                                        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound(zero(T)),NullBound{T}())]),1,
                                        Euclidean()))
        end
    end
end

function Matern3_2Kernel(θ::T1=1.0;variance::T2=one(T1),dim::Integer=0,ARD::Bool=false) where {T1<:Real,T2<:Real}
    if ARD
        Matern3_2Kernel{floattype(T1,T2),ARDKernel}([θ],variance=variance,dim=dim)
    else
        Matern3_2Kernel{floattype(T1,T2),IsoKernel}([θ],variance=variance)
    end
 end

function Matern3_2Kernel(θ::Array{T1,1};variance::T2=one(T1),dim::Integer=0) where {T1<:Real,T2<:Real}
    Matern3_2Kernel{floattype(T1,T2),ARDKernel}(θ,variance=variance,dim=dim)
end



@inline matern3_2kernel(z::T, l::T) where {T<:Real} = (one(T)+sqrt(3)*z/l)*exp(-sqrt(3)*z/l)

@inline matern3_2kernel(z::T) where {T<:Real}= (one(T)+sqrt(3)*z)*exp(-sqrt(3)*z)

function kappa(k::Matern3_2Kernel{T,IsoKernel}) where {T<:Real,KT}
    return z->matern3_2kernel(z,getlengthscales(k))
end

function kappa(k::Matern3_2Kernel{T,ARDKernel}) where {T<:Real,KT}
    return z->matern3_2kernel(z)
end

function updateweights!(k::Matern3_2Kernel{T,KT},w::Vector{T}) where {T,KT}
    k.fields.metric.weights .= 1.0./(w.^2)
end

function computeIndPointsJmm(k::Matern3_2Kernel{T,KT},X::Matrix{T},iPoint::Integer,K::Symmetric{T,Matrix{T}}) where {T,KT}
    l = (getlengthscales(k))
    P = pairwise(getmetric(k),X')
    if KT == IsoKernel
        return -3.0*((X[iPoint,:]'.-X)./(l^2) .* exp.(-sqrt(3.0)/l.*P[iPoint,:]))
    else
        return -3.0*((X[iPoint,:]'.-X)./(l.^2)' .* exp.(-sqrt(3.0).*P[iPoint,:]))
    end
end

function computeIndPointsJnm(k::Matern3_2Kernel{T,KT},X::Matrix{T},x::Vector{T},iPoint::Integer,K::Matrix{T}) where {T,KT}
    l2 = (getlengthscales(k)).^2
    P = pairwise(getmetric(k),x',X')
    if KT == IsoKernel
        return -3.0*((x'.-X)./(l^2) .* exp.(-sqrt(3.0)/l.*P))
    else
        return -3.0*((x'.-X)./(l.^2)' .* exp.(-sqrt(3.0).*P)) #TODO check if it works
    end
end


################# Matrix derivatives for the Matern3_2 kernel###################################
"""Return the derivatives of Knn for the Iso Matern3_2Kernel"""
function kernelderivativematrix(X::Array{T,N},kernel::Matern3_2Kernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(Euclidean(),X');
    return Symmetric(lmul!(3.0*v./(l^3),P.^2 .*exp.(-sqrt(3.0)./l.*P)))
end

"""Return the derivatives of Knn for the Iso Matern3_2Kernel with Knn precomputed"""
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::Matern3_2Kernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(Euclidean(),X')
    return Symmetric(lmul!(3.0*v./(l^3),P.^2 .*exp.(-sqrt(3.0)./l.*P)))
end

"Return the derivatives of Knn for the ARD Matern3_2Kernel"
function kernelderivativematrix(X::Array{T,N},kernel::Matern3_2Kernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X')
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:length(ls)]
    return Symmetric.(map((pi,l)->lmul!(3.0*v./(l^3),pi.^2 .*exp.(-sqrt(3.0).*K),Pi,ls)))
end

"""Return the derivatives of Knn for the ARD Matern3_2Kernel with Knn precomputed"""
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::Matern3_2Kernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K .= pairwise(getmetric(kernel),X')
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:length(ls)]
    return Symmetric.(map((pi,l)->lmul!(3.0*v./(l^3),pi.^2 .*exp.(-sqrt(3.0)*K),Pi,ls)))
end

########## DERIVATIVE MATRICES FOR TWO MATRICES #######
"""Return the derivatives of Knm for the Iso Matern3_2Kernel"""
function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::Matern3_2Kernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(Euclidean(),X',Y');
    return lmul!(3.0*v./(l^3),P.^2 .* exp.(-sqrt(3.0)./l.*P))
end

"""Return the derivatives of Knn for the Iso Matern3_2Kernel with Knm precomputed"""
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::Matern3_2Kernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(Euclidean(),X',Y')
    return lmul!(3.0*v./(l^3),P.^2 .*exp.(-sqrt(3.0./l.*P)))
end

"""Return the derivatives of Knm for the ARD Matern3_2Kernel"""
function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::Matern3_2Kernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X',Y')
    Pi = [pairwise(Euclidean(),X[:,i]',Y[:,i]') for i in 1:length(ls)]
    return map((pi,l)->lmul!(3.0*v./(l^3),pi.^2 .*exp.(-sqrt(3.0).*K)),Pi,ls)
end

"""Return the derivatives of Knn for the ARD Matern3_2Kernel with Knm precomputed"""
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::Matern3_2Kernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K .= pairwise(getmetric(kernel),X',Y')
    Pi = [pairwise(Euclidean(),X[:,i]',Y[:,i]') for i in 1:length(ls)]
    return map((pi,l)->lmul!(3.0*v./(l^3),pi.^2 .* exp.(-sqrt(3.0).*K)),Pi,ls)
end

############ DIAGONAL DERIVATIVES ###################


"""Return the derivate of the diagonal covariance matrix for the Iso Matern3_2Kernel"""
function kernelderivativediagmatrix(X::Array{T,N},kernel::Matern3_2Kernel{T,IsoKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n)
    return P
end

"""Return the derivate of the diagonal covariance matrix for the Iso Matern3_2Kernel"""
function kernelderivativediagmatrix(X::Array{T,N},kernel::Matern3_2Kernel{T,ARDKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return [P for _ in 1:kernel.fields.Ndim]
end
