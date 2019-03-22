"""
    Matern Kernel
"""
struct MaternKernel{T<:Real,KT<:KernelType} <: Kernel{T,KT}
    fields::KernelFields{T,KT}
    ν::T
    function MaternKernel{T,KT}(θ::Vector{T},ν::T,variance::T=one(T);dim::Integer=0) where {T<:Real,KT<:KernelType}
        @assert ν > 0 "ν should be bigger than 0!"
        if KT == ARDKernel
            if length(θ)==1 && dim ==0
                error("You defined an ARD Matern kernel without precising the number of dimensions or giving a vector for the lengthscale                   Please set dim in your kernel initialization")
            elseif dim!=0 && (length(θ)!=dim && length(θ)!=1)
                @warn "You did not use the same dimension for your params and dim, using the first value of params for all dimensions"
                θ = ones(dim,T=T)*θ[1]
            elseif length(θ)==1 && dim!=0
                θ = ones(dim)*θ[1]
            end
            dim = length(θ)
            return new{T,ARDKernel}(KernelFields{T,ARDKernel}(
                                        "General Matern",
                                        HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound(zero(T)),NullBound{T}()) for _ in 1:dim]),dim,
                                        WeightedEuclidean(one(T)./(θ.^2))),ν)
        else
            return new{T,IsoKernel}(KernelFields{T,IsoKernel}(
                                        "General Matern",
                                        HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
                                        HyperParameters{T}(θ,[interval(OpenBound{T}(zero(T)),NullBound{T}())]),1,
                                        Euclidean()),ν)
        end
    end
end


#TODO Implement specific kernels for matern kernel
function MaternKernel(θ::T1=1.0,ν::T2=2.5;variance::T3=one(T1),dim::Integer=0,ARD::Bool=false) where {T1<:Real,T2<:Real,T3<:Real}
    maxtype = floattype(T1,T2,T3)
    if ARD
        # if ν≈0.5
            # ExponentialKernel{floattype(T1,T2),ARDKernel}([θ],ν,variance=variance,dim=dim)
        # elseif ν≈1.5
            # Matern3_2Kernel{floattype(T1,T2),ARDKernel}([θ],ν,variance=variance,dim=dim)
        # elseif ν≈2.5
            # Matern5_2Kernel{floattype(T1,T2),ARDKernel}([θ],ν,variance=variance,dim=dim)
        # else
            MaternKernel{maxtype,ARDKernel}([θ],maxtype(ν),maxtype(variance),dim=dim)
        # end
    else
        # if ν≈0.5
            # ExponentialKernel{floattype(T1,T2),IsoKernel}([θ],ν,variance=variance)
            # elseif ν≈1.5
            # Matern3_2Kernel{floattype(T1,T2),IsoKernel}([θ],ν,variance=variance)
        # elseif ν≈2.5
            # Matern5_2Kernel{floattype(T1,T2),ARDKernel}([θ],ν,variance=variance)
        # else
            MaternKernel{maxtype,IsoKernel}([θ],maxtype(ν),maxtype(variance))
        # end
    end
 end

function MaternKernel(θ::Array{T1,1},ν::T2=2.5;variance::T3=one(T1),dim::Integer=0) where {T1<:Real,T2<:Real,T3<:Real}
    maxtype = floattype(T1,T2,T3)
    MaternKernel{maxtype,ARDKernel}(θ,maxtype(ν),maxtype(variance),dim=dim)
end



@inline function maternkernel(z::T, l::T, ν::T) where {T<:Real}
    v = sqrt(2.0*ν)*z/l
    v = v < eps(T) ? eps(T) : v
    return 2.0 * (v/2.0)^ν * besselk(ν,v) / gamma(ν)
end

@inline function maternkernel(z::T, ν::T) where {T<:Real}
    v = sqrt(2.0*ν)*z
    v = v < eps(T) ? eps(T) : v
    return 2.0 * (v/2.0)^ν * besselk(ν,v) / gamma(ν)
end

function kappa(k::MaternKernel{T,IsoKernel}) where {T<:Real,KT}
    return z->maternkernel(z,getlengthscales(k),k.ν)
end

function kappa(k::MaternKernel{T,ARDKernel}) where {T<:Real,KT}
    return z->maternkernel(z,k.ν)
end

function updateweights!(k::MaternKernel{T,KT},w::Vector{T}) where {T,KT}
    k.fields.metric.weights .= 1.0./(w.^2)
end

function computeIndPointsJmm(k::MaternKernel{T,KT},X::Matrix{T},iPoint::Integer,K::Symmetric{T,Matrix{T}}) where {T,KT}
    l = (getlengthscales(k)); v = getvariance(k); ν = k.ν; C = 2^(1.0-ν)/gamma(ν)
    P = sqrt(2.0*ν).*pairwise(getmetric(k),X,dims=1)
    P .= ifelse.(P.<eps(T),eps(T),P)
    if KT == IsoKernel
        return -(2.0*C*v*ν) .* (X[iPoint,:]'.-X)./(l^2) .* (P[iPoint,:]./l).^(ν-1.0) .* besselk.(ν-1.0,P[iPoint,:]./l)
    elseif KT == ARDKernel
        return -(2.0*C*v*ν) .* (X[iPoint,:]'.-X)./(l.^2)' .* (P[iPoint,:]).^(ν-1.0) .* besselk.(ν-1.0,P[iPoint,:])
    end
end

function computeIndPointsJnm(k::MaternKernel{T,KT},X::Matrix{T},x::Vector{T},iPoint::Integer,K::Matrix{T}) where {T,KT}
    l = (getlengthscales(k)); v = getvariance(k); ν = k.ν; C = 2^(1.0-ν)/gamma(ν)
    P = sqrt(2.0*ν).*pairwise(getmetric(k),x[:,:],X,dims=1)[:]
    P .= ifelse.(P.<eps(T),eps(T),P)
    if KT == IsoKernel
        return -(2.0*C*v*ν).* (x'.-X)./(l^2) .* (P./l).^(ν-1.0) .* besselk.(ν-1.0,P./l)
    else
        return -(2.0*C*v*ν).* (x'.-X)./(l.^2)' .* (P).^(ν-1.0) .* besselk.(ν-1.0,P)
    end
end


################# Matrix derivatives for the Matern kernel###################################
"""Return the derivatives of Knn for the Iso MaternKernel"""
function kernelderivativematrix(X::Array{T},kernel::MaternKernel{T,IsoKernel}) where {T<:Real}
    v = getvariance(kernel); l = getlengthscales(kernel); ν=kernel.ν; C = (2^(1.0-ν))/gamma(ν)
    P = (sqrt(2.0*ν)/l).*pairwise(Euclidean(),X,dims=1);
    P .= ifelse.(P.<eps(T),eps(T),P)
    return Symmetric(lmul!(C*v/l, P.^(ν+1.0) .* besselk.(ν-1.0,P)))
end

"""Return the derivatives of Knn for the Iso MaternKernel with Knn precomputed"""
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::MaternKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel); ν=kernel.ν;  C = (2^(1.0-ν))/gamma(ν)
    P = (sqrt(2.0*ν)/l).*pairwise(Euclidean(),X,dims=1);
    P .= ifelse.(P.<eps(T),eps(T),P)
    return Symmetric(lmul!(C*v/l,P.^(ν+1.0) .* besselk.(ν-1.0,P)))
end

"Return the derivatives of Knn for the ARD MaternKernel"
function kernelderivativematrix(X::Array{T,N},kernel::MaternKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel); ν = kernel.ν; C = (2^(1.0-ν))/gamma(ν)
    P = (sqrt(2.0*ν)).*pairwise(getmetric(kernel),X,dims=1) #d/ρ
    P .= ifelse.(P.<eps(T),eps(T),P)
    Pi = [pairwise(SqEuclidean(),X[:,i]',dims=2) for i in 1:length(ls)] # (x_i-x_i')
    return Symmetric.(map((pi,l)->lmul!(2.0*C*v*ν/(l^3),pi .* P.^(ν-1.0) .* besselk.(ν-1.0,P)),Pi,ls))
end

"""Return the derivatives of Knn for the ARD MaternKernel with Knn precomputed"""
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::MaternKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel); ν = kernel.ν; C = (2^(1.0-ν))/gamma(ν)
    P = (sqrt(2.0*ν)).*pairwise(getmetric(kernel),X,dims=1) #d/ρ
    P .= ifelse.(P.<eps(T),eps(T),P)
    Pi = [pairwise(SqEuclidean(),X[:,i]',dims=2) for i in 1:length(ls)] # (x_i-x_i')
    return Symmetric.(map((pi,l)->lmul!(2.0*C*v*ν/(l^3),pi .* P.^(ν-1.0) .* besselk.(ν-1.0,P)),Pi,ls))
end

########## DERIVATIVE MATRICES FOR TWO MATRICES #######
"""Return the derivatives of Knm for the Iso MaternKernel"""
function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::MaternKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel); ν=kernel.ν;  C = 2^(1.0-ν)/gamma(ν)
    P = (sqrt(2.0*ν)/l)*pairwise(Euclidean(),X,Y,dims=1);
    P .= ifelse.(P.<eps(T),eps(T),P)
    return lmul!(C*v/l,P.^(ν+1.0).*besselk.(ν-1.0,P))
end

"""Return the derivatives of Knn for the Iso MaternKernel with Knm precomputed"""
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::MaternKernel{T,IsoKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel); ν=kernel.ν; C = 2^(1.0-ν)/gamma(ν)
    P = (sqrt(2.0*ν)/l)*pairwise(Euclidean(),X,Y,dims=1);
    P .= ifelse.(P.<eps(T),eps(T),P)
    return lmul!(C*v/l,P.^(ν+1.0).*besselk.(ν-1.0,P))
end

"""Return the derivatives of Knm for the ARD MaternKernel"""
function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::MaternKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel); ν = kernel.ν; C = 2^(1.0-ν)/gamma(ν)
    P = (sqrt(2.0*ν)).*pairwise(getmetric(kernel),X,Y,dims=1)
    P .= ifelse.(P.<eps(T),eps(T),P)
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]',dims=2) for i in 1:length(ls)] # (x_i-x_i')
    return map((pi,l)->lmul!(2.0*C*v*ν/(l^3),pi .* P.^(ν-1.0).*besselk.(ν-1.0,P)),Pi,ls)
end

"""Return the derivatives of Knn for the ARD MaternKernel with Knm precomputed"""
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::MaternKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel); ν = kernel.ν; C = 2^(1.0-ν)/gamma(ν)
    P = (sqrt(2.0*ν)).*pairwise(getmetric(kernel),X,Y,dims=1)
    P .= ifelse.(P.<eps(T),eps(T),P)
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]',dims=2) for i in 1:length(ls)] # (x_i-x_i')
    return map((pi,l)->lmul!(2.0*C*v*ν/(l^3),pi .* P.^(ν-1.0).*besselk.(ν-1.0,P)),Pi,ls)
end

############ DIAGONAL DERIVATIVES ###################


"""Return the derivate of the diagonal covariance matrix for the Iso MaternKernel"""
function kernelderivativediagmatrix(X::Array{T,N},kernel::MaternKernel{T,IsoKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n)
    return P
end

"""Return the derivate of the diagonal covariance matrix for the Iso MaternKernel"""
function kernelderivativediagmatrix(X::Array{T,N},kernel::MaternKernel{T,ARDKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return [P for _ in 1:kernel.fields.Ndim]
end
