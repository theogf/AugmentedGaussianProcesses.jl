"""Create the covariance matrix between the matrix X1 and X2 with the covariance function `kernel`"""
function kernelmatrix(X1::AbstractArray{T1},X2::AbstractArray{T2},kernel::Kernel) where {T1<:Real,T2<:Real}
    K = pairwise(getmetric(kernel),X1,X2,dims=1)
    v = getvariance(kernel)
    return lmul!(v,map!(kappa(kernel),K,K))
end

"""Compute the covariance matrix between the matrix X1 and X2 with the covariance function `kernel` in preallocated matrix K"""
function kernelmatrix!(K::AbstractArray{T1},X1::AbstractArray{T2},X2::AbstractArray{T3},kernel::Kernel) where {T1<:Real,T2<:Real,T3<:Real}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    pairwise!(K,getmetric(kernel),X1,X2,dims=1)
    v = getvariance(kernel)
    map!(kappa(kernel),K,K)
    return lmul!(v,K)
end

"""Compute the covariance matrix of the matrix X, optionally only compute the diagonal terms"""
function kernelmatrix(X::AbstractArray{T1},kernel::Kernel;diag::Bool=false) where {T1<:Real}
    if diag
        return kerneldiagmatrix(X,kernel)
    end
    K = pairwise(getmetric(kernel),X,dims=1)
    v = getvariance(kernel)
    # return v.*map(kappa(kernel),K)
    return v*map!(kappa(kernel),K,K)
end

"""Compute the covariance matrix of the matrix X in preallocated matrix K, optionally only compute the diagonal terms"""
function kernelmatrix!(K::AbstractArray{T1},X::AbstractArray{T2},kernel::Kernel; diag::Bool=false) where {T1<:Real,T2<:Real}
    if diag
        kerneldiagmatrix!(K,X,kernel)
    end
    (n1,n2) = size(K)
    @assert n1==size(X,1)
    @assert n1==n2
    pairwise!(K,getmetric(kernel),X,dims=1)
    v = getvariance(kernel)
    map!(kappa(kernel),K,K)
    return lmul!(v,K)
end

"""Compute only the diagonal elements of the covariance matrix"""
function kerneldiagmatrix(X::AbstractArray{T1},kernel::Kernel) where {T1<:Real}
    n = size(X,1)
    K = zeros(T1,n)
    v = getvariance(kernel)
    f = kappa(kernel)
    @simd for i in eachindex(K)
        @inbounds K[i] = f(evaluate(getmetric(kernel),X[i,:],X[i,:]))
    end
    return lmul!(v,K)
end

"""Compute only the diagonal elements of the covariance matrix in preallocated vector K"""
function kerneldiagmatrix!(K::AbstractVector{T1},X::AbstractArray{T2},kernel::Kernel) where {T1<:Real,T2<:Real}
    @assert size(K,1) == size(X,1)
    v = getvariance(kernel)
    f = kappa(kernel)
    @simd for i in eachindex(K)
        @inbounds K[i] = f(evaluate(getmetric(kernel),X[i,:],X[i,:]))
    end
    return lmul!(v,K)
end


"""Remapping of the gradients into a matrix with only 1 column and 1 row being non-zero"""
function CreateColumnRowMatrix(n::Int,iter::Int,gradient::AbstractVector{T1}) where {T1<:Real}
    K = zeros(T1,n,n)
    K[iter,:] .= gradient; K[:,iter] .= gradient;
    return K
end

"""Remapping of the gradients into a matrix with only 1 column being non-zero"""
function CreateColumnMatrix(n::Int,m::Int,iter::Int,gradient::AbstractVector{T1}) where {T1 <:Real}
    K = zeros(T1,n,m)
    K[:,iter] .= gradient;
    return K
end

"Compute the gradients given the inducing point locations, (general gradients are computed to be then remapped correctly)"
function computeIndPointsJ(model,iter::Int)
    Dnm = computeIndPointsJnm(model.kernel,model.inference.x,model.inducingPoints[iter,:],iter,model.Knm)
    Dmm = computeIndPointsJmm(model.kernel,model.inducingPoints,iter,model.Kmm)
    Jnm = zeros(model.nDim,model.nSamplesUsed,model.m)
    Jmm = zeros(model.nDim,model.m,model.m)
    @inbounds for i in 1:model.nDim
        Jnm[i,:,:] .= CreateColumnMatrix(model.nSamplesUsed,model.m,iter,Dnm[:,i])
        Jmm[i,:,:] .= CreateColumnRowMatrix(model.m,iter,Dmm[:,i])
    end
    return Jnm,Jmm
    #Return dim*K*K tensors for computing the gradient
end
