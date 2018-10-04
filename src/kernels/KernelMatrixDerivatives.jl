
function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(SqEuclidean(),X')
    K = zero(P)
    v = getvalue(kernel.variance)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,P[i]/(l^2))
    end
    return Symmetric(v./(l^3).*P.*K)
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(SqEuclidean(),X')
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales[1])
    return Symmetric(v./(l^3).*P.*K)
end

function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales)
    K = pairwise(metric(kernel),X')
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:kernel.Ndim]
    @inbounds for j in eachindex(K)
        K[j] = compute(kernel,K[j])
    end
    return Symmetric.(map((pi,l)->v./(l^3).*pi.*K,Pi,ls))
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales)
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:kernel.Ndim]
    return Symmetric.(map((pi,l)->v./(l^3).*pi.*K,Pi,ls))
end

########## DERIVATIVE MATRICES FOR TWO MATRICES #######

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(SqEuclidean(),X',Y')
    K = zero(P)
    v = getvalue(kernel.variance)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,P[i]/(l^2))
    end
    return v./(l^3).*P.*K
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(SqEuclidean(),X',Y')
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales[1])
    return v./(l^3).*P.*K
end

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales)
    K = pairwise(metric(kernel),X',Y')
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]') for i in 1:kernel.Ndim]
    @inbounds for j in eachindex(K)
        K[j] = compute(kernel,K[j])
    end
    return map((pi,l)->v./(l^3).*pi.*K,Pi,ls)
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales)
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]') for i in 1:kernel.Ndim]
    return map((pi,l)->v./(l^3).*pi.*K,Pi,ls)
end


############ DIAGONAL DERIVATIVES ###################


function kernelderivativediagmatrix(X::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n)
    return P
end


function kernelderivativediagmatrix_K(X::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return P
end


function kernelderivativediagmatrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return [P for _ in 1:kernel.Ndim]
end

"When K has already been computed"
function kernelderivativediagmatrix_K(X::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return [P for _ in 1:kernel.Ndim]
end
