################# DERIVATIVES FOR THE RBF KERNEL ###################################


function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X'); K = zero(P)
    map!(kappa(kernel),K,P)
    return Symmetric(v./(l^3).*P.*K)
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X')
    return Symmetric(v./(l^3).*P.*K)
end

function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X')
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:length(ls)]
    map!(kappa(kernel),K,K)
    return Symmetric.(map((pi,l)->v./(l^3).*pi.*K,Pi,ls))
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    Pi = [pairwise(SqEuclidean(),X[:,i]') for i in 1:length(ls)]
    return Symmetric.(map((pi,l)->v./(l^3).*pi.*K,Pi,ls))
end

########## DERIVATIVE MATRICES FOR TWO MATRICES #######

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X',Y'); K = zero(P);
    map!(kappa(kernel),K,P)
    return v./(l^3).*P.*K
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    v = getvariance(kernel); l = getlengthscales(kernel)
    P = pairwise(SqEuclidean(),X',Y')
    return v./(l^3).*P.*K
end

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    K = pairwise(getmetric(kernel),X',Y')
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]') for i in 1:length(ls)]
    map!(kappa(kernel),K,K)
    return map((pi,l)->v./(l^3).*pi.*K,Pi,ls)
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvariance(kernel); ls = getlengthscales(kernel)
    Pi = [pairwise(SqEuclidean(),X[:,i]',Y[:,i]') for i in 1:length(ls)]
    return map((pi,l)->v./(l^3).*pi.*K,Pi,ls)
end


############ DIAGONAL DERIVATIVES ###################

function kernelderivativediagmatrix(X::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n)
    return P
end

function kernelderivativediagmatrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    n = size(X,1); P = zeros(T,n);
    return [P for _ in 1:kernel.fields.Ndim]
end
