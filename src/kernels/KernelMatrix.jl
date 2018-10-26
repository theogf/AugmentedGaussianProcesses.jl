"Create the kernel matrix from the training data or the correlation matrix one of set of vectors"
function kernelmatrix(X1::Array{T,N1},X2::Array{T,N2},kernel::Kernel{T,KT}) where {T,N1,N2,KT}
    K = pairwise(getmetric(kernel),X1',X2')
    v = getvariance(kernel)
    return lmul!(v,map!(kappa(kernel),K,K))
end

function kernelmatrix!(K::Array{T,N2},X1::Array{T,N1},X2::Array{T,N2},kernel::Kernel{T,KT}) where {T,N1,N2,KT}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    pairwise!(K,getmetric(kernel),X1',X2')
    v = getvariance(kernel)
    map!(kappa(kernel),K,K)
    return lmul!(v,K)
end

function kernelmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(getmetric(kernel),X')
    v = getvariance(kernel)
    return lmul!(v,map!(kappa(kernel),K,K))
end


function kernelmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X,1)
    @assert n1==n2
    pairwise!(K,getmetric(kernel),X')
    v = getvariance(kernel)
    map!(kappa(kernel),K,K)
    return lmul!(v,K)
end

function kerneldiagmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(X,1)
    K = zeros(T,n)
    v = getvariance(kernel)
    f = kappa(kernel)
    @simd for i in eachindex(K)
        @inbounds K[i] = f(evaluate(getmetric(kernel),X[i,:],X[i,:]))
    end
    return lmul!(v,K)
end

function kerneldiagmatrix!(K::Vector{T},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(K,1)
    @assert n == size(X,1)
    v = getvariance(kernel)
    f = kappa(kernel)
    @simd for i in eachindex(K)
        @inbounds K[i] = f(evaluate(getmetric(kernel),X[i,:],X[i,:]))
    end
    return lmul!(v,K)
end


"""
    Compute derivative matrices given the data points
"""
function CreateColumnRowMatrix(n,iter,gradient)
    K = zeros(n,n)
    K[iter,:] = gradient; K[:,iter] = gradient;
    return K
end

function CreateColumnMatrix(n,m,iter,gradient)
    K = zeros(n,m)
    K[:,iter] = gradient;
    return K
end

# function computeIndPointsJ(model,iPoint)
#     Jmm = [zeros(Float64,model.m,model.m) for _ in 1:model.nDim]
#     Jnm = [zeros(Float64,model.nSamplesUsed,model.m) for _ in 1:model.nDim]
#     map!(computeJmm,Jmm,)
#
#
# end


"Compute the gradients given the inducing point locations"
function computeIndPointsJ(model,iter)
    Dnm = computeIndPointsJnm(model.kernel,model.X[model.MBIndices,:],model.inducingPoints[iter,:],iter,model.Knm)
    Dmm = computeIndPointsJmm(model.kernel,model.inducingPoints,iter,model.Kmm)
    Jnm = zeros(model.nDim,model.nSamplesUsed,model.m)
    Jmm = zeros(model.nDim,model.m,model.m)
    for i in 1:model.nDim
        Jnm[i,:,:] = CreateColumnMatrix(model.nSamplesUsed,model.m,iter,Dnm[:,i])
        Jmm[i,:,:] = CreateColumnRowMatrix(model.m,iter,Dmm[:,i])
    end
    return Jnm,Jmm
    #Return dim * K*K tensors for computing the gradient
end
