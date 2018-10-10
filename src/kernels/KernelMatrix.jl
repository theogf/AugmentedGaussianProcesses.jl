"Create the kernel matrix from the training data or the correlation matrix one of set of vectors"
function kernelmatrix(X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(getmetric(kernel),X1',X2')
    v = getvariance(kernel)
    return v.*map!(kappa(kernel),K,K)
end

function kernelmatrix!(K::Array{T,N},X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    pairwise!(K,getmetric(kernel),X1',X2')
    v = getvariance(kernel)
    map!(kappa(kernel),K,K)
    return K .*= v
end

function kernelmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(getmetric(kernel),X')
    v = getvariance(kernel)
    return v.*map!(kappa(kernel),K,K)
end


function kernelmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X,1)
    @assert n1==n2
    pairwise!(K,getmetric(kernel),X')
    v = getvariance(kernel)
    return v.*map!(kappa(kernel),K,K)
end

function kerneldiagmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(X,1)
    K = zeros(T,n)
    v = getvariance(kernel)
    f = kappa(kernel)
    @simd for i in eachindex(K)
        @inbounds K[i] = f(evaluate(getmetric(kernel),X[i,:],X[i,:]))
    end
    return v.*K
end

function kerneldiagmatrix!(K::Vector{T},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(K,1)
    @assert n == size(X,1)
    v = getvariance(kernel)
    f = kappa(kernel)
    @simd for i in eachindex(K)
        @inbounds K[i] = f(evaluate(getmetric(kernel),X[i,:],X[i,:]))
    end
    K .*= v
    return K
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

#Compute the gradients given the inducing point locations
function computeIndPointsJ(model,iter)
    Dnm = zeros(model.nSamplesUsed,model.nDim)
    Dmm = zeros(model.m,model.nDim)
    Jnm = zeros(model.nDim,model.nSamplesUsed,model.m)
    Jmm = zeros(model.nDim,model.m,model.m)
    #Compute the gradients given every data point
    for i in 1:model.nSamplesUsed
        Dnm[i,:] = compute_point_deriv(model.kernel,model.X[model.MBIndices[i],:],model.inducingPoints[iter,:])
    end
    for i in 1:model.m
        Dmm[i,:] = compute_point_deriv(model.kernel,model.inducingPoints[iter,:],model.inducingPoints[i,:])
    end
    for i in 1:model.nDim
        Jnm[i,:,:] = CreateColumnMatrix(model.nSamplesUsed,model.m,iter,Dnm[:,i])
        Jmm[i,:,:] = CreateColumnRowMatrix(model.m,iter,Dmm[:,i])
    end
    return Jnm,Jmm
    #Return dim * K*K tensors for computing the gradient
end
