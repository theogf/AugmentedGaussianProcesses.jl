"""
    Create the kernel matrix from the training data or the correlation matrix one of set of vectors
"""
function kernelmatrix(X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(metric(kernel),X1',X2')
    @inbounds for i in eachindex(K)
        K[i] = compute!(kernel,K[i])
    end
    return getvalue(kernel.variance)*K
end

function kernelmatrix!(K::Array{T,N},X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    pairwise!(K,metric(kernel),X1',X2')
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,K[i])
    end
    K *= getvalue(kernel.variance)
    return K
end

function kernelmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(metric(kernel),X')
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,K[i])
    end
    return getvalue(kernel.variance)*K
end


function kernelmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X,1)
    @assert n1==n2
    pairwise!(K,metric(kernel),X')
    @inbounds for i in in eachindex(K)
        K[i] = compute(kernel,K[i])
    end
    K *= getvalue(kernel.variance)
    return K
end

function kerneldiagmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(X,1)
    K = zeros(T,n)
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,evaluate(metric(kernel),X[i,:],X[i,:]))
    end
    return getvalue(kernel.variance)*K
end

function kerneldiagmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(K,1)
    @assert n == size(X,1)
    @inbounds for i in 1:n
        K[i] = compute(kernel,evaluate(metric(kernel),X[i,:],X[i,:]))
    end
    return getvalue(kernel.variance)*K
end

############# DERIVATIVE MATRICES ##############

function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(SqEuclidean(),X')
    K = zero(P)
    v = getvalue(kernel.variance)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,P[i]/(l^2))
    end
    return v./(l^3).*P.*K
end

"When K has already been computed"
function kernelderivativematrix_K(X::Array{T,N},K::Symmetric{T,Array{T,N}},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(SqEuclidean(),X')
    v = getvalue(kernel.variance)
    ls = getvalue(kernel.lengthscales[1])
    return v./(l^3).*P.*K
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

"""
    Compute the gradients using a gradient function and matrices Js
"""
function compute_hyperparameter_gradient(k::KernelSum{T},gradient_function::Function,variance::Bool,Js::Vector{Any},Kindex::Int64,index::Int64) where T
    return [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
end

function compute_hyperparameter_gradient(k::KernelProduct{T},gradient_function::Function,variance::Bool,Js::Vector{Any},Kindex::Int64,index::Int64) where T
    gradients = [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
    if variance
        push!(gradients,[gradient_function(broadcast(x->x[end][1],Js),Kindex,index)])
    end
    return gradients
end

function compute_hyperparameter_gradient(k::Kernel{T},gradient_function::Function,variance::Bool,Js::Array{Array{T2,1} where T2,1},Kindex::Int64,index::Int64) where T
    return [gradient_function(broadcast(x->x[j],Js),Kindex,index) for j in 1:k.Ndim]
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
