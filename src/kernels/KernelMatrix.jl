"Create the kernel matrix from the training data or the correlation matrix one of set of vectors"
function kernelmatrix(X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(metric(kernel),X1',X2')
    v = getvalue(kernel.variance)
    return map!(x->v*compute(kernel,x),K,K)
end

function kernelmatrix!(K::Array{T,N},X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    pairwise!(K,metric(kernel),X1',X2')
    v = getvalue(kernel.variance)
    return map!(x->v*compute(kernel,x),K,K)
end

function kernelmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(metric(kernel),X')
    v = getvalue(kernel.variance)
    return map!(x->v*compute(kernel,x),K,K)
end


function kernelmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X,1)
    @assert n1==n2
    pairwise!(K,metric(kernel),X')
    v = getvalue(kernel.variance)
    return map!(x->v*compute(kernel,x),K,K)
end

function kerneldiagmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(X,1)
    K = zeros(T,n)
    v = getvalue(kernel.variance)
    @inbounds for i in eachindex(K)
        K[i] = v*compute(kernel,evaluate(metric(kernel),X[i,:],X[i,:]))
    end
    return K
end

function kerneldiagmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(K,1)
    @assert n == size(X,1)
    v = getvalue(kernel.variance)
    @inbounds for i in 1:n
        K[i] = v*compute(kernel,evaluate(metric(kernel),X[i,:],X[i,:]))
    end
    return K
end

############# DERIVATIVE MATRICES ##############


"""
    Compute the gradients using a gradient function and matrices Js
"""
function compute_hyperparameter_gradient(k::KernelSum{T},gradient_function::Function,Js::Vector{Any},Kindex::Int64,index::Int64) where T
    return [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
end

function compute_hyperparameter_gradient(k::KernelProduct{T},gradient_function::Function,Js::Vector{Any},Kindex::Int64,index::Int64) where T
    gradients = [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
    if variance
        push!(gradients,[gradient_function(broadcast(x->x[end][1],Js),Kindex,index)])
    end
    return gradients
end

function compute_hyperparameter_gradient(k::Kernel{T},gradient_function::Function,Js::Vector{Array{T2,1} where T2},Kindex::Int64,index::Int64) where T
    return map(gradient_function,Js[1],Js[2],Js[3],Kindex*ones(Int64,k.Ndim),index*ones(Int64,k.Ndim))
    # return map((jmm,jnm,jnn)->gradient_function([jmm,jnm,jnn],Kindex,index),Js[1],Js[2],Js[3])
end

function compute_hyperparameter_gradient(k::Kernel{T,PlainKernel},gradient_function::Function,Js::Array{AbstractArray{Float64,N} where N,1},Kindex::Int64,index::Int64) where T
    return gradient_function(Js[1],Js[2],Js[3],Kindex,index)
    # return map((jmm,jnm,jnn)->gradient_function([jmm,jnm,jnn],Kindex,index),Js[1],Js[2],Js[3])
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
